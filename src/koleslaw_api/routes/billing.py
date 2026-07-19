import json
import logging
import uuid

import stripe
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from koleslaw_api.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])


@router.post("/webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
    """Stripe webhook — the ONLY billing write path (stripe-billing-plan.md §3).

    No session/key auth: the Stripe signature is the authentication. Handles
    exactly two events; everything else is acknowledged and ignored so Stripe
    doesn't retry. Both handlers are idempotent under re-delivery.
    """
    settings = request.app.state.settings
    webhook_secret = settings.stripe_webhook_secret.get_secret_value() if settings.stripe_webhook_secret else ""
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Billing is not configured")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        stripe.Webhook.construct_event(payload, signature, webhook_secret)
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature") from None

    # construct_event above is the signature check; field access uses the
    # plain JSON so it can't break on StripeObject API changes (v15 dropped
    # dict-style .get()).
    event = json.loads(payload)
    event_type = event["type"]
    obj = event["data"]["object"]
    db_session_factory = request.app.state.db_session_factory

    if event_type == "checkout.session.completed":
        # client_reference_id is set by the portal's Payment Link URL (= user id)
        user_ref = obj.get("client_reference_id")
        try:
            user_id = uuid.UUID(user_ref) if user_ref else None
        except ValueError:
            user_id = None
        if user_id is None:
            logger.warning("checkout.session.completed without usable client_reference_id (session %s)", obj.get("id"))
            return {"status": "ignored"}

        async with db_session_factory() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if user is None:
                logger.warning("checkout.session.completed for unknown user %s", user_id)
                return {"status": "ignored"}
            user.plan = "pro"
            user.stripe_customer_id = obj.get("customer")
            user.stripe_subscription_id = obj.get("subscription")
            await session.commit()

        logger.info("user %s upgraded to pro (subscription %s)", user_id, obj.get("subscription"))
        return {"status": "ok"}

    if event_type == "customer.subscription.deleted":
        # Covers explicit cancellation AND Stripe's retry-then-cancel on
        # payment failure — the reason invoice.payment_failed is unhandled.
        subscription_id = obj.get("id")
        async with db_session_factory() as session:
            result = await session.execute(select(User).where(User.stripe_subscription_id == subscription_id))
            user = result.scalar_one_or_none()
            if user is None:
                logger.info("subscription.deleted for unknown subscription %s", subscription_id)
                return {"status": "ignored"}
            user.plan = "free"
            user.stripe_subscription_id = None
            await session.commit()

        logger.info("subscription %s deleted — user downgraded to free", subscription_id)
        return {"status": "ok"}

    logger.info("ignoring stripe event %s", event_type)
    return {"status": "ignored"}

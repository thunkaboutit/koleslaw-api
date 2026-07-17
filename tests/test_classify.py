from koleslaw_api.prompts.system import classify_enhancement_type


class TestClassifyShortPrompts:
    """Prompts under 15 words should classify as expand."""

    def test_very_short(self):
        assert classify_enhancement_type("write a function") == "expand"

    def test_single_word(self):
        assert classify_enhancement_type("hello") == "expand"

    def test_near_boundary(self):
        # 14 words — still short
        prompt = "write a Python function that takes a list and returns unique values from it"
        assert len(prompt.split()) == 14
        assert classify_enhancement_type(prompt) == "expand"


class TestClassifyMediumPrompts:
    """Prompts between 15–80 words should classify as refine."""

    def test_medium_prompt(self):
        prompt = (
            "I need a REST API endpoint that accepts a JSON payload with user "
            "registration data, validates the email format and password strength, "
            "stores the user in a PostgreSQL database, and returns a JWT token"
        )
        word_count = len(prompt.split())
        assert 15 <= word_count <= 80
        assert classify_enhancement_type(prompt) == "refine"

    def test_at_lower_boundary(self):
        # Exactly 15 words
        prompt = "build a web application that allows users to upload files and share them with others"
        assert len(prompt.split()) == 15
        assert classify_enhancement_type(prompt) == "refine"


class TestClassifyLongUnstructuredPrompts:
    """Long prompts (> 80 words) without structure markers should classify as structure."""

    def test_long_rambling(self):
        prompt = (
            "I want to build a system that handles real-time data processing "
            "from multiple IoT sensors and it needs to aggregate the data and "
            "store it in a time series database and also provide a dashboard "
            "for monitoring and alerting and the system should scale to handle "
            "thousands of sensors and the data format varies between sensor "
            "types and some send JSON and others send binary protobuf messages "
            "and we need to normalize everything into a common schema and "
            "there should be anomaly detection running on the streams and "
            "alerts should go to Slack and PagerDuty and we also want historical "
            "replay capability for debugging and the whole thing needs to run "
            "on Kubernetes with proper health checks and graceful shutdown"
        )
        assert len(prompt.split()) > 80
        assert classify_enhancement_type(prompt) == "structure"


class TestClassifyLongStructuredPrompts:
    """Long prompts with structure markers should classify as refine, not structure."""

    def test_long_with_bullet_points(self):
        prompt = (
            "Build a notification service with the following requirements:\n"
            "- Support email, SMS, and push notifications\n"
            "- Allow users to set notification preferences\n"
            "- Implement rate limiting per channel\n"
            "- Provide delivery status tracking\n"
            "- Support template-based messages with variable substitution\n"
            "- Handle retry logic for failed deliveries\n"
            "- Log all notification events for audit purposes\n"
            "- Support scheduled notifications with cron expressions\n"
            "- Implement priority levels for urgent vs normal messages\n"
            "- Provide an API for third-party integrations\n"
            "- Support batch sending for marketing campaigns\n"
            "- Include unsubscribe handling for compliance"
        )
        assert len(prompt.split()) > 80
        assert classify_enhancement_type(prompt) == "refine"

    def test_long_with_numbered_list(self):
        prompt = (
            "Design a caching layer for our application. Here are the requirements: "
            "1. Support both Redis and in-memory backends "
            "2. Implement cache-aside pattern with configurable TTL "
            "3. Support cache invalidation by key pattern "
            "4. Add metrics for hit rate and latency "
            "5. Handle serialization for complex objects "
            "6. Support distributed locking for cache stampede prevention "
            "7. Implement graceful degradation when Redis is unavailable "
            "8. Add support for cache warming on startup "
            "9. Implement per-tenant cache isolation "
            "10. Support cache compression for large values "
            "11. Add circuit breaker for Redis connection failures "
            "12. Provide cache statistics endpoint for monitoring"
        )
        assert len(prompt.split()) > 80
        assert classify_enhancement_type(prompt) == "refine"

    def test_long_with_markdown_headers(self):
        prompt = (
            "# Authentication System\n"
            "We need a new auth system for our platform.\n\n"
            "## Requirements\n"
            "Support OAuth2, SAML, and local credentials. "
            "Must integrate with our existing user database and directory services. "
            "Session management with configurable expiry and refresh tokens. "
            "Multi-factor authentication support with TOTP and WebAuthn.\n\n"
            "## Security\n"
            "All tokens must be encrypted at rest using AES-256. "
            "Implement rate limiting on login attempts with exponential backoff. "
            "Support IP allowlisting for admin accounts and service accounts. "
            "Audit logging for all auth events with tamper-proof storage.\n\n"
            "## Performance\n"
            "Token validation under 10ms at the 99th percentile. "
            "Support 10k concurrent sessions across multiple regions. "
            "Cache active sessions in Redis with automatic failover."
        )
        assert len(prompt.split()) > 80
        assert classify_enhancement_type(prompt) == "refine"


class TestExplicitTypePassthrough:
    """When the user specifies a type explicitly, auto-detection is not used."""

    def test_explicit_types_bypass_classification(self):
        from koleslaw_api.schemas.enhance import EnhanceRequest
        from koleslaw_api.services.llm import LLMService

        short_prompt = "write a function"
        for explicit_type in ("expand", "refine", "structure"):
            req = EnhanceRequest(prompt=short_prompt, enhancement_type=explicit_type)
            assert LLMService.resolve_enhancement_type(req) == explicit_type

    def test_auto_triggers_classification(self):
        from koleslaw_api.schemas.enhance import EnhanceRequest
        from koleslaw_api.services.llm import LLMService

        req = EnhanceRequest(prompt="write a function", enhancement_type="auto")
        assert LLMService.resolve_enhancement_type(req) == "expand"

    def test_default_is_auto(self):
        from koleslaw_api.schemas.enhance import EnhanceRequest

        req = EnhanceRequest(prompt="write a function")
        assert req.enhancement_type == "auto"

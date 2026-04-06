#!/usr/bin/env bash
set -euo pipefail

# Seed the test user into the local Postgres and print a session JWT.
#
# After running, paste the token into your browser devtools console:
#   allow paste;
#   document.cookie = "pe_session=<paste_token_here>;path=/";
#
# Refresh the page and you'll be logged in as Test User.

echo "Seeding test user..."
docker compose -f docker-compose.yml exec -T postgres \
  psql -U postgres -d prompt_enhance < db/seed_test_user.sql
echo "Test user seeded."

echo ""
echo "Session token (paste into browser cookie pe_session):"
uv run python -c "
import jwt
from datetime import datetime, timedelta, UTC
token = jwt.encode(
    {'sub': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'iat': datetime.now(UTC), 'exp': datetime.now(UTC) + timedelta(days=7)},
    'dev-session-secret-change-me',
    algorithm='HS256',
)
print(token)
"

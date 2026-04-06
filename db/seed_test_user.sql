-- Seed a test user with API keys and usage logs for testing profile deletion.
-- Run with: psql $DATABASE_URL -f db/seed_test_user.sql

BEGIN;

-- Test user (GitHub OAuth)
INSERT INTO users (id, email, name, oauth_provider, oauth_id, created_at)
VALUES (
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    'testuser@example.com',
    'Test User',
    'github',
    '12345678',
    '2026-01-15T10:30:00Z'
);

-- Active API key
INSERT INTO api_keys (id, user_id, key_hash, name, key_prefix, created_at, last_used_at, is_active, rate_limit)
VALUES (
    'b2c3d4e5-f6a7-8901-bcde-f12345678901',
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    'aa11bb22cc33dd44ee55ff6677889900aabbccddeeff00112233445566778899',
    'Production',
    'pk_live_test',
    '2026-02-01T09:00:00Z',
    '2026-04-04T14:22:00Z',
    TRUE,
    60
);

-- Revoked API key
INSERT INTO api_keys (id, user_id, key_hash, name, key_prefix, created_at, last_used_at, is_active, rate_limit)
VALUES (
    'c3d4e5f6-a7b8-9012-cdef-123456789012',
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    'bb22cc33dd44ee55ff6677889900aabb11223344556677889900aabbccddeeff',
    'Development (old)',
    'pk_live_dev_',
    '2026-01-20T11:00:00Z',
    '2026-03-10T08:15:00Z',
    FALSE,
    120
);

-- Usage logs for the active key
INSERT INTO usage_logs (api_key_id, timestamp, model, enhancement_type, input_tokens, output_tokens, cached, latency_ms)
VALUES
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-04-04T14:22:00Z', 'claude-sonnet-4-6', 'rewrite',   120, 340, FALSE, 1820),
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-04-04T11:05:00Z', 'claude-sonnet-4-6', 'expand',    85,  210, FALSE, 1340),
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-04-03T16:45:00Z', 'gpt-4o',            'rewrite',   200, 520, FALSE, 2100),
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-04-03T09:30:00Z', 'claude-sonnet-4-6', 'summarize', 450, 180, TRUE,  420),
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-04-02T13:10:00Z', 'claude-sonnet-4-6', 'rewrite',   95,  290, FALSE, 1560),
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-03-28T10:00:00Z', 'gpt-4o',            'expand',    110, 380, FALSE, 1980),
    ('b2c3d4e5-f6a7-8901-bcde-f12345678901', '2026-03-25T15:20:00Z', 'claude-sonnet-4-6', 'rewrite',   75,  260, TRUE,  380);

-- Usage logs for the revoked key (historical)
INSERT INTO usage_logs (api_key_id, timestamp, model, enhancement_type, input_tokens, output_tokens, cached, latency_ms)
VALUES
    ('c3d4e5f6-a7b8-9012-cdef-123456789012', '2026-03-10T08:15:00Z', 'claude-sonnet-4-6', 'rewrite',   140, 310, FALSE, 1650),
    ('c3d4e5f6-a7b8-9012-cdef-123456789012', '2026-03-08T12:40:00Z', 'gpt-4o',            'expand',    90,  270, FALSE, 1890),
    ('c3d4e5f6-a7b8-9012-cdef-123456789012', '2026-02-20T17:55:00Z', 'claude-sonnet-4-6', 'summarize', 320, 150, TRUE,  350);

COMMIT;

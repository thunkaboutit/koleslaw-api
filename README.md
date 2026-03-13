# prompt-enhance

AI prompt enhancement API service. Takes rough or incomplete prompts and returns improved versions using LLM-powered
enhancement strategies (expand, refine, structure).

Built with FastAPI, LiteLLM, Redis, and PostgreSQL.

## Features

- **Auto-detected enhancement modes**: expand (add detail), refine (improve clarity), structure (reorganize) — automatically selected based on prompt characteristics, with manual override
- **Streaming and synchronous endpoints** via SSE and JSON
- **Multi-provider LLM support** through LiteLLM (Anthropic, OpenAI)
- **Exact-match prompt caching** with Redis (in-memory LRU fallback)
- **API key authentication** with sliding-window rate limiting
- **Usage tracking** via PostgreSQL

## API Endpoints

| Method | Path               | Auth | Description                    |
|--------|--------------------|------|--------------------------------|
| `GET`  | `/v1/health`       | No   | Health check (DB + Redis)      |
| `GET`  | `/v1/models`       | No   | List available models          |
| `POST` | `/v1/enhance`      | Yes  | Streaming enhancement (SSE)    |
| `POST` | `/v1/enhance/sync` | Yes  | Synchronous enhancement (JSON) |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose (for infrastructure services)

### Setup

```bash
# Clone and enter the project
cd prompt-enhance

# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running with Docker Compose

This starts the API, PostgreSQL, and Redis together:

```bash
make up          # Start all services
make logs        # Tail logs
make down        # Stop all services
```

The API will be available at `http://localhost:8000`.

> **Note:** When running with Docker Compose, the `.env` file should use service names as hostnames (`postgres` and
`redis` instead of `localhost`).

### Running for Local Development

Start PostgreSQL and Redis separately (e.g., via Docker or local install), then:

```bash
make dev         # Start uvicorn with --reload
```

In development mode (`ENVIRONMENT=development`), requests without an API key are allowed through for easier testing.

### Verify It Works

```bash
curl http://localhost:8000/v1/health
curl http://localhost:8000/v1/models
```

### Testing the Enhancer

Synchronous (returns JSON):

```bash
curl http://localhost:8000/v1/enhance/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem"}'
```

Streaming (returns SSE):

```bash
curl -N http://localhost:8000/v1/enhance \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem"}'
```

The enhancement type is automatically selected based on prompt length and structure. To override, pass
`"enhancement_type": "expand"`, `"refine"`, or `"structure"` in the request body. To use a different model, add
`"model": "openai/gpt-4o-mini"`.

## Environment Variables

| Variable                         | Required | Default                                  | Description                               |
|----------------------------------|----------|------------------------------------------|-------------------------------------------|
| `DATABASE_URL`                   | Yes      | —                                        | PostgreSQL async connection string        |
| `REDIS_URL`                      | Yes      | —                                        | Redis connection string                   |
| `API_KEY_HASH_SECRET`            | Yes      | —                                        | Secret for API key hashing                |
| `OLLAMA_BASE_URL`                | No       | `http://localhost:11434`                 | Ollama API base URL                       |
| `ANTHROPIC_API_KEY`              | No       | —                                        | Anthropic API key (for frontier fallback) |
| `OPENAI_API_KEY`                 | No       | —                                        | OpenAI API key (for frontier fallback)    |
| `DEFAULT_MODEL`                  | No       | `ollama_chat/prompt-enhance-14b-think`   | Default LLM model                         |
| `ENVIRONMENT`                    | No       | `development`                            | `development`, `staging`, or `production` |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | No       | `60`                                     | Per-key rate limit                        |
| `CACHE_TTL_SECONDS`              | No       | `86400`                                  | Cache entry lifetime (seconds)            |
| `CORS_ORIGINS`                   | No       | `["*"]`                                  | Allowed CORS origins                      |

## Development

### Make Targets

```
make dev       Run uvicorn with hot reload
make test      Run the test suite
make lint      Run ruff linter
make format    Format code with ruff
make up        Start Docker Compose services
make down      Stop Docker Compose services
make logs      Tail Docker Compose logs
```

### Running Tests

Tests use mocked Redis and database connections — no running infrastructure required:

```bash
make test
# or
uv run pytest -v
```

### Linting

```bash
make lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### Project Structure

```
src/prompt_enhance/
├── main.py              # FastAPI app entry point and lifespan
├── config.py            # Pydantic Settings configuration
├── db/
│   ├── engine.py        # Async SQLAlchemy engine factory
│   └── models.py        # ORM models (ApiKey, UsageLog)
├── prompts/
│   └── system.py        # Enhancement type classification and meta-prompt templates
├── routes/
│   ├── health.py        # GET /v1/health
│   ├── models.py        # GET /v1/models
│   └── enhance.py       # POST /v1/enhance, /v1/enhance/sync
├── schemas/
│   └── enhance.py       # Pydantic request/response models
└── services/
    ├── auth.py          # API key auth and rate limiting
    ├── cache.py         # Exact-match prompt cache
    └── llm.py           # LiteLLM wrapper service
```

## CI

GitHub Actions runs lint and test jobs in parallel on every push to `main` and on pull requests. See
`.github/workflows/ci.yml`.

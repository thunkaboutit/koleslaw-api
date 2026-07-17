# koleslaw-api

AI prompt enhancement API service built with FastAPI, PostgreSQL, and Redis.

## Tech Stack

- **Language:** Python 3.14, managed with `uv`
- **Framework:** FastAPI + Uvicorn
- **Database:** PostgreSQL 16 (asyncpg + SQLAlchemy async)
- **Cache:** Redis 7
- **LLM:** LiteLLM (supports Anthropic, OpenAI, Ollama)
- **Linting/Formatting:** Ruff
- **Type Checking:** mypy
- **Testing:** pytest + pytest-asyncio

## Project Layout

```
src/koleslaw_api/     # Application package
  config.py             # Pydantic settings (reads .env)
  main.py               # FastAPI app entrypoint
  routes/               # API route handlers
  schemas/              # Pydantic request/response models
  services/             # Business logic (auth, cache, llm)
  db/                   # SQLAlchemy engine + models
  prompts/              # System prompt templates
tests/                  # pytest tests (async, mocked DB/Redis)
db/init.sql             # Database schema DDL
```

## Common Commands

```bash
make test           # Run tests: uv run pytest
make lint           # Lint: uv run ruff check src/ tests/
make format         # Format: uv run ruff format src/
make dev-local      # Start Postgres + Redis, run app with --reload
make up-local       # Docker compose up (all services)
make down-local     # Docker compose down
make db-init-local  # Initialize database schema
make db-reset-local # Wipe and re-init database
```

## Development Setup

1. Copy `.env.example` to `.env` and fill in keys
2. `uv sync` to install dependencies
3. `make db-init-local` to set up the database
4. `make dev-local` to run locally with hot reload

## Testing

Tests use mocked Redis and database sessions (no running services required). Run with `make test` or `uv run pytest`.

## Ruff Configuration

- Line length: 120
- Target: Python 3.12
- Rules: E, F, I, N, UP, B

## Code Navigation

IMPORTANT: When applicable, prefer using pycharm-index MCP tools for code navigation and refactoring.

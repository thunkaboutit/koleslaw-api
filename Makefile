.PHONY: up down logs test lint format dev db-init db-reset

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/

dev:
	docker compose up -d postgres redis
	uv run uvicorn prompt_enhance.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

db-init:
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL..."
	@until docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do sleep 1; done
	docker compose exec -T postgres psql -U postgres -d prompt_enhance < db/init.sql
	@echo "Database initialized."

db-reset:
	docker compose down -v
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL..."
	@until docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do sleep 1; done
	docker compose exec -T postgres psql -U postgres -d prompt_enhance < db/init.sql
	docker compose up -d redis
	@echo "Database reset complete."

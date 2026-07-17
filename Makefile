.PHONY: list up-local up-dev down-local down-dev logs test lint format dev-local db-init-local db-init-dev db-reset seed-test-user

list:
	cat Makefile | grep '^[^#[:space:]].*:' | grep -v PHONY

up-local:
	docker compose up -d

up-dev:
	docker compose -f docker-compose.dev.yml up -d

down-local:
	docker compose down

down-dev:
	docker compose -f docker-compose.dev.yml down

logs:
	docker compose logs -f

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/

dev-local:
	docker compose up -d postgres redis
	uv run uvicorn koleslaw_api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

db-init-local:
	docker compose -f docker-compose.yml up -d postgres
	@echo "Waiting for PostgresSQL..."
	@until docker compose -f docker-compose.yml exec postgres pg_isready -U postgres > /dev/null 2>&1; do sleep 1; done
	docker compose -f docker-compose.yml exec -T postgres psql -U postgres -d koleslaw_api < db/init.sql
	@echo "Database initialized."

db-init-dev:
	docker compose -f docker-compose.dev.yml up -d postgres
	@echo "Waiting for PostgresSQL..."
	@until docker compose -f docker-compose.dev.yml exec postgres pg_isready -U app > /dev/null 2>&1; do sleep 1; done
	docker compose -f docker-compose.dev.yml exec -T postgres psql -U app -d appdb < db/init.sql
	@echo "Database initialized."

db-reset-local:
	docker compose -f docker-compose.yml down -v
	docker compose -f docker-compose.yml up -d postgres
	@echo "Waiting for PostgresSQL..."
	@until docker compose -f docker-compose.yml exec postgres pg_isready -U postgres > /dev/null 2>&1; do sleep 1; done
	docker compose -f docker-compose.yml exec -T postgres psql -U postgres -d koleslaw_api < db/init.sql
	docker compose -f docker-compose.yml up -d redis
	@echo "Database reset complete."

db-reset-dev:
	docker compose -f docker-compose.dev.yml down -v
	docker compose -f docker-compose.dev.yml up -d postgres
	@echo "Waiting for PostgresSQL..."
	@until docker compose -f docker-compose.dev.yml exec postgres pg_isready -U app > /dev/null 2>&1; do sleep 1; done
	docker compose -f docker-compose.dev.yml exec -T postgres psql -U app -d appdb < db/init.sql
	docker compose -f docker-compose.dev.yml up -d redis
	@echo "Database reset complete."

seed-test-user:
	bash db/setup_test_user.sh

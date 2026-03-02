.PHONY: up down logs test lint format dev

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	uv run pytest

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

dev:
	uv run uvicorn prompt_enhance.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: help install dev lint type-check test test-cov clean docker-build docker-up docker-down migrate

PYTHON := python
PKG    := aumos_human_ai_collab

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package with dev dependencies
	pip install -e ".[dev]"

dev:  ## Run development server with hot reload
	uvicorn $(PKG).main:app --reload --host 0.0.0.0 --port 8000

lint:  ## Run ruff linter and formatter check
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:  ## Auto-fix lint and format issues
	ruff check --fix src/ tests/
	ruff format src/ tests/

type-check:  ## Run mypy type checker
	mypy src/

test:  ## Run tests without coverage
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ --cov=$(PKG) --cov-report=term-missing --cov-report=html

clean:  ## Remove build artifacts and caches
	rm -rf dist/ build/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker image
	docker build -t aumos-human-ai-collab:latest .

docker-up:  ## Start all services with docker-compose
	docker compose -f docker-compose.dev.yml up -d

docker-down:  ## Stop all services
	docker compose -f docker-compose.dev.yml down

migrate:  ## Run Alembic database migrations
	alembic upgrade head

migrate-new:  ## Create a new Alembic migration (usage: make migrate-new MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

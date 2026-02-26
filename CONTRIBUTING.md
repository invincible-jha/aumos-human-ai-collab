# Contributing to aumos-human-ai-collab

Thank you for contributing to AumOS Human-AI Collaboration.

## Development Setup

```bash
git clone <repo-url>
cd aumos-human-ai-collab
pip install -e ".[dev]"
cp .env.example .env
make docker-up   # Start Postgres, Kafka, Redis
make migrate     # Run DB migrations
make dev         # Start dev server on :8000
```

## Code Standards

- Python 3.11+ with full type hints on all function signatures
- Ruff for linting and formatting (`make lint-fix`)
- Mypy strict mode (`make type-check`)
- Tests alongside implementation (`make test-cov`)

## Branching

- `main` is protected — open PRs for all changes
- Branch naming: `feature/`, `fix/`, `docs/`, `refactor/`
- Squash-merge PRs to keep history linear

## Commit Messages

Follow Conventional Commits:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code improvement without behaviour change
- `docs:` documentation only
- `test:` tests only
- `chore:` tooling, deps, CI

## Architecture

This service uses hexagonal architecture:
- `core/` — domain models, services, and port interfaces (no framework imports)
- `api/` — FastAPI routers and Pydantic schemas
- `adapters/` — concrete repository and external service implementations

Services depend only on interfaces from `core/interfaces.py`, never on concrete adapters.
This makes all business logic independently testable via mock implementations.

## Routing Logic

The key invariant: **compliance gates always override confidence scores**.
If a task type matches any active ComplianceGate, it routes to human regardless of the AI confidence.

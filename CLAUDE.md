# CLAUDE.md — AumOS Human-AI Collaboration

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 78+ repositories. This repo (`aumos-human-ai-collab`) is part of **Tier 1: Governance
& Compliance Platform**: Confidence-based routing between AI and humans with regulatory
compliance gates, HITL review interface, productivity attribution, and feedback loops.

**Release Tier:** A: Enterprise Core
**Product Mapping:** Product 6 — AI Governance Platform
**Phase:** 3 (Enterprise Integration)
**Port:** 8000

## Repo Purpose

Implements the human-AI collaboration layer that determines whether each task is handled
autonomously by AI, routed to a human reviewer, or handled via a hybrid approach. Compliance
gates for regulated industries (healthcare, financial services) override confidence scores
to ensure mandatory human review where required by regulation.

## Architecture Position

```
aumos-agent-framework  → aumos-human-ai-collab → hitl reviewers (UI layer)
aumos-llm-serving      ↗                       ↘ aumos-attribution (ROI)
aumos-governance-engine ↗                      ↘ aumos-model-registry (recalibration)
aumos-common            ↗                      ↘ Kafka (hac.* events)
                                               ↘ Redis (review queue)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-governance-engine` — policy evaluation for compliance gate configuration
- `aumos-model-registry` — model confidence scoring endpoints

**Downstream dependents (other repos IMPORT from this):**
- Any service needing routing decisions before executing a task
- `aumos-agent-framework` — agents check routing before autonomous execution
- `aumos-llm-serving` — high-stakes prompts evaluated for routing before execution

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka events via aumos-common |
| redis | 5.0+ | HITL review queue and distributed locking |
| httpx | 0.27+ | HTTP client for model-registry confidence scoring |
| structlog | 24.1+ | Structured JSON logging |
| pytest | 8.0+ | Testing framework |

## Database Schema (Table Prefix: `hac_`)

All tables use the `hac_` prefix. All extend AumOSModel (id UUID, tenant_id, created_at, updated_at).

```
hac_routing_decisions       confidence score, routing outcome, compliance gate status
hac_compliance_gates        regulatory gates requiring mandatory human review
hac_hitl_reviews            human review tasks with AI output, status, and decisions
hac_attribution_records     AI vs. human contribution per completed task
hac_feedback_corrections    human corrections to recalibrate AI confidence
```

## Key Invariants

1. **Compliance gates always override confidence** — If a ComplianceGate matches a task_type,
   the routing is always "human" regardless of confidence score.
2. **HITL decisions are immutable** — Once a reviewer submits approved/rejected/modified,
   the decision cannot be changed. This is a compliance requirement.
3. **Attribution scores sum to 1.0** — ai_contribution_score + human_contribution_score = 1.0
4. **Recalibration is advisory** — Feedback corrections update an in-memory calibration table
   but never directly modify AI model weights. Threshold changes require human approval.

## Routing Decision Logic

```
confidence >= ai_confidence_threshold (0.85)  →  ai
confidence >= hybrid_confidence_lower (0.65)  →  hybrid
confidence < hybrid_confidence_lower          →  human
compliance gate match (any)                   →  human (overrides all)
```

## Environment Variables (Prefix: `AUMOS_HUMAN_AI_`)

Key variables:
- `AUMOS_HUMAN_AI_AI_CONFIDENCE_THRESHOLD` — AI autonomous threshold (default 0.85)
- `AUMOS_HUMAN_AI_HYBRID_CONFIDENCE_LOWER` — Hybrid lower bound (default 0.65)
- `AUMOS_HUMAN_AI_HITL_REVIEW_TIMEOUT_HOURS` — Review deadline (default 24h)
- `AUMOS_HUMAN_AI_FEEDBACK_CALIBRATION_MIN_SAMPLES` — Recalibration trigger count (default 10)

See `.env.example` for full list.

## Common Commands

```bash
make install    # Install with dev deps
make dev        # Start on :8000 with hot reload
make lint       # Ruff lint + format check
make type-check # Mypy strict
make test       # Run tests
make docker-up  # Start Postgres + Kafka + Redis
```

## File Structure

```
src/aumos_human_ai_collab/
├── __init__.py          — Package version
├── main.py              — FastAPI app + lifespan
├── settings.py          — Pydantic settings (AUMOS_HUMAN_AI_ prefix)
├── api/
│   ├── router.py        — FastAPI routes (thin layer, delegates to services)
│   └── schemas.py       — Pydantic request/response models
├── core/
│   ├── models.py        — SQLAlchemy ORM models (hac_ prefix)
│   ├── services.py      — Business logic (RoutingService, ComplianceGateService, etc.)
│   └── interfaces.py    — Protocol classes for all adapters
└── adapters/
    ├── repositories.py  — SQLAlchemy concrete implementations
    ├── kafka.py         — Kafka event publisher
    └── confidence_engine.py — AI confidence scoring adapter
```

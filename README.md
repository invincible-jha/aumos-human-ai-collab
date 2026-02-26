# aumos-human-ai-collab

**AumOS Human-AI Collaboration** — Confidence-based task routing between AI and humans,
regulatory compliance gates, human-in-the-loop (HITL) review interface, productivity
attribution, and feedback loops for AI confidence calibration.

## Features

- **Confidence-based routing** — Tasks routed to AI, human, or hybrid based on configurable confidence thresholds
- **Compliance gates** — Regulatory overrides (HIPAA, GDPR, SOX, PCI-DSS, FINRA, FDA) that force human review regardless of confidence
- **HITL reviews** — Human-in-the-loop review queue with priority, deadline tracking, and immutable decisions
- **Productivity attribution** — Track AI vs. human contribution per task type for ROI measurement
- **Feedback calibration** — Human corrections trigger automatic recalibration of AI confidence scoring

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
make docker-up
make migrate
make dev
```

API available at `http://localhost:8000`
OpenAPI docs at `http://localhost:8000/docs`

## API Surface

```
POST   /api/v1/routing/evaluate              # Evaluate task routing (AI vs human)
GET    /api/v1/routing/decisions              # List routing decisions
POST   /api/v1/compliance/gates/evaluate      # Evaluate compliance gate
POST   /api/v1/compliance/gates               # Create compliance gate
GET    /api/v1/compliance/gates               # List compliance gates
POST   /api/v1/hitl/reviews                   # Create HITL review
GET    /api/v1/hitl/reviews                   # List HITL reviews
GET    /api/v1/hitl/reviews/{id}              # Review detail
PATCH  /api/v1/hitl/reviews/{id}              # Submit review decision
GET    /api/v1/attribution/reports            # Attribution analytics
POST   /api/v1/feedback                       # Submit correction feedback
GET    /api/v1/feedback/calibration/{type}    # Calibration summary
```

## Routing Logic

```
task_type + task_payload
         │
         ▼
  ComplianceGate check  ─── match? ──► human (mandatory)
         │
    no match
         │
         ▼
  ConfidenceEngine.score()
         │
         ├── score >= 0.85  ──► ai
         ├── score >= 0.65  ──► hybrid
         └── score < 0.65   ──► human
```

## Database Tables

| Table | Purpose |
|-------|---------|
| `hac_routing_decisions` | Confidence score and routing outcome per task |
| `hac_compliance_gates` | Regulatory gates requiring human review |
| `hac_hitl_reviews` | Human review tasks with AI output |
| `hac_attribution_records` | AI vs. human contribution tracking |
| `hac_feedback_corrections` | Human corrections for confidence recalibration |

## Architecture

Hexagonal architecture:
- `core/` — domain models, services, interface protocols
- `api/` — FastAPI routers and Pydantic schemas
- `adapters/` — SQLAlchemy repositories, Kafka publisher, confidence engine

## Environment Variables

See `.env.example` for all configuration options.
All settings use the `AUMOS_HUMAN_AI_` prefix.

## License

Apache-2.0 — see [LICENSE](LICENSE)

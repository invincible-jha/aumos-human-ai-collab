# Changelog

All notable changes to `aumos-human-ai-collab` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial implementation of confidence-based task routing (AI / human / hybrid)
- `RoutingService` with configurable confidence thresholds and compliance gate integration
- `ComplianceGateService` for regulatory routing overrides (HIPAA, GDPR, SOX, PCI-DSS, FINRA, FDA)
- `HITLReviewService` for human-in-the-loop review task management
- `AttributionService` for AI vs. human productivity attribution reporting
- `FeedbackService` for human correction submission and AI confidence recalibration
- `ConfidenceEngineAdapter` with model-registry integration and stub fallback
- REST API: POST /api/v1/routing/evaluate, GET /api/v1/routing/decisions
- REST API: POST /api/v1/compliance/gates/evaluate, POST /api/v1/compliance/gates
- REST API: POST /api/v1/hitl/reviews, GET/PATCH /api/v1/hitl/reviews/{id}
- REST API: GET /api/v1/attribution/reports
- REST API: POST /api/v1/feedback, GET /api/v1/feedback/calibration/{task_type}
- SQLAlchemy ORM models: hac_routing_decisions, hac_compliance_gates, hac_hitl_reviews,
  hac_attribution_records, hac_feedback_corrections
- Hexagonal architecture with hexagonal ports/adapters pattern
- Kafka event publishing for all domain events
- Docker and docker-compose support

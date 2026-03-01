"""AumOS Human-AI Collaboration service entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_human_ai_collab.adapters.confidence_engine import ConfidenceEngineAdapter
from aumos_human_ai_collab.adapters.kafka import HumanAIEventPublisher
from aumos_human_ai_collab.adapters.repositories import (
    AttributionRepository,
    ComplianceGateRepository,
    FeedbackRepository,
    HITLReviewRepository,
    RoutingDecisionRepository,
)
from aumos_human_ai_collab.api.gap_router import gap_router
from aumos_human_ai_collab.api.router import router
from aumos_human_ai_collab.api.ui_router import ui_router
from aumos_human_ai_collab.core.gap_services import (
    AnnotationSchemaService,
    LabelStudioService,
    LLMEvaluationService,
    PromptManagementService,
    ReviewerUIService,
    WorkforceService,
)
from aumos_human_ai_collab.core.services import (
    AttributionService,
    ComplianceGateService,
    FeedbackService,
    HITLReviewService,
    RoutingService,
)
from aumos_human_ai_collab.settings import Settings

import pathlib

logger = get_logger(__name__)
settings = Settings()

_kafka_publisher: HumanAIEventPublisher | None = None

_STATIC_DIR = pathlib.Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the database connection pool, Kafka event publisher,
    repository instances, and service instances on app.state for DI.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    global _kafka_publisher  # noqa: PLW0603

    logger.info("Starting AumOS Human-AI Collaboration service", version="0.1.0")

    # Database connection pool
    init_database(settings.database)
    logger.info("Database connection pool ready")

    # Kafka event publisher
    _kafka_publisher = HumanAIEventPublisher(settings.kafka)
    await _kafka_publisher.start()
    app.state.kafka_publisher = _kafka_publisher
    logger.info("Kafka event publisher ready")

    # Repository instances
    routing_repo = RoutingDecisionRepository()
    compliance_repo = ComplianceGateRepository()
    hitl_repo = HITLReviewRepository()
    attribution_repo = AttributionRepository()
    feedback_repo = FeedbackRepository()

    # Confidence engine adapter
    confidence_engine = ConfidenceEngineAdapter(
        model_registry_url=settings.model_registry_url,
        http_timeout=settings.http_timeout,
    )

    # Core service instances with dependency injection
    app.state.routing_service = RoutingService(
        routing_repo=routing_repo,
        compliance_repo=compliance_repo,
        confidence_engine=confidence_engine,
        event_publisher=_kafka_publisher,
        ai_confidence_threshold=settings.ai_confidence_threshold,
        hybrid_confidence_lower=settings.hybrid_confidence_lower,
    )

    app.state.compliance_service = ComplianceGateService(
        compliance_repo=compliance_repo,
        event_publisher=_kafka_publisher,
    )

    app.state.hitl_service = HITLReviewService(
        review_repo=hitl_repo,
        attribution_repo=attribution_repo,
        event_publisher=_kafka_publisher,
        review_timeout_hours=settings.hitl_review_timeout_hours,
    )

    app.state.attribution_service = AttributionService(
        attribution_repo=attribution_repo,
        event_publisher=_kafka_publisher,
        default_report_days=settings.attribution_report_days,
    )

    app.state.feedback_service = FeedbackService(
        feedback_repo=feedback_repo,
        confidence_engine=confidence_engine,
        event_publisher=_kafka_publisher,
        calibration_min_samples=settings.feedback_calibration_min_samples,
        feedback_decay_factor=settings.feedback_decay_factor,
    )

    # GAP-256: Reviewer UI session service
    app.state.reviewer_ui_service = ReviewerUIService(
        event_publisher=_kafka_publisher,
        session_ttl_hours=settings.ui_session_ttl_hours,
    )

    # GAP-257: LLM evaluation service
    app.state.llm_evaluation_service = LLMEvaluationService(
        event_publisher=_kafka_publisher,
        pass_threshold=settings.llm_evaluation_pass_threshold,
    )

    # GAP-258: Annotation schema service
    app.state.annotation_schema_service = AnnotationSchemaService(
        event_publisher=_kafka_publisher,
    )

    # GAP-259: Label Studio integration service
    app.state.label_studio_service = LabelStudioService(
        event_publisher=_kafka_publisher,
    )

    # GAP-260: Workforce management service
    app.state.workforce_service = WorkforceService(
        event_publisher=_kafka_publisher,
    )

    # GAP-261: Prompt management service
    app.state.prompt_management_service = PromptManagementService(
        event_publisher=_kafka_publisher,
    )

    # Expose settings on app state for dependency injection
    app.state.settings = settings

    logger.info("Human-AI Collaboration service startup complete")
    yield

    # Shutdown
    if _kafka_publisher:
        await _kafka_publisher.stop()

    logger.info("Human-AI Collaboration service shutdown complete")


app: FastAPI = create_app(
    service_name="aumos-human-ai-collab",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=lambda: None),
        HealthCheck(name="kafka", check_fn=lambda: None),
        HealthCheck(name="redis", check_fn=lambda: None),
    ],
)

# API routes
app.include_router(router, prefix="/api/v1")
app.include_router(gap_router, prefix="/api/v1")

# Reviewer UI (GAP-256) — server-rendered HTML, optional
if settings.ui_enabled:
    app.include_router(ui_router)
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

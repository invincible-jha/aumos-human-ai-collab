"""Shared pytest fixtures for the Human-AI Collaboration service tests.

Provides mock adapters and factory helpers that satisfy the Protocol
interfaces without requiring real database, Kafka, or Redis connections.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_human_ai_collab.core.models import (
    AttributionRecord,
    ComplianceGate,
    FeedbackCorrection,
    HITLReview,
    RoutingDecision,
)


# ---------------------------------------------------------------------------
# Stable UUIDs for deterministic tests
# ---------------------------------------------------------------------------

TENANT_ID: uuid.UUID = uuid.UUID("10000000-0000-0000-0000-000000000001")
TASK_ID: uuid.UUID = uuid.UUID("20000000-0000-0000-0000-000000000002")
DECISION_ID: uuid.UUID = uuid.UUID("30000000-0000-0000-0000-000000000003")
GATE_ID: uuid.UUID = uuid.UUID("40000000-0000-0000-0000-000000000004")
REVIEW_ID: uuid.UUID = uuid.UUID("50000000-0000-0000-0000-000000000005")
REVIEWER_ID: uuid.UUID = uuid.UUID("60000000-0000-0000-0000-000000000006")
CORRECTION_ID: uuid.UUID = uuid.UUID("70000000-0000-0000-0000-000000000007")
ATTRIBUTION_ID: uuid.UUID = uuid.UUID("80000000-0000-0000-0000-000000000008")


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------


def make_routing_decision(
    *,
    routing_outcome: str = "ai",
    confidence_score: float = 0.90,
    compliance_gate_triggered: bool = False,
    task_type: str = "document_analysis",
) -> RoutingDecision:
    """Build a RoutingDecision ORM instance without a database session.

    Args:
        routing_outcome: Routing result: ai | human | hybrid.
        confidence_score: Confidence score at routing time.
        compliance_gate_triggered: Whether a compliance gate forced routing.
        task_type: Semantic task category.

    Returns:
        Populated RoutingDecision instance.
    """
    decision = MagicMock(spec=RoutingDecision)
    decision.id = DECISION_ID
    decision.tenant_id = TENANT_ID
    decision.task_id = TASK_ID
    decision.task_type = task_type
    decision.confidence_score = confidence_score
    decision.routing_outcome = routing_outcome
    decision.ai_threshold_applied = 0.85
    decision.model_id = "stub-model-v1"
    decision.compliance_gate_triggered = compliance_gate_triggered
    decision.compliance_gate_id = GATE_ID if compliance_gate_triggered else None
    decision.routing_metadata = {}
    decision.resolved_by = None
    decision.resolved_at = None
    decision.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    decision.updated_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return decision


def make_compliance_gate(
    *,
    regulation: str = "hipaa",
    task_types: list[str] | None = None,
    is_active: bool = True,
) -> ComplianceGate:
    """Build a ComplianceGate ORM instance without a database session.

    Args:
        regulation: Regulatory framework string.
        task_types: Task types covered by this gate.
        is_active: Whether the gate is currently active.

    Returns:
        Populated ComplianceGate instance.
    """
    gate = MagicMock(spec=ComplianceGate)
    gate.id = GATE_ID
    gate.tenant_id = TENANT_ID
    gate.gate_name = "hipaa-clinical-notes"
    gate.regulation = regulation
    gate.task_types = task_types or ["medical_diagnosis", "clinical_notes"]
    gate.description = "HIPAA-mandated human review for clinical tasks"
    gate.is_active = is_active
    gate.reviewer_role = "compliance_officer"
    gate.metadata = {}
    gate.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    gate.updated_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return gate


def make_hitl_review(
    *,
    status: str = "pending",
    task_type: str = "document_analysis",
    priority: int = 2,
) -> HITLReview:
    """Build a HITLReview ORM instance without a database session.

    Args:
        status: Review status string.
        task_type: Semantic task category.
        priority: Review priority 1–4.

    Returns:
        Populated HITLReview instance.
    """
    review = MagicMock(spec=HITLReview)
    review.id = REVIEW_ID
    review.tenant_id = TENANT_ID
    review.routing_decision_id = DECISION_ID
    review.task_type = task_type
    review.status = status
    review.ai_output = {"result": "draft answer"}
    review.reviewer_id = None
    review.review_started_at = None
    review.review_completed_at = None
    review.decision = None
    review.reviewer_output = {}
    review.reviewer_notes = None
    review.due_at = datetime(2024, 1, 16, 12, 0, 0, tzinfo=timezone.utc)
    review.priority = priority
    review.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    review.updated_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return review


def make_attribution_record(
    *,
    ai_contribution_score: float = 1.0,
    human_contribution_score: float = 0.0,
    resolution: str = "ai",
) -> AttributionRecord:
    """Build an AttributionRecord ORM instance without a database session.

    Args:
        ai_contribution_score: Fraction attributed to AI (0–1).
        human_contribution_score: Fraction attributed to human (0–1).
        resolution: Final resolution: ai | human | hybrid.

    Returns:
        Populated AttributionRecord instance.
    """
    record = MagicMock(spec=AttributionRecord)
    record.id = ATTRIBUTION_ID
    record.tenant_id = TENANT_ID
    record.task_id = TASK_ID
    record.task_type = "document_analysis"
    record.routing_decision_id = DECISION_ID
    record.hitl_review_id = None
    record.ai_contribution_score = ai_contribution_score
    record.human_contribution_score = human_contribution_score
    record.time_saved_seconds = None
    record.attribution_method = "routing_outcome"
    record.resolution = resolution
    record.attribution_metadata = {}
    record.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    record.updated_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return record


def make_feedback_correction(
    *,
    original_routing: str = "ai",
    corrected_routing: str = "human",
    correction_type: str = "routing_error",
    calibration_applied: bool = False,
) -> FeedbackCorrection:
    """Build a FeedbackCorrection ORM instance without a database session.

    Args:
        original_routing: Original routing outcome.
        corrected_routing: Correct routing outcome.
        correction_type: Category of correction.
        calibration_applied: Whether this correction has been used in calibration.

    Returns:
        Populated FeedbackCorrection instance.
    """
    correction = MagicMock(spec=FeedbackCorrection)
    correction.id = CORRECTION_ID
    correction.tenant_id = TENANT_ID
    correction.routing_decision_id = DECISION_ID
    correction.hitl_review_id = None
    correction.task_type = "document_analysis"
    correction.original_confidence = 0.92
    correction.original_routing = original_routing
    correction.corrected_routing = corrected_routing
    correction.correction_reason = "AI was overconfident"
    correction.correction_type = correction_type
    correction.submitted_by = REVIEWER_ID
    correction.calibration_applied = calibration_applied
    correction.calibration_applied_at = None
    correction.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return correction


# ---------------------------------------------------------------------------
# Mock adapter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_routing_repo() -> AsyncMock:
    """Provide an AsyncMock implementing IRoutingDecisionRepository.

    Returns:
        Configured AsyncMock for routing decision repository.
    """
    repo = AsyncMock()
    repo.create.return_value = make_routing_decision()
    repo.get_by_id.return_value = make_routing_decision()
    repo.list_by_tenant.return_value = ([make_routing_decision()], 1)
    repo.mark_resolved.return_value = make_routing_decision(routing_outcome="ai")
    return repo


@pytest.fixture()
def mock_compliance_repo() -> AsyncMock:
    """Provide an AsyncMock implementing IComplianceGateRepository.

    Returns:
        Configured AsyncMock for compliance gate repository.
    """
    repo = AsyncMock()
    repo.create.return_value = make_compliance_gate()
    repo.get_by_id.return_value = make_compliance_gate()
    repo.find_matching_gates.return_value = []
    repo.list_by_tenant.return_value = [make_compliance_gate()]
    repo.update.return_value = make_compliance_gate()
    return repo


@pytest.fixture()
def mock_hitl_repo() -> AsyncMock:
    """Provide an AsyncMock implementing IHITLReviewRepository.

    Returns:
        Configured AsyncMock for HITL review repository.
    """
    repo = AsyncMock()
    repo.create.return_value = make_hitl_review()
    repo.get_by_id.return_value = make_hitl_review()
    repo.list_by_tenant.return_value = ([make_hitl_review()], 1)
    repo.submit_decision.return_value = make_hitl_review(status="approved")
    repo.update_status.return_value = make_hitl_review(status="in_review")
    return repo


@pytest.fixture()
def mock_attribution_repo() -> AsyncMock:
    """Provide an AsyncMock implementing IAttributionRepository.

    Returns:
        Configured AsyncMock for attribution repository.
    """
    repo = AsyncMock()
    repo.create.return_value = make_attribution_record()
    repo.get_report.return_value = {
        "total_tasks": 100,
        "ai_handled": 70,
        "human_handled": 20,
        "hybrid_handled": 10,
        "ai_contribution_pct": 75.0,
        "human_contribution_pct": 25.0,
        "total_time_saved_seconds": 3600,
        "by_task_type": [],
    }
    return repo


@pytest.fixture()
def mock_feedback_repo() -> AsyncMock:
    """Provide an AsyncMock implementing IFeedbackRepository.

    Returns:
        Configured AsyncMock for feedback repository.
    """
    repo = AsyncMock()
    repo.create.return_value = make_feedback_correction()
    repo.list_uncalibrated.return_value = []
    repo.mark_calibrated.return_value = 10
    repo.get_calibration_summary.return_value = {
        "total_corrections": 15,
        "uncalibrated_count": 5,
        "error_rate": 0.33,
        "mean_confidence_delta": 0.12,
        "correction_type_breakdown": {"routing_error": 10, "confidence_overestimate": 5},
        "last_calibration_at": None,
    }
    return repo


@pytest.fixture()
def mock_confidence_engine() -> AsyncMock:
    """Provide an AsyncMock implementing IConfidenceEngineAdapter.

    Returns:
        Configured AsyncMock for confidence engine adapter.
    """
    engine = AsyncMock()
    engine.score_task.return_value = (0.90, "stub-model-v1")
    engine.recalibrate.return_value = {
        "task_type": "document_analysis",
        "sample_count": 10,
        "confidence_delta": -0.02,
        "new_calibration_adjustment": -0.02,
        "new_threshold_recommendation": None,
    }
    return engine


@pytest.fixture()
def mock_event_publisher() -> AsyncMock:
    """Provide an AsyncMock for the Kafka EventPublisher.

    Returns:
        Configured AsyncMock for event publisher.
    """
    publisher = AsyncMock()
    publisher.publish.return_value = None
    return publisher

"""Pydantic request and response schemas for the Human-AI Collaboration API.

All API inputs and outputs are typed Pydantic models — never raw dicts.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Routing schemas
# ---------------------------------------------------------------------------


class RoutingEvaluateRequest(BaseModel):
    """Request body for evaluating task routing."""

    task_id: uuid.UUID = Field(
        ...,
        description="External task UUID to route",
    )
    task_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Semantic task category (e.g., medical_diagnosis, contract_review)",
        examples=["medical_diagnosis"],
    )
    task_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Task input data used for confidence scoring",
    )
    model_id: str | None = Field(
        default=None,
        max_length=255,
        description="Optional preferred AI model identifier",
    )
    override_routing: str | None = Field(
        default=None,
        pattern="^(ai|human|hybrid)$",
        description="Optional manual routing override: ai | human | hybrid",
    )


class RoutingDecisionResponse(BaseModel):
    """Response schema for a routing decision."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    task_id: uuid.UUID
    task_type: str
    confidence_score: float
    routing_outcome: str
    ai_threshold_applied: float
    model_id: str | None
    compliance_gate_triggered: bool
    compliance_gate_id: uuid.UUID | None
    routing_metadata: dict[str, Any]
    resolved_by: str | None
    resolved_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RoutingDecisionListResponse(BaseModel):
    """Paginated list of routing decisions."""

    items: list[RoutingDecisionResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Compliance gate schemas
# ---------------------------------------------------------------------------


class ComplianceGateEvaluateRequest(BaseModel):
    """Request body for evaluating compliance gates against a task type."""

    task_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Task type to evaluate against active compliance gates",
    )


class ComplianceGateEvaluateResponse(BaseModel):
    """Response for a compliance gate evaluation."""

    triggered: bool = Field(description="True if any compliance gate applies to this task type")
    gate_ids: list[str] = Field(description="UUIDs of matching gates")
    gate_names: list[str] = Field(description="Names of matching gates")
    regulations: list[str] = Field(description="Distinct regulation frameworks triggered")
    required_reviewer_roles: list[str] = Field(description="Required reviewer roles from matching gates")


class ComplianceGateCreateRequest(BaseModel):
    """Request body for creating a compliance gate."""

    gate_name: str = Field(
        ...,
        min_length=3,
        max_length=255,
        description="Unique gate name within the tenant",
        examples=["hipaa-clinical-notes"],
    )
    regulation: str = Field(
        ...,
        pattern="^(hipaa|gdpr|sox|pci_dss|finra|fda)$",
        description="Regulatory framework: hipaa | gdpr | sox | pci_dss | finra | fda",
    )
    task_types: list[str] = Field(
        ...,
        min_length=1,
        description="Task types this gate applies to",
        examples=[["medical_diagnosis", "clinical_notes"]],
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Human-readable description of why this gate exists",
    )
    reviewer_role: str | None = Field(
        default=None,
        max_length=100,
        description="Required reviewer role when gate triggers (e.g., compliance_officer)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional gate configuration",
    )


class ComplianceGateResponse(BaseModel):
    """Response schema for a compliance gate."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    gate_name: str
    regulation: str
    task_types: list[str]
    description: str | None
    is_active: bool
    reviewer_role: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ComplianceGateListResponse(BaseModel):
    """List of compliance gates."""

    items: list[ComplianceGateResponse]
    total: int


# ---------------------------------------------------------------------------
# HITL review schemas
# ---------------------------------------------------------------------------


class HITLReviewCreateRequest(BaseModel):
    """Request body for creating a HITL review."""

    routing_decision_id: uuid.UUID = Field(
        ...,
        description="Parent RoutingDecision UUID that triggered this review",
    )
    task_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Task category",
    )
    ai_output: dict[str, Any] = Field(
        default_factory=dict,
        description="AI-generated output for the reviewer to assess",
    )
    reviewer_id: uuid.UUID | None = Field(
        default=None,
        description="Optional pre-assigned reviewer UUID",
    )
    priority: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Review priority: 1=low, 2=normal, 3=high, 4=critical",
    )


class HITLReviewDecisionRequest(BaseModel):
    """Request body for submitting a HITL review decision."""

    decision: str = Field(
        ...,
        pattern="^(approved|rejected|modified)$",
        description="Reviewer decision: approved | rejected | modified",
    )
    reviewer_output: dict[str, Any] = Field(
        default_factory=dict,
        description="Final output from the reviewer (may be edited AI output or original human output)",
    )
    reviewer_id: uuid.UUID = Field(
        ...,
        description="Reviewer user UUID",
    )
    reviewer_notes: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional reviewer comments explaining the decision",
    )


class HITLReviewResponse(BaseModel):
    """Response schema for a HITL review."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    routing_decision_id: uuid.UUID
    task_type: str
    status: str
    ai_output: dict[str, Any]
    reviewer_id: uuid.UUID | None
    review_started_at: datetime | None
    review_completed_at: datetime | None
    decision: str | None
    reviewer_output: dict[str, Any]
    reviewer_notes: str | None
    due_at: datetime | None
    priority: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HITLReviewListResponse(BaseModel):
    """Paginated list of HITL reviews."""

    items: list[HITLReviewResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Attribution schemas
# ---------------------------------------------------------------------------


class AttributionReportRequest(BaseModel):
    """Query parameters for attribution reports."""

    days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Lookback window in days",
    )
    task_type: str | None = Field(
        default=None,
        description="Optional task type filter",
    )


class AttributionReportResponse(BaseModel):
    """Attribution analytics report response."""

    tenant_id: uuid.UUID
    period_days: int
    task_type_filter: str | None
    total_tasks: int
    ai_handled: int
    human_handled: int
    hybrid_handled: int
    ai_contribution_pct: float
    human_contribution_pct: float
    total_time_saved_seconds: int
    by_task_type: list[dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Feedback schemas
# ---------------------------------------------------------------------------


class FeedbackCorrectionRequest(BaseModel):
    """Request body for submitting a feedback correction."""

    routing_decision_id: uuid.UUID = Field(
        ...,
        description="The routing decision being corrected",
    )
    task_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Task type for recalibration grouping",
    )
    original_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="AI confidence score at routing time",
    )
    original_routing: str = Field(
        ...,
        pattern="^(ai|human|hybrid)$",
        description="Original routing outcome: ai | human | hybrid",
    )
    corrected_routing: str = Field(
        ...,
        pattern="^(ai|human|hybrid)$",
        description="What the routing should have been: ai | human | hybrid",
    )
    correction_type: str = Field(
        ...,
        pattern="^(routing_error|confidence_overestimate|confidence_underestimate|output_error)$",
        description="Correction category",
    )
    hitl_review_id: uuid.UUID | None = Field(
        default=None,
        description="Optional linked HITL review",
    )
    correction_reason: str | None = Field(
        default=None,
        max_length=2000,
        description="Human explanation of why the routing was incorrect",
    )
    submitted_by: uuid.UUID | None = Field(
        default=None,
        description="Submitting user UUID",
    )


class FeedbackCorrectionResponse(BaseModel):
    """Response schema for a feedback correction."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    routing_decision_id: uuid.UUID
    hitl_review_id: uuid.UUID | None
    task_type: str
    original_confidence: float
    original_routing: str
    corrected_routing: str
    correction_reason: str | None
    correction_type: str
    submitted_by: uuid.UUID | None
    calibration_applied: bool
    calibration_applied_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CalibrationSummaryResponse(BaseModel):
    """Summary of feedback calibration data for a task type."""

    task_type: str
    total_corrections: int
    uncalibrated_count: int
    error_rate: float
    mean_confidence_delta: float
    correction_type_breakdown: dict[str, int]
    last_calibration_at: datetime | None

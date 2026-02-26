"""FastAPI router for the AumOS Human-AI Collaboration REST API.

All endpoints are prefixed with /api/v1. Authentication and tenant extraction
are handled by aumos-auth-gateway upstream; tenant_id is available via JWT.

Business logic is never implemented here — routes delegate entirely to services.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from aumos_common.errors import ConflictError, NotFoundError
from aumos_common.observability import get_logger

from aumos_human_ai_collab.api.schemas import (
    AttributionReportResponse,
    CalibrationSummaryResponse,
    ComplianceGateCreateRequest,
    ComplianceGateEvaluateRequest,
    ComplianceGateEvaluateResponse,
    ComplianceGateListResponse,
    ComplianceGateResponse,
    FeedbackCorrectionRequest,
    FeedbackCorrectionResponse,
    HITLReviewCreateRequest,
    HITLReviewDecisionRequest,
    HITLReviewListResponse,
    HITLReviewResponse,
    RoutingDecisionListResponse,
    RoutingDecisionResponse,
    RoutingEvaluateRequest,
)
from aumos_human_ai_collab.core.services import (
    AttributionService,
    ComplianceGateService,
    FeedbackService,
    HITLReviewService,
    RoutingService,
)

logger = get_logger(__name__)

router = APIRouter(tags=["human-ai-collab"])


# ---------------------------------------------------------------------------
# Dependency helpers — replaced by real DI in production startup
# ---------------------------------------------------------------------------


def _get_routing_service(request: Request) -> RoutingService:
    """Retrieve RoutingService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        RoutingService instance.
    """
    return request.app.state.routing_service  # type: ignore[no-any-return]


def _get_compliance_service(request: Request) -> ComplianceGateService:
    """Retrieve ComplianceGateService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        ComplianceGateService instance.
    """
    return request.app.state.compliance_service  # type: ignore[no-any-return]


def _get_hitl_service(request: Request) -> HITLReviewService:
    """Retrieve HITLReviewService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        HITLReviewService instance.
    """
    return request.app.state.hitl_service  # type: ignore[no-any-return]


def _get_attribution_service(request: Request) -> AttributionService:
    """Retrieve AttributionService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        AttributionService instance.
    """
    return request.app.state.attribution_service  # type: ignore[no-any-return]


def _get_feedback_service(request: Request) -> FeedbackService:
    """Retrieve FeedbackService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        FeedbackService instance.
    """
    return request.app.state.feedback_service  # type: ignore[no-any-return]


def _tenant_id_from_request(request: Request) -> uuid.UUID:
    """Extract tenant UUID from request headers (set by auth middleware).

    Falls back to a random UUID in development mode.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Tenant UUID.
    """
    tenant_header = request.headers.get("X-Tenant-ID")
    if tenant_header:
        return uuid.UUID(tenant_header)
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Routing endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/routing/evaluate",
    response_model=RoutingDecisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Evaluate task routing",
    description=(
        "Evaluate whether a task should be handled by AI, human, or hybrid. "
        "Applies compliance gates and confidence thresholds, then records the decision."
    ),
)
async def evaluate_routing(
    request_body: RoutingEvaluateRequest,
    request: Request,
    service: RoutingService = Depends(_get_routing_service),
) -> RoutingDecisionResponse:
    """Evaluate and record a routing decision for a task.

    Args:
        request_body: Routing evaluation parameters.
        request: FastAPI request for tenant extraction.
        service: RoutingService dependency.

    Returns:
        RoutingDecisionResponse with the routing outcome.

    Raises:
        HTTPException 400: If override_routing value is invalid.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        decision = await service.evaluate_routing(
            tenant_id=tenant_id,
            task_id=request_body.task_id,
            task_type=request_body.task_type,
            task_payload=request_body.task_payload,
            model_id=request_body.model_id,
            override_routing=request_body.override_routing,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "Routing evaluation API call",
        tenant_id=str(tenant_id),
        decision_id=str(decision.id),
        routing_outcome=decision.routing_outcome,
    )
    return RoutingDecisionResponse.model_validate(decision)


@router.get(
    "/routing/decisions",
    response_model=RoutingDecisionListResponse,
    summary="List routing decisions",
    description="List routing decisions for the current tenant with optional filters.",
)
async def list_routing_decisions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    task_type: str | None = Query(default=None),
    routing_outcome: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    request: Request = ...,  # type: ignore[assignment]
    service: RoutingService = Depends(_get_routing_service),
) -> RoutingDecisionListResponse:
    """List routing decisions for the current tenant.

    Args:
        page: 1-based page number.
        page_size: Results per page (max 100).
        task_type: Optional task type filter.
        routing_outcome: Optional routing outcome filter.
        date_from: Optional start date filter.
        date_to: Optional end date filter.
        request: FastAPI request for tenant extraction.
        service: RoutingService dependency.

    Returns:
        RoutingDecisionListResponse with pagination metadata.
    """
    tenant_id = _tenant_id_from_request(request)
    decisions, total = await service.list_decisions(
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
        task_type=task_type,
        routing_outcome=routing_outcome,
        date_from=date_from,
        date_to=date_to,
    )

    return RoutingDecisionListResponse(
        items=[RoutingDecisionResponse.model_validate(d) for d in decisions],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Compliance gate endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/compliance/gates/evaluate",
    response_model=ComplianceGateEvaluateResponse,
    summary="Evaluate compliance gate",
    description="Check whether a task type triggers any active compliance gates for the tenant.",
)
async def evaluate_compliance_gate(
    request_body: ComplianceGateEvaluateRequest,
    request: Request,
    service: ComplianceGateService = Depends(_get_compliance_service),
) -> ComplianceGateEvaluateResponse:
    """Evaluate compliance gates for a task type.

    Args:
        request_body: Task type to evaluate.
        request: FastAPI request for tenant extraction.
        service: ComplianceGateService dependency.

    Returns:
        ComplianceGateEvaluateResponse indicating whether routing is forced to human.
    """
    tenant_id = _tenant_id_from_request(request)
    result = await service.evaluate_gate(tenant_id=tenant_id, task_type=request_body.task_type)
    return ComplianceGateEvaluateResponse(**result)


@router.post(
    "/compliance/gates",
    response_model=ComplianceGateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create compliance gate",
    description="Create a regulatory compliance gate that forces human review for specified task types.",
)
async def create_compliance_gate(
    request_body: ComplianceGateCreateRequest,
    request: Request,
    service: ComplianceGateService = Depends(_get_compliance_service),
) -> ComplianceGateResponse:
    """Create a compliance gate.

    Args:
        request_body: Gate creation parameters.
        request: FastAPI request for tenant extraction.
        service: ComplianceGateService dependency.

    Returns:
        ComplianceGateResponse for the new gate.

    Raises:
        HTTPException 400: If regulation is invalid or task_types is empty.
        HTTPException 409: If a gate with the same name already exists.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        gate = await service.create_gate(
            tenant_id=tenant_id,
            gate_name=request_body.gate_name,
            regulation=request_body.regulation,
            task_types=request_body.task_types,
            description=request_body.description,
            reviewer_role=request_body.reviewer_role,
            metadata=request_body.metadata,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return ComplianceGateResponse.model_validate(gate)


@router.get(
    "/compliance/gates",
    response_model=ComplianceGateListResponse,
    summary="List compliance gates",
    description="List all compliance gates for the current tenant.",
)
async def list_compliance_gates(
    active_only: bool = Query(default=True),
    request: Request = ...,  # type: ignore[assignment]
    service: ComplianceGateService = Depends(_get_compliance_service),
) -> ComplianceGateListResponse:
    """List compliance gates for the current tenant.

    Args:
        active_only: If True, exclude soft-deleted gates.
        request: FastAPI request for tenant extraction.
        service: ComplianceGateService dependency.

    Returns:
        ComplianceGateListResponse with all matching gates.
    """
    tenant_id = _tenant_id_from_request(request)
    gates = await service.list_gates(tenant_id, active_only)

    return ComplianceGateListResponse(
        items=[ComplianceGateResponse.model_validate(g) for g in gates],
        total=len(gates),
    )


# ---------------------------------------------------------------------------
# HITL review endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/hitl/reviews",
    response_model=HITLReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create HITL review",
    description=(
        "Create a human-in-the-loop review task for a routing decision "
        "that was directed to human or hybrid."
    ),
)
async def create_hitl_review(
    request_body: HITLReviewCreateRequest,
    request: Request,
    service: HITLReviewService = Depends(_get_hitl_service),
) -> HITLReviewResponse:
    """Create a HITL review task.

    Args:
        request_body: Review creation parameters.
        request: FastAPI request for tenant extraction.
        service: HITLReviewService dependency.

    Returns:
        HITLReviewResponse with the new review in pending status.
    """
    tenant_id = _tenant_id_from_request(request)

    review = await service.create_review(
        tenant_id=tenant_id,
        routing_decision_id=request_body.routing_decision_id,
        task_type=request_body.task_type,
        ai_output=request_body.ai_output,
        reviewer_id=request_body.reviewer_id,
        priority=request_body.priority,
    )

    return HITLReviewResponse.model_validate(review)


@router.get(
    "/hitl/reviews",
    response_model=HITLReviewListResponse,
    summary="List HITL reviews",
    description="List HITL reviews for the current tenant with optional filters.",
)
async def list_hitl_reviews(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: str | None = Query(default=None, alias="status"),
    reviewer_id: uuid.UUID | None = Query(default=None),
    task_type: str | None = Query(default=None),
    request: Request = ...,  # type: ignore[assignment]
    service: HITLReviewService = Depends(_get_hitl_service),
) -> HITLReviewListResponse:
    """List HITL reviews for the current tenant.

    Args:
        page: 1-based page number.
        page_size: Results per page (max 100).
        status_filter: Optional status filter.
        reviewer_id: Optional reviewer UUID filter.
        task_type: Optional task type filter.
        request: FastAPI request for tenant extraction.
        service: HITLReviewService dependency.

    Returns:
        HITLReviewListResponse with pagination metadata.
    """
    tenant_id = _tenant_id_from_request(request)
    reviews, total = await service.list_reviews(
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
        status=status_filter,
        reviewer_id=reviewer_id,
        task_type=task_type,
    )

    return HITLReviewListResponse(
        items=[HITLReviewResponse.model_validate(r) for r in reviews],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/hitl/reviews/{review_id}",
    response_model=HITLReviewResponse,
    summary="Get HITL review",
    description="Retrieve a single HITL review by ID.",
)
async def get_hitl_review(
    review_id: uuid.UUID,
    request: Request,
    service: HITLReviewService = Depends(_get_hitl_service),
) -> HITLReviewResponse:
    """Retrieve a single HITL review.

    Args:
        review_id: HITLReview UUID.
        request: FastAPI request for tenant extraction.
        service: HITLReviewService dependency.

    Returns:
        HITLReviewResponse.

    Raises:
        HTTPException 404: If review not found.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        review = await service.get_review(review_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return HITLReviewResponse.model_validate(review)


@router.patch(
    "/hitl/reviews/{review_id}",
    response_model=HITLReviewResponse,
    summary="Submit review decision",
    description=(
        "Submit a human reviewer's decision on a HITL review task. "
        "Decision is immutable once submitted."
    ),
)
async def submit_hitl_decision(
    review_id: uuid.UUID,
    request_body: HITLReviewDecisionRequest,
    request: Request,
    service: HITLReviewService = Depends(_get_hitl_service),
) -> HITLReviewResponse:
    """Submit a reviewer decision on a HITL review.

    Args:
        review_id: HITLReview UUID.
        request_body: Decision with outcome and reviewer output.
        request: FastAPI request for tenant extraction.
        service: HITLReviewService dependency.

    Returns:
        Updated HITLReviewResponse with decision recorded.

    Raises:
        HTTPException 404: If review not found.
        HTTPException 409: If review is already decided.
        HTTPException 400: If decision value is invalid.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        review = await service.submit_decision(
            review_id=review_id,
            tenant_id=tenant_id,
            decision=request_body.decision,
            reviewer_output=request_body.reviewer_output,
            reviewer_id=request_body.reviewer_id,
            reviewer_notes=request_body.reviewer_notes,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return HITLReviewResponse.model_validate(review)


# ---------------------------------------------------------------------------
# Attribution endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/attribution/reports",
    response_model=AttributionReportResponse,
    summary="Attribution analytics report",
    description=(
        "Generate a productivity attribution report showing AI vs. human contribution "
        "across task types for the current tenant."
    ),
)
async def get_attribution_report(
    days: int = Query(default=30, ge=1, le=365),
    task_type: str | None = Query(default=None),
    request: Request = ...,  # type: ignore[assignment]
    service: AttributionService = Depends(_get_attribution_service),
) -> AttributionReportResponse:
    """Generate an attribution analytics report.

    Args:
        days: Lookback window in days.
        task_type: Optional task type filter.
        request: FastAPI request for tenant extraction.
        service: AttributionService dependency.

    Returns:
        AttributionReportResponse with AI/human contribution breakdown.
    """
    tenant_id = _tenant_id_from_request(request)
    report_data = await service.get_report(
        tenant_id=tenant_id,
        days=days,
        task_type=task_type,
    )

    return AttributionReportResponse(
        tenant_id=tenant_id,
        period_days=days,
        task_type_filter=task_type,
        total_tasks=report_data.get("total_tasks", 0),
        ai_handled=report_data.get("ai_handled", 0),
        human_handled=report_data.get("human_handled", 0),
        hybrid_handled=report_data.get("hybrid_handled", 0),
        ai_contribution_pct=report_data.get("ai_contribution_pct", 0.0),
        human_contribution_pct=report_data.get("human_contribution_pct", 0.0),
        total_time_saved_seconds=report_data.get("total_time_saved_seconds", 0),
        by_task_type=report_data.get("by_task_type", []),
        generated_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feedback",
    response_model=FeedbackCorrectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit correction feedback",
    description=(
        "Submit a human correction for a routing decision to recalibrate AI confidence. "
        "Recalibration triggers automatically when the sample threshold is reached."
    ),
)
async def submit_feedback(
    request_body: FeedbackCorrectionRequest,
    request: Request,
    service: FeedbackService = Depends(_get_feedback_service),
) -> FeedbackCorrectionResponse:
    """Submit a feedback correction for AI confidence recalibration.

    Args:
        request_body: Correction parameters.
        request: FastAPI request for tenant extraction.
        service: FeedbackService dependency.

    Returns:
        FeedbackCorrectionResponse with the recorded correction.

    Raises:
        HTTPException 400: If routing values or correction_type are invalid.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        correction = await service.submit_correction(
            tenant_id=tenant_id,
            routing_decision_id=request_body.routing_decision_id,
            task_type=request_body.task_type,
            original_confidence=request_body.original_confidence,
            original_routing=request_body.original_routing,
            corrected_routing=request_body.corrected_routing,
            correction_type=request_body.correction_type,
            hitl_review_id=request_body.hitl_review_id,
            correction_reason=request_body.correction_reason,
            submitted_by=request_body.submitted_by,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return FeedbackCorrectionResponse.model_validate(correction)


@router.get(
    "/feedback/calibration/{task_type}",
    response_model=CalibrationSummaryResponse,
    summary="Get calibration summary",
    description="Get feedback calibration summary and statistics for a specific task type.",
)
async def get_calibration_summary(
    task_type: str,
    request: Request,
    service: FeedbackService = Depends(_get_feedback_service),
) -> CalibrationSummaryResponse:
    """Get calibration summary for a task type.

    Args:
        task_type: Task type to summarise.
        request: FastAPI request for tenant extraction.
        service: FeedbackService dependency.

    Returns:
        CalibrationSummaryResponse with correction statistics.
    """
    tenant_id = _tenant_id_from_request(request)
    summary = await service.get_calibration_summary(tenant_id, task_type)

    return CalibrationSummaryResponse(
        task_type=task_type,
        total_corrections=summary.get("total_corrections", 0),
        uncalibrated_count=summary.get("uncalibrated_count", 0),
        error_rate=summary.get("error_rate", 0.0),
        mean_confidence_delta=summary.get("mean_confidence_delta", 0.0),
        correction_type_breakdown=summary.get("correction_type_breakdown", {}),
        last_calibration_at=summary.get("last_calibration_at"),
    )

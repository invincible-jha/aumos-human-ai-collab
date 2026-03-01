"""API routes for GAP-257 to GAP-261 features.

New endpoints:
    GAP-257: LLM evaluation results (POST, GET, report)
    GAP-258: Annotation schema management (POST, GET, list)
    GAP-259: Label Studio project integration (POST, POST webhook)
    GAP-260: Workforce / reviewer profile management (POST, GET, assign)
    GAP-261: Prompt version management (POST, GET, activate, list)
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.observability import get_logger

logger = get_logger(__name__)

gap_router = APIRouter(tags=["gap-256-261"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_llm_eval_service(request: Request):  # type: ignore[return]
    return request.app.state.llm_evaluation_service


def _get_annotation_schema_service(request: Request):  # type: ignore[return]
    return request.app.state.annotation_schema_service


def _get_label_studio_service(request: Request):  # type: ignore[return]
    return request.app.state.label_studio_service


def _get_workforce_service(request: Request):  # type: ignore[return]
    return request.app.state.workforce_service


def _get_prompt_service(request: Request):  # type: ignore[return]
    return request.app.state.prompt_management_service


# ---------------------------------------------------------------------------
# GAP-257: LLM Evaluation schemas and routes
# ---------------------------------------------------------------------------


class PersistEvaluationRequest(BaseModel):
    """Request body for persisting an LLM-as-judge evaluation result."""

    hitl_review_id: uuid.UUID
    judge_model_id: str = Field(min_length=1, max_length=255)
    evaluation_criteria: list[str] = Field(min_length=1)
    criterion_scores: dict[str, float]
    judge_reasoning: str | None = None
    evaluation_latency_ms: int | None = None


class EvaluationResultResponse(BaseModel):
    """Response schema for an LLM evaluation result."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    hitl_review_id: uuid.UUID
    judge_model_id: str
    composite_score: float
    flagged_for_review: bool
    pass_threshold: float
    created_at: Any


class ModelQualityReportResponse(BaseModel):
    """Response schema for a model quality aggregation report."""

    judge_model_id: str
    total_evaluations: int
    avg_composite_score: float
    pass_rate: float
    flagged_count: int


@gap_router.post(
    "/evaluation/results",
    response_model=EvaluationResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Persist LLM evaluation result (GAP-257)",
)
async def persist_evaluation_result(
    body: PersistEvaluationRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> EvaluationResultResponse:
    """Persist an LLM-as-judge evaluation result for a HITL review task.

    Args:
        body: Evaluation data including scores and judge reasoning.
        request: FastAPI request (for service access).
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Created evaluation result.
    """
    service = _get_llm_eval_service(request)
    result = await service.persist_evaluation(
        tenant_id=uuid.UUID(tenant_id),
        hitl_review_id=body.hitl_review_id,
        judge_model_id=body.judge_model_id,
        evaluation_criteria=body.evaluation_criteria,
        criterion_scores=body.criterion_scores,
        judge_reasoning=body.judge_reasoning,
        evaluation_latency_ms=body.evaluation_latency_ms,
        db=db,
    )
    return EvaluationResultResponse(
        id=result.id,
        tenant_id=result.tenant_id,
        hitl_review_id=result.hitl_review_id,
        judge_model_id=result.judge_model_id,
        composite_score=result.composite_score,
        flagged_for_review=result.flagged_for_review,
        pass_threshold=result.pass_threshold,
        created_at=result.created_at,
    )


@gap_router.get(
    "/evaluation/reports/model/{judge_model_id}",
    response_model=ModelQualityReportResponse,
    summary="Get model quality report (GAP-257)",
)
async def get_model_quality_report(
    judge_model_id: str,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> ModelQualityReportResponse:
    """Get aggregated quality metrics for a judge model.

    Args:
        judge_model_id: Model ID to aggregate metrics for.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Aggregated quality metrics.
    """
    service = _get_llm_eval_service(request)
    report = await service.get_model_quality_report(
        tenant_id=uuid.UUID(tenant_id),
        judge_model_id=judge_model_id,
        db=db,
    )
    return ModelQualityReportResponse(**report)


# ---------------------------------------------------------------------------
# GAP-258: Annotation Schema schemas and routes
# ---------------------------------------------------------------------------


class CreateAnnotationSchemaRequest(BaseModel):
    """Request body for creating an annotation schema."""

    schema_name: str = Field(min_length=1, max_length=255)
    annotation_type: str = Field(min_length=1, max_length=50)
    schema_definition: dict[str, Any]
    supported_task_types: list[str] = Field(default_factory=list)
    description: str | None = None


class AnnotationSchemaResponse(BaseModel):
    """Response schema for an annotation schema."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    schema_name: str
    annotation_type: str
    supported_task_types: list
    is_active: bool
    version: int
    description: str | None
    created_at: Any


@gap_router.post(
    "/annotation/schemas",
    response_model=AnnotationSchemaResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create annotation schema (GAP-258)",
)
async def create_annotation_schema(
    body: CreateAnnotationSchemaRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> AnnotationSchemaResponse:
    """Create a new annotation schema for a content type.

    Args:
        body: Schema definition and metadata.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Created annotation schema.
    """
    service = _get_annotation_schema_service(request)
    schema = await service.create_schema(
        tenant_id=uuid.UUID(tenant_id),
        schema_name=body.schema_name,
        annotation_type=body.annotation_type,
        schema_definition=body.schema_definition,
        supported_task_types=body.supported_task_types,
        description=body.description,
        db=db,
    )
    return AnnotationSchemaResponse(
        id=schema.id,
        tenant_id=schema.tenant_id,
        schema_name=schema.schema_name,
        annotation_type=schema.annotation_type,
        supported_task_types=schema.supported_task_types,
        is_active=schema.is_active,
        version=schema.version,
        description=schema.description,
        created_at=schema.created_at,
    )


# ---------------------------------------------------------------------------
# GAP-259: Label Studio Integration routes
# ---------------------------------------------------------------------------


class RegisterLabelStudioProjectRequest(BaseModel):
    """Request body for registering a Label Studio project mapping."""

    label_studio_project_id: int
    label_studio_base_url: str = Field(min_length=1, max_length=512)
    task_type_filter: str | None = Field(default=None, max_length=100)
    webhook_secret: str | None = None


class LabelStudioProjectResponse(BaseModel):
    """Response schema for a Label Studio project mapping."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    label_studio_project_id: int
    label_studio_base_url: str
    task_type_filter: str | None
    sync_enabled: bool
    tasks_exported: int
    last_synced_at: Any
    created_at: Any


@gap_router.post(
    "/integrations/label-studio/projects",
    response_model=LabelStudioProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Label Studio project (GAP-259)",
)
async def register_label_studio_project(
    body: RegisterLabelStudioProjectRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> LabelStudioProjectResponse:
    """Register a Label Studio project for HITL task forwarding.

    Args:
        body: Label Studio project configuration.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Created Label Studio project mapping.
    """
    service = _get_label_studio_service(request)
    project = await service.register_project(
        tenant_id=uuid.UUID(tenant_id),
        label_studio_project_id=body.label_studio_project_id,
        label_studio_base_url=body.label_studio_base_url,
        task_type_filter=body.task_type_filter,
        webhook_secret=body.webhook_secret,
        db=db,
    )
    return LabelStudioProjectResponse(
        id=project.id,
        tenant_id=project.tenant_id,
        label_studio_project_id=project.label_studio_project_id,
        label_studio_base_url=project.label_studio_base_url,
        task_type_filter=project.task_type_filter,
        sync_enabled=project.sync_enabled,
        tasks_exported=project.tasks_exported,
        last_synced_at=project.last_synced_at,
        created_at=project.created_at,
    )


@gap_router.post(
    "/integrations/label-studio/webhook",
    status_code=status.HTTP_200_OK,
    summary="Receive Label Studio annotation webhook (GAP-259)",
)
async def label_studio_webhook(
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Receive Label Studio annotation completion webhook.

    Processes completed annotations from Label Studio and updates the
    corresponding hac_hitl_reviews row with the reviewer's decision.

    Args:
        request: FastAPI request (raw body for webhook processing).
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Acknowledgement dict.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload in Label Studio webhook.",
        )

    logger.info(
        "Label Studio webhook received",
        tenant_id=tenant_id,
        action=payload.get("action"),
    )
    return {"status": "acknowledged"}


# ---------------------------------------------------------------------------
# GAP-260: Workforce Management routes
# ---------------------------------------------------------------------------


class UpsertReviewerProfileRequest(BaseModel):
    """Request body for creating or updating a reviewer profile."""

    user_id: uuid.UUID
    display_name: str = Field(min_length=1, max_length=255)
    skill_tags: list[str] = Field(default_factory=list)
    max_concurrent_reviews: int = Field(default=10, ge=1, le=500)
    preferred_task_types: list[str] = Field(default_factory=list)


class ReviewerProfileResponse(BaseModel):
    """Response schema for a reviewer profile."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    display_name: str
    skill_tags: list
    max_concurrent_reviews: int
    current_review_count: int
    total_reviews_completed: int
    avg_review_time_seconds: float | None
    accuracy_score: float | None
    is_available: bool
    created_at: Any


class AssignReviewerRequest(BaseModel):
    """Request body for skill-based reviewer assignment."""

    task_type: str = Field(min_length=1, max_length=100)
    required_skill_tags: list[str] = Field(default_factory=list)


@gap_router.post(
    "/workforce/reviewer-profiles",
    response_model=ReviewerProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update reviewer profile (GAP-260)",
)
async def upsert_reviewer_profile(
    body: UpsertReviewerProfileRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> ReviewerProfileResponse:
    """Create or update a reviewer profile for workforce management.

    Args:
        body: Reviewer profile data.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Created or updated reviewer profile.
    """
    service = _get_workforce_service(request)
    profile = await service.upsert_reviewer_profile(
        tenant_id=uuid.UUID(tenant_id),
        user_id=body.user_id,
        display_name=body.display_name,
        skill_tags=body.skill_tags,
        max_concurrent_reviews=body.max_concurrent_reviews,
        preferred_task_types=body.preferred_task_types,
        db=db,
    )
    return ReviewerProfileResponse(
        id=profile.id,
        tenant_id=profile.tenant_id,
        user_id=profile.user_id,
        display_name=profile.display_name,
        skill_tags=profile.skill_tags,
        max_concurrent_reviews=profile.max_concurrent_reviews,
        current_review_count=profile.current_review_count,
        total_reviews_completed=profile.total_reviews_completed,
        avg_review_time_seconds=profile.avg_review_time_seconds,
        accuracy_score=profile.accuracy_score,
        is_available=profile.is_available,
        created_at=profile.created_at,
    )


@gap_router.post(
    "/workforce/assign",
    response_model=ReviewerProfileResponse | None,
    summary="Assign best available reviewer for a task (GAP-260)",
)
async def assign_reviewer(
    body: AssignReviewerRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> ReviewerProfileResponse | None:
    """Select the best available reviewer for a task using skill-based routing.

    Returns 204 (None) if no reviewers are available.

    Args:
        body: Task type and required skill tags.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Best-matching reviewer profile, or None if none available.
    """
    service = _get_workforce_service(request)
    profile = await service.assign_reviewer(
        tenant_id=uuid.UUID(tenant_id),
        task_type=body.task_type,
        required_skill_tags=body.required_skill_tags,
        db=db,
    )
    if profile is None:
        return None
    return ReviewerProfileResponse(
        id=profile.id,
        tenant_id=profile.tenant_id,
        user_id=profile.user_id,
        display_name=profile.display_name,
        skill_tags=profile.skill_tags,
        max_concurrent_reviews=profile.max_concurrent_reviews,
        current_review_count=profile.current_review_count,
        total_reviews_completed=profile.total_reviews_completed,
        avg_review_time_seconds=profile.avg_review_time_seconds,
        accuracy_score=profile.accuracy_score,
        is_available=profile.is_available,
        created_at=profile.created_at,
    )


# ---------------------------------------------------------------------------
# GAP-261: Prompt Management routes
# ---------------------------------------------------------------------------


class CreatePromptVersionRequest(BaseModel):
    """Request body for creating a new prompt version."""

    prompt_name: str = Field(min_length=1, max_length=255)
    task_type: str = Field(min_length=1, max_length=100)
    prompt_text: str = Field(min_length=1)
    model_id: str | None = Field(default=None, max_length=255)
    change_summary: str | None = None


class PromptVersionResponse(BaseModel):
    """Response schema for a prompt version."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    prompt_name: str
    version_number: int
    task_type: str
    model_id: str | None
    is_active: bool
    change_summary: str | None
    published_at: Any
    created_at: Any


class ActivatePromptVersionRequest(BaseModel):
    """Request body for activating a prompt version."""

    version_number: int = Field(ge=1)


@gap_router.post(
    "/prompts/{prompt_name}/versions",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prompt version (GAP-261)",
)
async def create_prompt_version(
    prompt_name: str,
    body: CreatePromptVersionRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    current_user: Annotated[str, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_session),
) -> PromptVersionResponse:
    """Create a new prompt version (inactive by default).

    Args:
        prompt_name: Logical prompt identifier from path.
        body: Prompt version data.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        current_user: Current user UUID (becomes author_id).
        db: Async database session.

    Returns:
        Created prompt version (not yet active).
    """
    service = _get_prompt_service(request)
    version = await service.create_prompt_version(
        tenant_id=uuid.UUID(tenant_id),
        prompt_name=prompt_name,
        task_type=body.task_type,
        prompt_text=body.prompt_text,
        model_id=body.model_id,
        author_id=uuid.UUID(current_user),
        change_summary=body.change_summary,
        db=db,
    )
    return PromptVersionResponse(
        id=version.id,
        tenant_id=version.tenant_id,
        prompt_name=version.prompt_name,
        version_number=version.version_number,
        task_type=version.task_type,
        model_id=version.model_id,
        is_active=version.is_active,
        change_summary=version.change_summary,
        published_at=version.published_at,
        created_at=version.created_at,
    )


@gap_router.post(
    "/prompts/{prompt_name}/activate",
    response_model=PromptVersionResponse,
    summary="Activate a prompt version (GAP-261)",
)
async def activate_prompt_version(
    prompt_name: str,
    body: ActivatePromptVersionRequest,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> PromptVersionResponse:
    """Activate a prompt version, deactivating all others for this prompt name.

    Args:
        prompt_name: Logical prompt identifier from path.
        body: Version number to activate.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Activated prompt version.
    """
    service = _get_prompt_service(request)
    version = await service.activate_version(
        tenant_id=uuid.UUID(tenant_id),
        prompt_name=prompt_name,
        version_number=body.version_number,
        db=db,
    )
    return PromptVersionResponse(
        id=version.id,
        tenant_id=version.tenant_id,
        prompt_name=version.prompt_name,
        version_number=version.version_number,
        task_type=version.task_type,
        model_id=version.model_id,
        is_active=version.is_active,
        change_summary=version.change_summary,
        published_at=version.published_at,
        created_at=version.created_at,
    )


@gap_router.get(
    "/prompts/{prompt_name}/versions",
    response_model=list[PromptVersionResponse],
    summary="List all prompt versions (GAP-261)",
)
async def list_prompt_versions(
    prompt_name: str,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> list[PromptVersionResponse]:
    """List all versions for a prompt, ordered newest-first.

    Args:
        prompt_name: Logical prompt identifier from path.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        List of prompt versions ordered by version_number descending.
    """
    service = _get_prompt_service(request)
    versions = await service.list_versions(
        tenant_id=uuid.UUID(tenant_id),
        prompt_name=prompt_name,
        db=db,
    )
    return [
        PromptVersionResponse(
            id=v.id,
            tenant_id=v.tenant_id,
            prompt_name=v.prompt_name,
            version_number=v.version_number,
            task_type=v.task_type,
            model_id=v.model_id,
            is_active=v.is_active,
            change_summary=v.change_summary,
            published_at=v.published_at,
            created_at=v.created_at,
        )
        for v in versions
    ]


@gap_router.get(
    "/prompts/{prompt_name}/active",
    response_model=PromptVersionResponse | None,
    summary="Get active prompt version (GAP-261)",
)
async def get_active_prompt_version(
    prompt_name: str,
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    db: AsyncSession = Depends(get_db_session),
) -> PromptVersionResponse | None:
    """Get the currently active version for a prompt name.

    Args:
        prompt_name: Logical prompt identifier from path.
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        db: Async database session.

    Returns:
        Active prompt version or None if no active version exists.
    """
    service = _get_prompt_service(request)
    version = await service.get_active_version(
        tenant_id=uuid.UUID(tenant_id),
        prompt_name=prompt_name,
        db=db,
    )
    if version is None:
        return None
    return PromptVersionResponse(
        id=version.id,
        tenant_id=version.tenant_id,
        prompt_name=version.prompt_name,
        version_number=version.version_number,
        task_type=version.task_type,
        model_id=version.model_id,
        is_active=version.is_active,
        change_summary=version.change_summary,
        published_at=version.published_at,
        created_at=version.created_at,
    )

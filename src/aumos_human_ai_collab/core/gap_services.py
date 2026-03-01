"""Business logic services for GAP-256 to GAP-261.

New services:
    ReviewerUIService     — manage reviewer interface sessions (GAP-256)
    LLMEvaluationService  — LLM-as-judge quality evaluation (GAP-257)
    AnnotationSchemaService — multi-type annotation schema management (GAP-258)
    LabelStudioService    — Label Studio project integration (GAP-259)
    WorkforceService      — reviewer assignment and workload management (GAP-260)
    PromptManagementService — versioned prompt management (GAP-261)
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_human_ai_collab.core.gap_models import (
    HacAnnotationSchema,
    HacLabelStudioProject,
    HacLLMEvaluationResult,
    HacPromptVersion,
    HacReviewerInterface,
    HacReviewerProfile,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# GAP-256: Reviewer UI Service
# ---------------------------------------------------------------------------


class ReviewerUIService:
    """Manages server-rendered review interface sessions for HITL reviewers.

    Creates and tracks browser session state for the Jinja2-rendered review
    interface. Sessions expire after a configurable TTL to prevent stale
    review tasks from being acted on.

    Args:
        event_publisher: Kafka publisher for session lifecycle events.
        session_ttl_hours: Session expiry TTL in hours (default 8).
    """

    def __init__(
        self,
        event_publisher: EventPublisher,
        session_ttl_hours: int = 8,
    ) -> None:
        self._event_publisher = event_publisher
        self._session_ttl_hours = session_ttl_hours

    async def create_session(
        self,
        tenant_id: uuid.UUID,
        hitl_review_id: uuid.UUID,
        reviewer_id: uuid.UUID,
        annotation_type: str,
        db: AsyncSession,
    ) -> HacReviewerInterface:
        """Create a new reviewer interface session.

        Args:
            tenant_id: Tenant UUID.
            hitl_review_id: The HITL review this session is for.
            reviewer_id: User UUID of the reviewer.
            annotation_type: Content type: text | image | audio | video | document.
            db: Database session.

        Returns:
            Created HacReviewerInterface instance.
        """
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=self._session_ttl_hours)

        interface_session = HacReviewerInterface(
            tenant_id=tenant_id,
            hitl_review_id=hitl_review_id,
            reviewer_id=reviewer_id,
            session_token=session_token,
            annotation_type=annotation_type,
            current_state={},
            submitted=False,
            expires_at=expires_at,
        )
        db.add(interface_session)
        await db.flush()

        logger.info(
            "Reviewer interface session created",
            tenant_id=str(tenant_id),
            hitl_review_id=str(hitl_review_id),
            reviewer_id=str(reviewer_id),
            session_id=str(interface_session.id),
        )
        return interface_session

    async def save_session_state(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        state_update: dict[str, Any],
        db: AsyncSession,
    ) -> HacReviewerInterface:
        """Persist intermediate annotation state for an in-progress session.

        Args:
            session_id: Session UUID to update.
            tenant_id: Tenant UUID for RLS.
            state_update: Partial state to merge into current_state.
            db: Database session.

        Returns:
            Updated HacReviewerInterface.

        Raises:
            NotFoundError: If session not found.
            ConflictError: If session already submitted or expired.
        """
        result = await db.execute(
            select(HacReviewerInterface).where(
                and_(
                    HacReviewerInterface.id == session_id,
                    HacReviewerInterface.tenant_id == tenant_id,
                )
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise NotFoundError(
                message=f"Reviewer interface session {session_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        if session.submitted:
            raise ConflictError(
                message="Session already submitted — decisions are immutable.",
                error_code=ErrorCode.CONFLICT,
            )

        now = datetime.now(tz=timezone.utc)
        if session.expires_at < now:
            raise ConflictError(
                message="Reviewer interface session has expired.",
                error_code=ErrorCode.CONFLICT,
            )

        session.current_state = {**session.current_state, **state_update}
        session.last_activity_at = now
        await db.flush()
        return session

    async def mark_session_submitted(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> HacReviewerInterface:
        """Mark a session as submitted after the reviewer finalises their decision.

        Args:
            session_id: Session UUID.
            tenant_id: Tenant UUID for RLS.
            db: Database session.

        Returns:
            Updated HacReviewerInterface.

        Raises:
            NotFoundError: If session not found.
            ConflictError: If already submitted.
        """
        result = await db.execute(
            select(HacReviewerInterface).where(
                and_(
                    HacReviewerInterface.id == session_id,
                    HacReviewerInterface.tenant_id == tenant_id,
                )
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise NotFoundError(
                message=f"Reviewer interface session {session_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        if session.submitted:
            raise ConflictError(
                message="Session already submitted — decisions are immutable.",
                error_code=ErrorCode.CONFLICT,
            )

        session.submitted = True
        session.last_activity_at = datetime.now(tz=timezone.utc)
        await db.flush()
        return session


# ---------------------------------------------------------------------------
# GAP-257: LLM Evaluation Service
# ---------------------------------------------------------------------------


class LLMEvaluationService:
    """Stores and queries LLM-as-judge evaluation results.

    Works with the LLMEvaluatorAdapter (in adapters/llm_evaluator.py) which
    performs the actual LLM calls. This service handles persistence and
    triggers feedback recalibration when quality drops below threshold.

    Args:
        event_publisher: Kafka publisher for evaluation events.
        pass_threshold: Composite score below which evaluation is flagged (default 0.70).
    """

    def __init__(
        self,
        event_publisher: EventPublisher,
        pass_threshold: float = 0.70,
    ) -> None:
        self._event_publisher = event_publisher
        self._pass_threshold = pass_threshold

    async def persist_evaluation(
        self,
        tenant_id: uuid.UUID,
        hitl_review_id: uuid.UUID,
        judge_model_id: str,
        evaluation_criteria: list[str],
        criterion_scores: dict[str, float],
        judge_reasoning: str | None,
        evaluation_latency_ms: int | None,
        db: AsyncSession,
    ) -> HacLLMEvaluationResult:
        """Persist an LLM evaluation result and flag low-quality outputs.

        Args:
            tenant_id: Tenant UUID.
            hitl_review_id: The HITL review this evaluation is for.
            judge_model_id: Model ID of the judge LLM.
            evaluation_criteria: List of criterion names evaluated.
            criterion_scores: Per-criterion scores (0.0-1.0).
            judge_reasoning: Free-text reasoning from the judge LLM.
            evaluation_latency_ms: Evaluation time in milliseconds.
            db: Database session.

        Returns:
            Created HacLLMEvaluationResult.
        """
        if criterion_scores:
            composite_score = sum(criterion_scores.values()) / len(criterion_scores)
        else:
            composite_score = 0.0

        flagged = composite_score < self._pass_threshold

        evaluation = HacLLMEvaluationResult(
            tenant_id=tenant_id,
            hitl_review_id=hitl_review_id,
            judge_model_id=judge_model_id,
            evaluation_criteria=evaluation_criteria,
            criterion_scores=criterion_scores,
            composite_score=round(composite_score, 4),
            judge_reasoning=judge_reasoning,
            pass_threshold=self._pass_threshold,
            flagged_for_review=flagged,
            evaluation_latency_ms=evaluation_latency_ms,
            evaluation_metadata={},
        )
        db.add(evaluation)
        await db.flush()

        if flagged:
            logger.warning(
                "LLM output flagged for priority review",
                tenant_id=str(tenant_id),
                hitl_review_id=str(hitl_review_id),
                composite_score=composite_score,
                pass_threshold=self._pass_threshold,
            )

        await self._event_publisher.publish(
            topic=Topics.HAC_LLM_EVALUATED,
            payload={
                "tenant_id": str(tenant_id),
                "hitl_review_id": str(hitl_review_id),
                "composite_score": composite_score,
                "flagged_for_review": flagged,
            },
        )

        return evaluation

    async def get_model_quality_report(
        self,
        tenant_id: uuid.UUID,
        judge_model_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Aggregate quality metrics for a specific judge model.

        Args:
            tenant_id: Tenant UUID.
            judge_model_id: Model ID to aggregate metrics for.
            db: Database session.

        Returns:
            Dict with avg_score, pass_rate, total_evaluations, flagged_count.
        """
        result = await db.execute(
            select(HacLLMEvaluationResult).where(
                and_(
                    HacLLMEvaluationResult.tenant_id == tenant_id,
                    HacLLMEvaluationResult.judge_model_id == judge_model_id,
                )
            )
        )
        evaluations = result.scalars().all()

        if not evaluations:
            return {
                "judge_model_id": judge_model_id,
                "total_evaluations": 0,
                "avg_composite_score": 0.0,
                "pass_rate": 0.0,
                "flagged_count": 0,
            }

        scores = [e.composite_score for e in evaluations]
        avg_score = sum(scores) / len(scores)
        flagged = sum(1 for e in evaluations if e.flagged_for_review)

        return {
            "judge_model_id": judge_model_id,
            "total_evaluations": len(evaluations),
            "avg_composite_score": round(avg_score, 4),
            "pass_rate": round((len(evaluations) - flagged) / len(evaluations), 4),
            "flagged_count": flagged,
        }


# ---------------------------------------------------------------------------
# GAP-258: Annotation Schema Service
# ---------------------------------------------------------------------------


class AnnotationSchemaService:
    """Manages multi-type annotation schema definitions.

    Annotation schemas define the structure and UI configuration for reviewing
    different content types (text, image, audio, video, document). Schemas are
    immutable once in use — changes require creating a new version.

    Args:
        event_publisher: Kafka publisher for schema lifecycle events.
    """

    VALID_ANNOTATION_TYPES: frozenset[str] = frozenset(
        {"text", "image", "audio", "video", "document", "structured_data"}
    )

    def __init__(self, event_publisher: EventPublisher) -> None:
        self._event_publisher = event_publisher

    async def create_schema(
        self,
        tenant_id: uuid.UUID,
        schema_name: str,
        annotation_type: str,
        schema_definition: dict[str, Any],
        supported_task_types: list[str],
        description: str | None,
        db: AsyncSession,
    ) -> HacAnnotationSchema:
        """Create a new annotation schema definition.

        Args:
            tenant_id: Tenant UUID.
            schema_name: Unique schema identifier within the tenant.
            annotation_type: Content type (text | image | audio | video | document).
            schema_definition: JSON schema for annotation fields and tools.
            supported_task_types: Task types this schema applies to.
            description: Human-readable description.
            db: Database session.

        Returns:
            Created HacAnnotationSchema.

        Raises:
            ConflictError: If schema_name already exists for tenant.
        """
        if annotation_type not in self.VALID_ANNOTATION_TYPES:
            raise ValueError(
                f"Invalid annotation_type: {annotation_type!r}. "
                f"Must be one of: {sorted(self.VALID_ANNOTATION_TYPES)}"
            )

        existing = await db.execute(
            select(HacAnnotationSchema).where(
                and_(
                    HacAnnotationSchema.tenant_id == tenant_id,
                    HacAnnotationSchema.schema_name == schema_name,
                )
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                message=f"Annotation schema '{schema_name}' already exists for this tenant.",
                error_code=ErrorCode.CONFLICT,
            )

        schema = HacAnnotationSchema(
            tenant_id=tenant_id,
            schema_name=schema_name,
            annotation_type=annotation_type,
            description=description,
            schema_definition=schema_definition,
            supported_task_types=supported_task_types,
            is_active=True,
            version=1,
        )
        db.add(schema)
        await db.flush()

        logger.info(
            "Annotation schema created",
            tenant_id=str(tenant_id),
            schema_name=schema_name,
            annotation_type=annotation_type,
        )
        return schema

    async def get_schema_for_task_type(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        annotation_type: str,
        db: AsyncSession,
    ) -> HacAnnotationSchema | None:
        """Find the best matching annotation schema for a task type.

        Args:
            tenant_id: Tenant UUID.
            task_type: Task type string to match.
            annotation_type: Content type to filter by.
            db: Database session.

        Returns:
            Matching HacAnnotationSchema or None.
        """
        result = await db.execute(
            select(HacAnnotationSchema).where(
                and_(
                    HacAnnotationSchema.tenant_id == tenant_id,
                    HacAnnotationSchema.annotation_type == annotation_type,
                    HacAnnotationSchema.is_active.is_(True),
                )
            )
        )
        schemas = result.scalars().all()
        for schema in schemas:
            if task_type in (schema.supported_task_types or []):
                return schema
        return None


# ---------------------------------------------------------------------------
# GAP-259: Label Studio Integration Service
# ---------------------------------------------------------------------------


class LabelStudioService:
    """Manages Label Studio project mappings and sync state.

    Tracks which AumOS task types are mapped to Label Studio projects,
    records export counts, and manages webhook configuration. The actual
    HTTP calls to Label Studio are handled by the LabelStudioAdapter
    in adapters/label_studio_adapter.py.

    Args:
        event_publisher: Kafka publisher for integration events.
    """

    def __init__(self, event_publisher: EventPublisher) -> None:
        self._event_publisher = event_publisher

    async def register_project(
        self,
        tenant_id: uuid.UUID,
        label_studio_project_id: int,
        label_studio_base_url: str,
        task_type_filter: str | None,
        webhook_secret: str | None,
        db: AsyncSession,
    ) -> HacLabelStudioProject:
        """Register a Label Studio project mapping for a tenant.

        Args:
            tenant_id: Tenant UUID.
            label_studio_project_id: Label Studio's internal project ID.
            label_studio_base_url: Base URL of the Label Studio instance.
            task_type_filter: Task type to forward to this project (None = all).
            webhook_secret: Shared secret for webhook verification.
            db: Database session.

        Returns:
            Created HacLabelStudioProject.

        Raises:
            ConflictError: If this project ID is already registered for the tenant.
        """
        existing = await db.execute(
            select(HacLabelStudioProject).where(
                and_(
                    HacLabelStudioProject.tenant_id == tenant_id,
                    HacLabelStudioProject.label_studio_project_id == label_studio_project_id,
                )
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                message=f"Label Studio project {label_studio_project_id} already registered.",
                error_code=ErrorCode.CONFLICT,
            )

        project = HacLabelStudioProject(
            tenant_id=tenant_id,
            label_studio_project_id=label_studio_project_id,
            label_studio_base_url=label_studio_base_url.rstrip("/"),
            task_type_filter=task_type_filter,
            webhook_secret=webhook_secret,
            sync_enabled=True,
            tasks_exported=0,
        )
        db.add(project)
        await db.flush()

        logger.info(
            "Label Studio project registered",
            tenant_id=str(tenant_id),
            label_studio_project_id=label_studio_project_id,
            task_type_filter=task_type_filter,
        )
        return project

    async def record_task_export(
        self,
        tenant_id: uuid.UUID,
        project_id: uuid.UUID,
        tasks_exported_count: int,
        db: AsyncSession,
    ) -> HacLabelStudioProject:
        """Update export count and last_synced_at after a successful export.

        Args:
            tenant_id: Tenant UUID.
            project_id: HacLabelStudioProject UUID.
            tasks_exported_count: Number of tasks exported in this batch.
            db: Database session.

        Returns:
            Updated HacLabelStudioProject.

        Raises:
            NotFoundError: If project not found.
        """
        result = await db.execute(
            select(HacLabelStudioProject).where(
                and_(
                    HacLabelStudioProject.id == project_id,
                    HacLabelStudioProject.tenant_id == tenant_id,
                )
            )
        )
        project = result.scalar_one_or_none()
        if project is None:
            raise NotFoundError(
                message=f"Label Studio project {project_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        project.tasks_exported += tasks_exported_count
        project.last_synced_at = datetime.now(tz=timezone.utc)
        await db.flush()
        return project


# ---------------------------------------------------------------------------
# GAP-260: Workforce Management Service
# ---------------------------------------------------------------------------


class WorkforceService:
    """Reviewer assignment, workload balancing, and skill-based routing.

    Selects the best available reviewer for a HITL task based on:
    1. Skill tag match with the task type.
    2. Current queue depth (prefer least loaded).
    3. Average review time (prefer faster reviewers as tiebreaker).

    Args:
        event_publisher: Kafka publisher for assignment events.
    """

    def __init__(self, event_publisher: EventPublisher) -> None:
        self._event_publisher = event_publisher

    async def upsert_reviewer_profile(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        display_name: str,
        skill_tags: list[str],
        max_concurrent_reviews: int,
        preferred_task_types: list[str],
        db: AsyncSession,
    ) -> HacReviewerProfile:
        """Create or update a reviewer profile.

        Args:
            tenant_id: Tenant UUID.
            user_id: User UUID.
            display_name: Display name for dashboards.
            skill_tags: List of skill/expertise tags.
            max_concurrent_reviews: Maximum simultaneous reviews.
            preferred_task_types: Ordered list of preferred task types.
            db: Database session.

        Returns:
            Created or updated HacReviewerProfile.
        """
        result = await db.execute(
            select(HacReviewerProfile).where(
                and_(
                    HacReviewerProfile.tenant_id == tenant_id,
                    HacReviewerProfile.user_id == user_id,
                )
            )
        )
        profile = result.scalar_one_or_none()

        if profile is None:
            profile = HacReviewerProfile(
                tenant_id=tenant_id,
                user_id=user_id,
                display_name=display_name,
                skill_tags=skill_tags,
                max_concurrent_reviews=max_concurrent_reviews,
                current_review_count=0,
                total_reviews_completed=0,
                is_available=True,
                preferred_task_types=preferred_task_types,
            )
            db.add(profile)
        else:
            profile.display_name = display_name
            profile.skill_tags = skill_tags
            profile.max_concurrent_reviews = max_concurrent_reviews
            profile.preferred_task_types = preferred_task_types

        await db.flush()
        return profile

    async def assign_reviewer(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        required_skill_tags: list[str],
        db: AsyncSession,
    ) -> HacReviewerProfile | None:
        """Select the best available reviewer for a task.

        Selection algorithm:
        1. Filter to profiles that are available and under capacity.
        2. Score by Jaccard similarity between required_skill_tags and profile.skill_tags.
        3. Among equal skill matches, prefer lowest current_review_count.
        4. Among equal queue depth, prefer lowest avg_review_time_seconds.

        Args:
            tenant_id: Tenant UUID.
            task_type: Task type to route.
            required_skill_tags: Skill tags required for this task.
            db: Database session.

        Returns:
            Best-matching available HacReviewerProfile, or None if none available.
        """
        result = await db.execute(
            select(HacReviewerProfile).where(
                and_(
                    HacReviewerProfile.tenant_id == tenant_id,
                    HacReviewerProfile.is_available.is_(True),
                )
            )
        )
        candidates = result.scalars().all()

        # Filter to under-capacity reviewers
        available = [
            p for p in candidates
            if p.current_review_count < p.max_concurrent_reviews
        ]

        if not available:
            return None

        required_set = set(required_skill_tags)

        def _score(profile: HacReviewerProfile) -> tuple[float, int, float]:
            skill_set = set(profile.skill_tags or [])
            union = required_set | skill_set
            jaccard = len(required_set & skill_set) / len(union) if union else 0.0
            avg_time = profile.avg_review_time_seconds or float("inf")
            # Sort key: descending jaccard, ascending queue depth, ascending avg time
            return (-jaccard, profile.current_review_count, avg_time)

        best = sorted(available, key=_score)[0]
        best.current_review_count += 1
        await db.flush()

        logger.info(
            "Reviewer assigned",
            tenant_id=str(tenant_id),
            reviewer_id=str(best.user_id),
            task_type=task_type,
            current_review_count=best.current_review_count,
        )
        return best

    async def release_reviewer_slot(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        db: AsyncSession,
    ) -> None:
        """Decrement a reviewer's active review count after task completion.

        Args:
            tenant_id: Tenant UUID.
            user_id: Reviewer user UUID.
            db: Database session.
        """
        result = await db.execute(
            select(HacReviewerProfile).where(
                and_(
                    HacReviewerProfile.tenant_id == tenant_id,
                    HacReviewerProfile.user_id == user_id,
                )
            )
        )
        profile = result.scalar_one_or_none()
        if profile is not None:
            profile.current_review_count = max(0, profile.current_review_count - 1)
            profile.total_reviews_completed += 1
            await db.flush()


# ---------------------------------------------------------------------------
# GAP-261: Prompt Management Service
# ---------------------------------------------------------------------------


class PromptManagementService:
    """Version and manage prompts alongside routing decisions.

    Implements Humanloop-style prompt versioning: each prompt has a name
    (logical identifier) and a version number. Only one version per prompt
    name can be active (is_active=True) at a time. Activating a new version
    deactivates the previous one.

    Args:
        event_publisher: Kafka publisher for prompt lifecycle events.
    """

    def __init__(self, event_publisher: EventPublisher) -> None:
        self._event_publisher = event_publisher

    async def create_prompt_version(
        self,
        tenant_id: uuid.UUID,
        prompt_name: str,
        task_type: str,
        prompt_text: str,
        model_id: str | None,
        author_id: uuid.UUID | None,
        change_summary: str | None,
        db: AsyncSession,
    ) -> HacPromptVersion:
        """Create a new prompt version (not yet active).

        The version number is auto-incremented from the highest existing
        version for this prompt_name within the tenant.

        Args:
            tenant_id: Tenant UUID.
            prompt_name: Logical prompt identifier.
            task_type: Task type this prompt is used for.
            prompt_text: Full prompt template text.
            model_id: Optional model ID this prompt is optimised for.
            author_id: Optional user UUID of the author.
            change_summary: Description of what changed.
            db: Database session.

        Returns:
            Created HacPromptVersion (not active — call activate_version() to activate).
        """
        result = await db.execute(
            select(HacPromptVersion).where(
                and_(
                    HacPromptVersion.tenant_id == tenant_id,
                    HacPromptVersion.prompt_name == prompt_name,
                )
            )
        )
        existing_versions = result.scalars().all()
        next_version = max((v.version_number for v in existing_versions), default=0) + 1

        prompt_version = HacPromptVersion(
            tenant_id=tenant_id,
            prompt_name=prompt_name,
            version_number=next_version,
            task_type=task_type,
            prompt_text=prompt_text,
            model_id=model_id,
            is_active=False,
            author_id=author_id,
            change_summary=change_summary,
            performance_metrics={},
        )
        db.add(prompt_version)
        await db.flush()

        logger.info(
            "Prompt version created",
            tenant_id=str(tenant_id),
            prompt_name=prompt_name,
            version_number=next_version,
        )
        return prompt_version

    async def activate_version(
        self,
        tenant_id: uuid.UUID,
        prompt_name: str,
        version_number: int,
        db: AsyncSession,
    ) -> HacPromptVersion:
        """Activate a prompt version, deactivating all others for this name.

        Args:
            tenant_id: Tenant UUID.
            prompt_name: Logical prompt identifier.
            version_number: Version number to activate.
            db: Database session.

        Returns:
            Activated HacPromptVersion.

        Raises:
            NotFoundError: If the specified version does not exist.
        """
        result = await db.execute(
            select(HacPromptVersion).where(
                and_(
                    HacPromptVersion.tenant_id == tenant_id,
                    HacPromptVersion.prompt_name == prompt_name,
                )
            )
        )
        all_versions = result.scalars().all()

        target: HacPromptVersion | None = None
        for version in all_versions:
            if version.version_number == version_number:
                target = version
            version.is_active = False  # deactivate all

        if target is None:
            raise NotFoundError(
                message=f"Prompt '{prompt_name}' version {version_number} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        target.is_active = True
        target.published_at = datetime.now(tz=timezone.utc)
        await db.flush()

        await self._event_publisher.publish(
            topic=Topics.HAC_PROMPT_ACTIVATED,
            payload={
                "tenant_id": str(tenant_id),
                "prompt_name": prompt_name,
                "version_number": version_number,
            },
        )

        logger.info(
            "Prompt version activated",
            tenant_id=str(tenant_id),
            prompt_name=prompt_name,
            version_number=version_number,
        )
        return target

    async def get_active_version(
        self,
        tenant_id: uuid.UUID,
        prompt_name: str,
        db: AsyncSession,
    ) -> HacPromptVersion | None:
        """Get the currently active version for a prompt name.

        Args:
            tenant_id: Tenant UUID.
            prompt_name: Logical prompt identifier.
            db: Database session.

        Returns:
            Active HacPromptVersion or None if no active version.
        """
        result = await db.execute(
            select(HacPromptVersion).where(
                and_(
                    HacPromptVersion.tenant_id == tenant_id,
                    HacPromptVersion.prompt_name == prompt_name,
                    HacPromptVersion.is_active.is_(True),
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_versions(
        self,
        tenant_id: uuid.UUID,
        prompt_name: str,
        db: AsyncSession,
    ) -> list[HacPromptVersion]:
        """List all versions for a prompt name, ordered by version_number desc.

        Args:
            tenant_id: Tenant UUID.
            prompt_name: Logical prompt identifier.
            db: Database session.

        Returns:
            List of HacPromptVersion ordered by version_number descending.
        """
        result = await db.execute(
            select(HacPromptVersion).where(
                and_(
                    HacPromptVersion.tenant_id == tenant_id,
                    HacPromptVersion.prompt_name == prompt_name,
                )
            )
        )
        versions = result.scalars().all()
        return sorted(versions, key=lambda v: v.version_number, reverse=True)

    async def record_performance_metrics(
        self,
        tenant_id: uuid.UUID,
        version_id: uuid.UUID,
        metrics: dict[str, Any],
        db: AsyncSession,
    ) -> HacPromptVersion:
        """Update performance metrics for a prompt version after evaluation.

        Args:
            tenant_id: Tenant UUID.
            version_id: HacPromptVersion UUID.
            metrics: Metrics dict to merge into existing performance_metrics.
            db: Database session.

        Returns:
            Updated HacPromptVersion.

        Raises:
            NotFoundError: If version not found.
        """
        result = await db.execute(
            select(HacPromptVersion).where(
                and_(
                    HacPromptVersion.id == version_id,
                    HacPromptVersion.tenant_id == tenant_id,
                )
            )
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise NotFoundError(
                message=f"Prompt version {version_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        version.performance_metrics = {**(version.performance_metrics or {}), **metrics}
        await db.flush()
        return version

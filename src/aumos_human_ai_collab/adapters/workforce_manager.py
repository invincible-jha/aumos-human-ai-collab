"""Reviewer workforce management adapter for aumos-human-ai-collab.

Handles skill-based reviewer assignment, workload balancing, capacity
tracking, and availability management for the HITL review pool.
Reads and updates HacReviewerProfile records to assign reviews to the
most appropriate available reviewer.

Gap Coverage: GAP-260 (Workforce Management)
"""

import uuid
from dataclasses import dataclass
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewerSummary:
    """Lightweight reviewer summary used during assignment decisions.

    Attributes:
        user_id: Reviewer's user UUID.
        display_name: Display name for logging and dashboards.
        skill_tags: List of expertise tags.
        current_review_count: Active review assignments.
        max_concurrent_reviews: Capacity ceiling.
        is_available: Whether the reviewer is active/not on leave.
        preferred_task_types: Ordered list of preferred task types.
        accuracy_score: Historical accuracy score (None if no history).
    """

    user_id: uuid.UUID
    display_name: str
    skill_tags: list[str]
    current_review_count: int
    max_concurrent_reviews: int
    is_available: bool
    preferred_task_types: list[str]
    accuracy_score: float | None


@dataclass
class AssignmentResult:
    """Result of a reviewer assignment attempt.

    Attributes:
        assigned: True if a reviewer was successfully assigned.
        reviewer_id: UUID of the assigned reviewer (None if unassigned).
        reviewer_name: Display name of the assigned reviewer.
        assignment_reason: Short explanation of why this reviewer was selected.
    """

    assigned: bool
    reviewer_id: uuid.UUID | None
    reviewer_name: str | None
    assignment_reason: str


class WorkforceManager:
    """Manages HITL reviewer assignment using skill-based routing and load balancing.

    Selects the best available reviewer for each HITL review task based on:
    1. Skill tag overlap with the task type
    2. Preferred task type affinity
    3. Current workload (favours reviewers with capacity headroom)
    4. Historical accuracy score (prefers higher accuracy reviewers)

    Does not perform database writes directly — delegates to the repository
    and uses an event publisher to notify when reviewers are overloaded.

    Args:
        reviewer_repository: Repository for HacReviewerProfile records.
        event_publisher: Kafka event publisher for workforce events.
        overload_threshold_pct: Workload % above which a warning event fires (default: 90).
    """

    def __init__(
        self,
        reviewer_repository: Any,
        event_publisher: Any,
        overload_threshold_pct: float = 90.0,
    ) -> None:
        """Initialize the workforce manager.

        Args:
            reviewer_repository: Repository for reviewer profiles.
            event_publisher: Kafka publisher for overload/capacity events.
            overload_threshold_pct: Capacity % that triggers an overload warning.
        """
        self._reviewer_repo = reviewer_repository
        self._event_publisher = event_publisher
        self._overload_threshold_pct = overload_threshold_pct

    async def assign_reviewer(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        required_skills: list[str],
        exclude_reviewer_ids: list[uuid.UUID] | None = None,
    ) -> AssignmentResult:
        """Find and assign the best available reviewer for a task.

        Queries all available reviewers for the tenant, scores each candidate,
        and returns the highest-scoring reviewer who has capacity.

        Args:
            tenant_id: The owning tenant.
            task_type: The task type string (used for skill and preference matching).
            required_skills: List of skill tags the task requires.
            exclude_reviewer_ids: Reviewer IDs to exclude (e.g., conflict of interest).

        Returns:
            AssignmentResult with the selected reviewer or unassigned status.
        """
        try:
            reviewers = await self._reviewer_repo.list_available(tenant_id=tenant_id)
        except Exception as exc:
            logger.error(
                "Failed to list reviewers for assignment",
                tenant_id=str(tenant_id),
                task_type=task_type,
                error=str(exc),
            )
            return AssignmentResult(
                assigned=False,
                reviewer_id=None,
                reviewer_name=None,
                assignment_reason="reviewer list unavailable",
            )

        excluded = set(exclude_reviewer_ids or [])
        candidates = [
            r for r in reviewers
            if r.is_available
            and r.current_review_count < r.max_concurrent_reviews
            and r.user_id not in excluded
        ]

        if not candidates:
            logger.warning(
                "No available reviewers for task assignment",
                tenant_id=str(tenant_id),
                task_type=task_type,
                total_reviewers=len(reviewers),
            )
            return AssignmentResult(
                assigned=False,
                reviewer_id=None,
                reviewer_name=None,
                assignment_reason="no reviewers with capacity available",
            )

        scored = [
            (self._score_reviewer(reviewer, task_type, required_skills), reviewer)
            for reviewer in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_reviewer = scored[0]

        logger.info(
            "Reviewer assigned to task",
            tenant_id=str(tenant_id),
            reviewer_id=str(best_reviewer.user_id),
            reviewer_name=best_reviewer.display_name,
            task_type=task_type,
            assignment_score=round(best_score, 3),
            current_load=best_reviewer.current_review_count,
            max_load=best_reviewer.max_concurrent_reviews,
        )

        # Check if reviewer is approaching capacity
        new_load = best_reviewer.current_review_count + 1
        capacity_pct = (new_load / best_reviewer.max_concurrent_reviews) * 100
        if capacity_pct >= self._overload_threshold_pct:
            logger.warning(
                "Reviewer approaching capacity limit",
                reviewer_id=str(best_reviewer.user_id),
                capacity_pct=round(capacity_pct, 1),
                threshold_pct=self._overload_threshold_pct,
            )

        return AssignmentResult(
            assigned=True,
            reviewer_id=best_reviewer.user_id,
            reviewer_name=best_reviewer.display_name,
            assignment_reason=(
                f"score={round(best_score, 3)}, skill_match, "
                f"load={best_reviewer.current_review_count}/{best_reviewer.max_concurrent_reviews}"
            ),
        )

    def _score_reviewer(
        self,
        reviewer: ReviewerSummary,
        task_type: str,
        required_skills: list[str],
    ) -> float:
        """Compute an assignment score for a reviewer candidate.

        Scoring components (all normalised to 0-1 range):
        - Skill tag overlap: Jaccard similarity between reviewer skills and required skills
        - Task type preference: 1.0 if task_type is in preferred list, 0.5 otherwise
        - Capacity headroom: Fraction of capacity still available
        - Accuracy score: Historical accuracy (0.5 if unknown)

        Weights: skill=0.35, preference=0.25, capacity=0.25, accuracy=0.15

        Args:
            reviewer: Reviewer candidate to score.
            task_type: Task type to assign.
            required_skills: Skills the task requires.

        Returns:
            Composite score in range [0.0, 1.0].
        """
        # Skill overlap (Jaccard similarity)
        if required_skills:
            reviewer_skill_set = set(reviewer.skill_tags)
            required_set = set(required_skills)
            intersection = len(reviewer_skill_set & required_set)
            union = len(reviewer_skill_set | required_set)
            skill_score = intersection / union if union > 0 else 0.0
        else:
            skill_score = 1.0  # No skills required — all reviewers are equally qualified

        # Task type preference
        preference_score = 1.0 if task_type in reviewer.preferred_task_types else 0.5

        # Capacity headroom (higher is better)
        capacity_headroom = (
            reviewer.max_concurrent_reviews - reviewer.current_review_count
        ) / reviewer.max_concurrent_reviews
        capacity_score = max(0.0, capacity_headroom)

        # Historical accuracy
        accuracy_score = reviewer.accuracy_score if reviewer.accuracy_score is not None else 0.5

        composite = (
            skill_score * 0.35
            + preference_score * 0.25
            + capacity_score * 0.25
            + accuracy_score * 0.15
        )
        return round(composite, 4)

    async def get_capacity_report(
        self,
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate a workforce capacity report for a tenant.

        Returns aggregate statistics on reviewer availability, workload
        distribution, and overloaded reviewers.

        Args:
            tenant_id: The tenant to report on.

        Returns:
            Dict with total_reviewers, available_count, total_capacity,
            current_load, utilisation_pct, and overloaded_reviewers list.
        """
        try:
            reviewers = await self._reviewer_repo.list_available(tenant_id=tenant_id)
        except Exception as exc:
            logger.error(
                "Failed to generate capacity report",
                tenant_id=str(tenant_id),
                error=str(exc),
            )
            return {
                "error": "Failed to retrieve reviewer data",
                "total_reviewers": 0,
            }

        available = [r for r in reviewers if r.is_available]
        total_capacity = sum(r.max_concurrent_reviews for r in available)
        current_load = sum(r.current_review_count for r in available)
        utilisation_pct = (current_load / total_capacity * 100) if total_capacity > 0 else 0.0

        overloaded = [
            {
                "reviewer_id": str(r.user_id),
                "display_name": r.display_name,
                "load": r.current_review_count,
                "max": r.max_concurrent_reviews,
                "utilisation_pct": round(r.current_review_count / r.max_concurrent_reviews * 100, 1),
            }
            for r in available
            if r.current_review_count >= r.max_concurrent_reviews
        ]

        return {
            "total_reviewers": len(reviewers),
            "available_reviewers": len(available),
            "unavailable_reviewers": len(reviewers) - len(available),
            "total_capacity": total_capacity,
            "current_load": current_load,
            "available_slots": total_capacity - current_load,
            "utilisation_pct": round(utilisation_pct, 1),
            "overloaded_reviewers": overloaded,
            "overloaded_count": len(overloaded),
        }


__all__ = [
    "ReviewerSummary",
    "AssignmentResult",
    "WorkforceManager",
]

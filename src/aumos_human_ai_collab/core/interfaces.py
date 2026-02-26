"""Abstract interfaces (Protocol classes) for the AumOS Human-AI Collaboration service.

All adapters implement these protocols so services depend only on abstractions,
enabling straightforward testing via mock implementations.
"""

import uuid
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from aumos_human_ai_collab.core.models import (
    AttributionRecord,
    ComplianceGate,
    FeedbackCorrection,
    HITLReview,
    RoutingDecision,
)


@runtime_checkable
class IRoutingDecisionRepository(Protocol):
    """Persistence interface for RoutingDecision entities."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        task_id: uuid.UUID,
        task_type: str,
        confidence_score: float,
        routing_outcome: str,
        ai_threshold_applied: float,
        model_id: str | None,
        compliance_gate_triggered: bool,
        compliance_gate_id: uuid.UUID | None,
        routing_metadata: dict[str, Any],
    ) -> RoutingDecision:
        """Create and persist a routing decision.

        Args:
            tenant_id: Owning tenant UUID.
            task_id: External task UUID.
            task_type: Semantic task category.
            confidence_score: AI confidence score (0–1).
            routing_outcome: ai | human | hybrid.
            ai_threshold_applied: Threshold active at routing time.
            model_id: AI model identifier.
            compliance_gate_triggered: Whether a compliance gate forced routing.
            compliance_gate_id: UUID of the triggering gate (if any).
            routing_metadata: Arbitrary routing context.

        Returns:
            Newly created RoutingDecision.
        """
        ...

    async def get_by_id(
        self, decision_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> RoutingDecision | None:
        """Retrieve a routing decision by UUID within a tenant.

        Args:
            decision_id: RoutingDecision UUID.
            tenant_id: Requesting tenant.

        Returns:
            RoutingDecision or None if not found.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        task_type: str | None,
        routing_outcome: str | None,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> tuple[list[RoutingDecision], int]:
        """List routing decisions for a tenant with optional filters.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            task_type: Optional task_type filter.
            routing_outcome: Optional outcome filter.
            date_from: Optional start date for range filter.
            date_to: Optional end date for range filter.

        Returns:
            Tuple of (decisions, total_count).
        """
        ...

    async def mark_resolved(
        self,
        decision_id: uuid.UUID,
        resolved_by: str,
        resolved_at: datetime,
    ) -> RoutingDecision:
        """Mark a routing decision as resolved.

        Args:
            decision_id: RoutingDecision UUID.
            resolved_by: ai | human | hybrid.
            resolved_at: Resolution timestamp.

        Returns:
            Updated RoutingDecision.
        """
        ...


@runtime_checkable
class IComplianceGateRepository(Protocol):
    """Persistence interface for ComplianceGate entities."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        gate_name: str,
        regulation: str,
        task_types: list[str],
        description: str | None,
        reviewer_role: str | None,
        metadata: dict[str, Any],
    ) -> ComplianceGate:
        """Create a compliance gate.

        Args:
            tenant_id: Owning tenant UUID.
            gate_name: Unique name for this gate.
            regulation: Regulatory framework (hipaa, gdpr, etc.).
            task_types: Task types this gate covers.
            description: Optional description.
            reviewer_role: Required reviewer role.
            metadata: Additional gate configuration.

        Returns:
            Newly created ComplianceGate.
        """
        ...

    async def get_by_id(
        self, gate_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> ComplianceGate | None:
        """Retrieve a compliance gate by UUID.

        Args:
            gate_id: ComplianceGate UUID.
            tenant_id: Requesting tenant.

        Returns:
            ComplianceGate or None if not found.
        """
        ...

    async def find_matching_gates(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> list[ComplianceGate]:
        """Find all active compliance gates that apply to a given task type.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to match against gate task_types lists.

        Returns:
            List of active ComplianceGate instances that match.
        """
        ...

    async def list_by_tenant(
        self, tenant_id: uuid.UUID, active_only: bool
    ) -> list[ComplianceGate]:
        """List compliance gates for a tenant.

        Args:
            tenant_id: Requesting tenant.
            active_only: If True, exclude soft-deleted gates.

        Returns:
            List of ComplianceGate instances.
        """
        ...

    async def update(
        self, gate_id: uuid.UUID, tenant_id: uuid.UUID, updates: dict[str, Any]
    ) -> ComplianceGate:
        """Apply partial updates to a compliance gate.

        Args:
            gate_id: Gate UUID.
            tenant_id: Owning tenant.
            updates: Dict of field_name → new_value.

        Returns:
            Updated ComplianceGate.
        """
        ...


@runtime_checkable
class IHITLReviewRepository(Protocol):
    """Persistence interface for HITLReview entities."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        routing_decision_id: uuid.UUID,
        task_type: str,
        ai_output: dict[str, Any],
        reviewer_id: uuid.UUID | None,
        due_at: datetime | None,
        priority: int,
    ) -> HITLReview:
        """Create a HITL review task.

        Args:
            tenant_id: Owning tenant UUID.
            routing_decision_id: Parent RoutingDecision UUID.
            task_type: Task category.
            ai_output: AI-generated output for review.
            reviewer_id: Assigned reviewer UUID.
            due_at: Review deadline.
            priority: Review priority (1–4).

        Returns:
            Newly created HITLReview in pending status.
        """
        ...

    async def get_by_id(
        self, review_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> HITLReview | None:
        """Retrieve a HITL review by UUID.

        Args:
            review_id: HITLReview UUID.
            tenant_id: Requesting tenant.

        Returns:
            HITLReview or None if not found.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        status: str | None,
        reviewer_id: uuid.UUID | None,
        task_type: str | None,
    ) -> tuple[list[HITLReview], int]:
        """List HITL reviews for a tenant with optional filters.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            status: Optional status filter.
            reviewer_id: Optional reviewer UUID filter.
            task_type: Optional task_type filter.

        Returns:
            Tuple of (reviews, total_count).
        """
        ...

    async def submit_decision(
        self,
        review_id: uuid.UUID,
        decision: str,
        reviewer_output: dict[str, Any],
        reviewer_notes: str | None,
        review_completed_at: datetime,
    ) -> HITLReview:
        """Record the reviewer's decision and output.

        Args:
            review_id: HITLReview UUID.
            decision: approved | rejected | modified.
            reviewer_output: Final output from reviewer.
            reviewer_notes: Optional reviewer comments.
            review_completed_at: Decision timestamp.

        Returns:
            Updated HITLReview with decision recorded.
        """
        ...

    async def update_status(
        self,
        review_id: uuid.UUID,
        status: str,
        reviewer_id: uuid.UUID | None,
        review_started_at: datetime | None,
    ) -> HITLReview:
        """Update the status of a HITL review.

        Args:
            review_id: HITLReview UUID.
            status: New status value.
            reviewer_id: Optional reviewer UUID.
            review_started_at: Optional start timestamp.

        Returns:
            Updated HITLReview.
        """
        ...


@runtime_checkable
class IAttributionRepository(Protocol):
    """Persistence interface for AttributionRecord entities."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        task_id: uuid.UUID,
        task_type: str,
        routing_decision_id: uuid.UUID | None,
        hitl_review_id: uuid.UUID | None,
        ai_contribution_score: float,
        human_contribution_score: float,
        time_saved_seconds: int | None,
        attribution_method: str,
        resolution: str,
        attribution_metadata: dict[str, Any],
    ) -> AttributionRecord:
        """Create an attribution record.

        Args:
            tenant_id: Owning tenant UUID.
            task_id: External task UUID.
            task_type: Task category.
            routing_decision_id: Linked routing decision.
            hitl_review_id: Linked HITL review.
            ai_contribution_score: AI fraction (0–1).
            human_contribution_score: Human fraction (0–1).
            time_saved_seconds: Estimated time saved.
            attribution_method: Computation method.
            resolution: Final resolution: ai | human | hybrid.
            attribution_metadata: Additional attribution data.

        Returns:
            Newly created AttributionRecord.
        """
        ...

    async def get_report(
        self,
        tenant_id: uuid.UUID,
        days: int,
        task_type: str | None,
    ) -> dict[str, Any]:
        """Generate an attribution analytics report for a tenant.

        Args:
            tenant_id: Requesting tenant.
            days: Lookback window in days.
            task_type: Optional filter by task type.

        Returns:
            Dict with aggregate AI/human contribution metrics and per-task-type breakdown.
        """
        ...


@runtime_checkable
class IFeedbackRepository(Protocol):
    """Persistence interface for FeedbackCorrection entities."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        routing_decision_id: uuid.UUID,
        hitl_review_id: uuid.UUID | None,
        task_type: str,
        original_confidence: float,
        original_routing: str,
        corrected_routing: str,
        correction_reason: str | None,
        correction_type: str,
        submitted_by: uuid.UUID | None,
    ) -> FeedbackCorrection:
        """Record a human feedback correction.

        Args:
            tenant_id: Owning tenant UUID.
            routing_decision_id: The routing decision being corrected.
            hitl_review_id: Optional linked HITL review.
            task_type: Task type for recalibration grouping.
            original_confidence: Confidence score at routing time.
            original_routing: Original routing outcome.
            corrected_routing: Correct routing outcome.
            correction_reason: Human explanation.
            correction_type: Correction category.
            submitted_by: Submitting user UUID.

        Returns:
            Newly created FeedbackCorrection.
        """
        ...

    async def list_uncalibrated(
        self, tenant_id: uuid.UUID, task_type: str, limit: int
    ) -> list[FeedbackCorrection]:
        """List feedback corrections not yet applied to calibration.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to query.
            limit: Maximum records to return.

        Returns:
            List of FeedbackCorrection instances where calibration_applied=False.
        """
        ...

    async def mark_calibrated(
        self,
        correction_ids: list[uuid.UUID],
        calibration_applied_at: datetime,
    ) -> int:
        """Mark feedback corrections as applied to calibration.

        Args:
            correction_ids: List of FeedbackCorrection UUIDs.
            calibration_applied_at: Timestamp of calibration application.

        Returns:
            Number of records updated.
        """
        ...

    async def get_calibration_summary(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> dict[str, Any]:
        """Summarise feedback corrections for recalibration input.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to summarise.

        Returns:
            Dict with sample count, error rate, mean correction delta, etc.
        """
        ...


@runtime_checkable
class IConfidenceEngineAdapter(Protocol):
    """Interface for the AI confidence scoring adapter."""

    async def score_task(
        self,
        task_type: str,
        task_payload: dict[str, Any],
        model_id: str | None,
        tenant_id: uuid.UUID,
    ) -> tuple[float, str | None]:
        """Produce a confidence score for a task.

        Args:
            task_type: Semantic task category.
            task_payload: Task input data for scoring.
            model_id: Optional target model identifier.
            tenant_id: Requesting tenant.

        Returns:
            Tuple of (confidence_score, model_id_used).
        """
        ...

    async def recalibrate(
        self,
        task_type: str,
        corrections: list[dict[str, Any]],
        decay_factor: float,
    ) -> dict[str, Any]:
        """Recalibrate confidence thresholds using human corrections.

        Args:
            task_type: Task type to recalibrate.
            corrections: List of correction dicts from FeedbackCorrection records.
            decay_factor: Exponential decay applied to older samples.

        Returns:
            Recalibration result dict with new threshold recommendations.
        """
        ...

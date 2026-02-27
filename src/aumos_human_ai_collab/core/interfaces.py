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

# ---------------------------------------------------------------------------
# Re-export adapter types so services can type-hint against them
# ---------------------------------------------------------------------------
from aumos_human_ai_collab.adapters.feedback_aggregator import (  # noqa: F401
    FeedbackAggregator,
    FeedbackRecord,
)
from aumos_human_ai_collab.adapters.annotation_engine import (  # noqa: F401
    AnnotationEngine,
    AnnotationTask,
    Annotation,
)
from aumos_human_ai_collab.adapters.consensus_scorer import ConsensusScorer  # noqa: F401
from aumos_human_ai_collab.adapters.active_learner import (  # noqa: F401
    ActiveLearner,
    LearningCurvePoint,
)
from aumos_human_ai_collab.adapters.explainability_bridge import (  # noqa: F401
    ExplainabilityBridge,
)
from aumos_human_ai_collab.adapters.performance_tracker import (  # noqa: F401
    PerformanceTracker,
    MetricSnapshot,
)
from aumos_human_ai_collab.adapters.calibration_engine import (  # noqa: F401
    CalibrationEngine,
    CalibrationRecord,
)
from aumos_human_ai_collab.adapters.review_queue_manager import (  # noqa: F401
    ReviewQueueManager,
    QueueItem,
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


# ---------------------------------------------------------------------------
# New domain-specific adapter protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class IFeedbackAggregator(Protocol):
    """Interface for user feedback collection and trend analysis."""

    async def store_feedback(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        feature: str,
        rating: int,
        category: str,
        text: str | None,
        submitted_by: uuid.UUID | None,
    ) -> Any:
        """Store a single feedback record.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Model being rated.
            feature: Feature being rated.
            rating: Integer rating 1–5.
            category: Feedback category.
            text: Optional free-text comment.
            submitted_by: Optional submitter UUID.

        Returns:
            Stored FeedbackRecord.
        """
        ...

    async def aggregate_by_model(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        window_days: int,
    ) -> dict[str, Any]:
        """Aggregate feedback metrics for a model over a time window.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            window_days: Lookback window in days.

        Returns:
            Aggregate statistics dict.
        """
        ...

    async def get_top_issues(
        self,
        tenant_id: uuid.UUID,
        window_days: int,
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Identify top issues by rating and volume.

        Args:
            tenant_id: Requesting tenant.
            window_days: Lookback window.
            top_n: Number of issues to return.

        Returns:
            List of issue dicts sorted by severity.
        """
        ...

    async def export_records(
        self,
        tenant_id: uuid.UUID,
        model_id: str | None,
        feature: str | None,
        category: str | None,
        window_days: int | None,
    ) -> list[dict[str, Any]]:
        """Export feedback records as a list of dicts.

        Args:
            tenant_id: Requesting tenant.
            model_id: Optional model filter.
            feature: Optional feature filter.
            category: Optional category filter.
            window_days: Optional lookback window.

        Returns:
            List of serialised feedback record dicts.
        """
        ...


@runtime_checkable
class IAnnotationEngine(Protocol):
    """Interface for crowdsourced annotation task management."""

    async def create_task(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        name: str,
        items: list[dict[str, Any]],
        labels: list[str],
        gold_items: list[dict[str, Any]] | None,
        annotations_per_item: int | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create a new annotation task.

        Args:
            tenant_id: Owning tenant UUID.
            task_type: Type of labeling task.
            name: Human-readable task name.
            items: Items to annotate.
            labels: Valid label strings.
            gold_items: Optional gold standard items.
            annotations_per_item: Override default.
            metadata: Additional configuration.

        Returns:
            Task summary dict.
        """
        ...

    async def assign_annotators(
        self,
        task_id: uuid.UUID,
        annotator_ids: list[uuid.UUID],
    ) -> dict[str, Any]:
        """Assign annotators to a task.

        Args:
            task_id: Task UUID.
            annotator_ids: Annotator UUIDs.

        Returns:
            Assignment summary dict.
        """
        ...

    async def store_annotation(
        self,
        task_id: uuid.UUID,
        item_id: str,
        annotator_id: uuid.UUID,
        label: str,
        confidence: float | None,
        notes: str | None,
    ) -> dict[str, Any]:
        """Store an annotation from an annotator.

        Args:
            task_id: Task UUID.
            item_id: Item being annotated.
            annotator_id: Annotator UUID.
            label: Chosen label.
            confidence: Optional annotator confidence.
            notes: Optional notes.

        Returns:
            Annotation result dict.
        """
        ...

    async def get_task_progress(self, task_id: uuid.UUID) -> dict[str, Any]:
        """Return progress statistics for a task.

        Args:
            task_id: Task UUID.

        Returns:
            Progress dict.
        """
        ...

    async def export_annotations(
        self,
        task_id: uuid.UUID,
        format: str,
        exclude_gold: bool,
    ) -> str:
        """Export annotations in JSONL or CSV format.

        Args:
            task_id: Task UUID.
            format: "jsonl" or "csv".
            exclude_gold: Exclude gold items if True.

        Returns:
            Exported content string.
        """
        ...


@runtime_checkable
class IConsensusScorer(Protocol):
    """Interface for multi-annotator agreement computation."""

    def compute_fleiss_kappa(
        self,
        annotations: list[dict[str, str]],
        labels: list[str],
    ) -> dict[str, Any]:
        """Compute Fleiss' kappa.

        Args:
            annotations: Annotation dicts with item_id, annotator_id, label.
            labels: All possible labels.

        Returns:
            Kappa result dict.
        """
        ...

    def compute_cohen_kappa(
        self,
        annotator_a_labels: list[str],
        annotator_b_labels: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """Compute Cohen's kappa for two annotators.

        Args:
            annotator_a_labels: Labels from annotator A.
            annotator_b_labels: Labels from annotator B.
            labels: All possible labels.

        Returns:
            Kappa result dict.
        """
        ...

    def majority_vote(
        self,
        item_id: str,
        annotations: list[dict[str, str]],
        min_agreement_pct: float,
    ) -> dict[str, Any]:
        """Resolve label via majority voting.

        Args:
            item_id: Item being resolved.
            annotations: Annotation dicts.
            min_agreement_pct: Minimum agreement fraction.

        Returns:
            Resolution dict.
        """
        ...

    def generate_consensus_report(
        self,
        task_id: str,
        annotations: list[dict[str, str]],
        labels: list[str],
        annotator_ids: list[uuid.UUID],
    ) -> dict[str, Any]:
        """Generate a consensus report for a task.

        Args:
            task_id: Task identifier.
            annotations: All task annotations.
            labels: All possible labels.
            annotator_ids: All annotators.

        Returns:
            Consensus report dict.
        """
        ...


@runtime_checkable
class IActiveLearner(Protocol):
    """Interface for uncertainty-based sample selection."""

    def add_to_pool(
        self,
        session_id: uuid.UUID,
        samples: list[dict[str, Any]],
    ) -> int:
        """Add unlabelled samples to the pool.

        Args:
            session_id: Session UUID.
            samples: Sample dicts.

        Returns:
            Total pool size.
        """
        ...

    async def select_samples(
        self,
        session_id: uuid.UUID,
        batch_size: int | None,
    ) -> dict[str, Any]:
        """Select informative samples for labelling.

        Args:
            session_id: Session UUID.
            batch_size: Override batch size.

        Returns:
            Selection result dict.
        """
        ...

    async def record_round_result(
        self,
        session_id: uuid.UUID,
        labels_used: int,
        estimated_accuracy: float,
    ) -> Any:
        """Record a round outcome for learning curve tracking.

        Args:
            session_id: Session UUID.
            labels_used: Labels used this round.
            estimated_accuracy: Model accuracy after this round.

        Returns:
            LearningCurvePoint.
        """
        ...

    def get_budget_status(self) -> dict[str, Any]:
        """Return current budget usage.

        Returns:
            Budget status dict.
        """
        ...


@runtime_checkable
class IExplainabilityBridge(Protocol):
    """Interface for SHAP/LIME explanation retrieval for human reviewers."""

    async def get_shap_explanation(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        tenant_id: uuid.UUID,
        top_features: int,
    ) -> dict[str, Any]:
        """Retrieve SHAP feature importance for a prediction.

        Args:
            model_id: Model identifier.
            input_data: Prediction input.
            prediction: Model output.
            tenant_id: Requesting tenant.
            top_features: Number of top features.

        Returns:
            SHAP explanation dict.
        """
        ...

    async def get_lime_explanation(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        tenant_id: uuid.UUID,
        num_samples: int,
        top_features: int,
    ) -> dict[str, Any]:
        """Retrieve a LIME explanation for a prediction.

        Args:
            model_id: Model identifier.
            input_data: Prediction input.
            prediction: Model output.
            tenant_id: Requesting tenant.
            num_samples: LIME perturbation count.
            top_features: Number of top features.

        Returns:
            LIME explanation dict.
        """
        ...

    async def get_explanation_for_review(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        tenant_id: uuid.UUID,
        method: str,
    ) -> dict[str, Any]:
        """Get a complete explanation package for a reviewer.

        Args:
            model_id: Model identifier.
            input_data: Prediction input.
            prediction: Model output.
            tenant_id: Requesting tenant.
            method: "shap" or "lime".

        Returns:
            Reviewer explanation package dict.
        """
        ...


@runtime_checkable
class IPerformanceTracker(Protocol):
    """Interface for continuous model performance tracking."""

    async def record_metric(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        model_version: str,
        metric_name: str,
        metric_value: float,
        segment: str | None,
        data_type: str | None,
        feedback_round_id: str | None,
    ) -> Any:
        """Record a model metric snapshot.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Model identifier.
            model_version: Model version string.
            metric_name: Metric name.
            metric_value: Measured value.
            segment: Optional user segment.
            data_type: Optional data type label.
            feedback_round_id: Optional feedback round link.

        Returns:
            MetricSnapshot.
        """
        ...

    async def get_accuracy_trend(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        metric_name: str,
        window_days: int,
        segment: str | None,
    ) -> dict[str, Any]:
        """Return metric trend over a time window.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            metric_name: Metric to trend.
            window_days: Lookback window.
            segment: Optional segment filter.

        Returns:
            Trend dict with data_points, direction, and delta.
        """
        ...

    async def compare_versions(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        version_a: str,
        version_b: str,
        metric_name: str,
        window_days: int,
    ) -> dict[str, Any]:
        """Compare performance between two model versions.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            version_a: First version.
            version_b: Second version.
            metric_name: Metric to compare.
            window_days: Lookback window.

        Returns:
            A/B comparison dict.
        """
        ...

    async def generate_performance_report(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        window_days: int,
        include_versions: list[str] | None,
    ) -> dict[str, Any]:
        """Generate a comprehensive performance report.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            window_days: Lookback window.
            include_versions: Optional specific versions.

        Returns:
            Performance report dict.
        """
        ...


@runtime_checkable
class ICalibrationEngine(Protocol):
    """Interface for confidence score recalibration."""

    def compute_ece(
        self,
        confidences: list[float],
        labels: list[int],
        n_bins: int,
    ) -> dict[str, Any]:
        """Compute Expected Calibration Error.

        Args:
            confidences: Predicted confidence scores.
            labels: Binary ground-truth labels.
            n_bins: Number of histogram bins.

        Returns:
            ECE result dict with bin_data.
        """
        ...

    def apply_calibration(
        self,
        raw_confidence: float,
        params: dict[str, Any],
    ) -> float:
        """Apply calibration parameters to a raw confidence score.

        Args:
            raw_confidence: Raw model confidence.
            params: Calibration params dict.

        Returns:
            Calibrated confidence score.
        """
        ...

    async def recalibrate_from_feedback(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        confidences: list[float],
        labels: list[int],
        method: str,
    ) -> Any:
        """Recalibrate from feedback-derived ground truth.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type being calibrated.
            confidences: Historical confidence scores.
            labels: Ground truth labels.
            method: Calibration method.

        Returns:
            CalibrationRecord.
        """
        ...

    def get_calibration_history(
        self,
        tenant_id: uuid.UUID,
        task_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return calibration history.

        Args:
            tenant_id: Requesting tenant.
            task_type: Optional task type filter.
            limit: Maximum records.

        Returns:
            List of CalibrationRecord dicts.
        """
        ...


@runtime_checkable
class IReviewQueueManager(Protocol):
    """Interface for the Redis-backed HITL review priority queue."""

    async def enqueue(
        self,
        tenant_id: uuid.UUID,
        review_id: uuid.UUID,
        task_type: str,
        confidence: float,
        urgency: int,
        metadata: dict[str, Any] | None,
    ) -> Any:
        """Enqueue a review item.

        Args:
            tenant_id: Owning tenant UUID.
            review_id: HITLReview UUID.
            task_type: Task category.
            confidence: Model confidence score.
            urgency: Review urgency 1–4.
            metadata: Optional context.

        Returns:
            QueueItem.
        """
        ...

    async def assign_to_reviewer(
        self,
        tenant_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> Any | None:
        """Dequeue highest-priority item and assign to reviewer.

        Args:
            tenant_id: Requesting tenant.
            reviewer_id: Reviewer UUID.

        Returns:
            Assigned QueueItem or None if queue is empty.
        """
        ...

    async def mark_completed(
        self,
        tenant_id: uuid.UUID,
        item_id: uuid.UUID,
    ) -> Any:
        """Mark a queue item as completed.

        Args:
            tenant_id: Owning tenant UUID.
            item_id: Queue item UUID.

        Returns:
            Updated QueueItem.
        """
        ...

    async def get_queue_depth(self, tenant_id: uuid.UUID) -> int:
        """Return number of pending items.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Pending item count.
        """
        ...

    async def get_queue_analytics(self, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Return queue analytics.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Analytics dict.
        """
        ...

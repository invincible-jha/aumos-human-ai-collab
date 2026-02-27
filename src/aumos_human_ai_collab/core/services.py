"""Business logic services for the AumOS Human-AI Collaboration service.

All services depend on repository and adapter interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code (FastAPI, SQLAlchemy) belongs here.

Key invariants enforced by services:
- Compliance gates always override confidence-based routing.
- Review decisions are immutable once submitted.
- Attribution scores must sum to 1.0 (ai + human = 1.0).
- Feedback corrections trigger recalibration when sample threshold is reached.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_human_ai_collab.core.interfaces import (
    IActiveLearner,
    IAnnotationEngine,
    IAttributionRepository,
    ICalibrationEngine,
    IComplianceGateRepository,
    IConfidenceEngineAdapter,
    IConsensusScorer,
    IExplainabilityBridge,
    IFeedbackAggregator,
    IFeedbackRepository,
    IHITLReviewRepository,
    IPerformanceTracker,
    IReviewQueueManager,
    IRoutingDecisionRepository,
)
from aumos_human_ai_collab.core.models import (
    AttributionRecord,
    ComplianceGate,
    FeedbackCorrection,
    HITLReview,
    RoutingDecision,
)

logger = get_logger(__name__)

# Valid routing outcomes
VALID_ROUTING_OUTCOMES: frozenset[str] = frozenset({"ai", "human", "hybrid"})

# Valid compliance regulations
VALID_REGULATIONS: frozenset[str] = frozenset(
    {"hipaa", "gdpr", "sox", "pci_dss", "finra", "fda"}
)

# Valid review decision statuses
VALID_REVIEW_DECISIONS: frozenset[str] = frozenset({"approved", "rejected", "modified"})

# Terminal review statuses — immutable once reached
TERMINAL_REVIEW_STATUSES: frozenset[str] = frozenset(
    {"approved", "rejected", "modified"}
)

# Valid correction types
VALID_CORRECTION_TYPES: frozenset[str] = frozenset(
    {"routing_error", "confidence_overestimate", "confidence_underestimate", "output_error"}
)


class RoutingService:
    """Evaluate and record confidence-based task routing decisions.

    Applies confidence thresholds and compliance gates to determine whether
    a task should be handled by AI, a human, or a hybrid approach.
    """

    def __init__(
        self,
        routing_repo: IRoutingDecisionRepository,
        compliance_repo: IComplianceGateRepository,
        confidence_engine: IConfidenceEngineAdapter,
        event_publisher: EventPublisher,
        ai_confidence_threshold: float = 0.85,
        hybrid_confidence_lower: float = 0.65,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            routing_repo: RoutingDecision persistence.
            compliance_repo: ComplianceGate persistence.
            confidence_engine: AI confidence scoring adapter.
            event_publisher: Kafka event publisher.
            ai_confidence_threshold: Minimum confidence for AI autonomous handling.
            hybrid_confidence_lower: Confidence floor for hybrid routing.
        """
        self._routing = routing_repo
        self._compliance = compliance_repo
        self._confidence = confidence_engine
        self._publisher = event_publisher
        self._ai_threshold = ai_confidence_threshold
        self._hybrid_lower = hybrid_confidence_lower

    async def evaluate_routing(
        self,
        tenant_id: uuid.UUID,
        task_id: uuid.UUID,
        task_type: str,
        task_payload: dict[str, Any],
        model_id: str | None = None,
        override_routing: str | None = None,
    ) -> RoutingDecision:
        """Evaluate routing for a task and record the decision.

        Checks compliance gates first (mandatory human routing overrides
        all confidence scores), then applies thresholds for AI/human/hybrid.

        Args:
            tenant_id: Owning tenant UUID.
            task_id: External task UUID.
            task_type: Semantic task category.
            task_payload: Task input data (used for confidence scoring).
            model_id: Optional preferred model identifier.
            override_routing: Optional manual override: ai | human | hybrid.

        Returns:
            Newly created RoutingDecision.

        Raises:
            ConflictError: If override_routing value is invalid.
        """
        if override_routing is not None and override_routing not in VALID_ROUTING_OUTCOMES:
            raise ConflictError(
                message=f"Invalid override_routing '{override_routing}'. Valid: {VALID_ROUTING_OUTCOMES}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        # Step 1: Score confidence
        confidence_score, used_model_id = await self._confidence.score_task(
            task_type=task_type,
            task_payload=task_payload,
            model_id=model_id,
            tenant_id=tenant_id,
        )

        # Step 2: Check compliance gates (always override)
        matching_gates = await self._compliance.find_matching_gates(tenant_id, task_type)
        compliance_gate_triggered = len(matching_gates) > 0
        compliance_gate_id: uuid.UUID | None = matching_gates[0].id if matching_gates else None

        # Step 3: Determine routing outcome
        if override_routing is not None:
            routing_outcome = override_routing
        elif compliance_gate_triggered:
            routing_outcome = "human"
        elif confidence_score >= self._ai_threshold:
            routing_outcome = "ai"
        elif confidence_score >= self._hybrid_lower:
            routing_outcome = "hybrid"
        else:
            routing_outcome = "human"

        decision = await self._routing.create(
            tenant_id=tenant_id,
            task_id=task_id,
            task_type=task_type,
            confidence_score=confidence_score,
            routing_outcome=routing_outcome,
            ai_threshold_applied=self._ai_threshold,
            model_id=used_model_id,
            compliance_gate_triggered=compliance_gate_triggered,
            compliance_gate_id=compliance_gate_id,
            routing_metadata={
                "task_payload_keys": list(task_payload.keys()),
                "override_applied": override_routing is not None,
                "gates_matched": [str(g.id) for g in matching_gates],
            },
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_ROUTING,
            {
                "event_type": "hac.routing.evaluated",
                "tenant_id": str(tenant_id),
                "decision_id": str(decision.id),
                "task_id": str(task_id),
                "task_type": task_type,
                "confidence_score": confidence_score,
                "routing_outcome": routing_outcome,
                "compliance_gate_triggered": compliance_gate_triggered,
            },
        )

        logger.info(
            "Routing decision recorded",
            tenant_id=str(tenant_id),
            task_id=str(task_id),
            task_type=task_type,
            confidence_score=confidence_score,
            routing_outcome=routing_outcome,
            compliance_gate_triggered=compliance_gate_triggered,
        )

        return decision

    async def get_decision(
        self, decision_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> RoutingDecision:
        """Retrieve a routing decision by ID.

        Args:
            decision_id: RoutingDecision UUID.
            tenant_id: Requesting tenant.

        Returns:
            RoutingDecision.

        Raises:
            NotFoundError: If decision not found.
        """
        decision = await self._routing.get_by_id(decision_id, tenant_id)
        if decision is None:
            raise NotFoundError(
                message=f"Routing decision {decision_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return decision

    async def list_decisions(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        task_type: str | None = None,
        routing_outcome: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
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
        return await self._routing.list_by_tenant(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            task_type=task_type,
            routing_outcome=routing_outcome,
            date_from=date_from,
            date_to=date_to,
        )


class ComplianceGateService:
    """CRUD and evaluation operations for compliance gates.

    Compliance gates define which task types require mandatory human review
    due to regulatory requirements (HIPAA, GDPR, SOX, etc.).
    """

    def __init__(
        self,
        compliance_repo: IComplianceGateRepository,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            compliance_repo: ComplianceGate persistence.
            event_publisher: Kafka event publisher.
        """
        self._gates = compliance_repo
        self._publisher = event_publisher

    async def create_gate(
        self,
        tenant_id: uuid.UUID,
        gate_name: str,
        regulation: str,
        task_types: list[str],
        description: str | None = None,
        reviewer_role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComplianceGate:
        """Create a compliance gate.

        Args:
            tenant_id: Owning tenant UUID.
            gate_name: Unique gate name within tenant.
            regulation: Regulatory framework.
            task_types: Task types the gate covers.
            description: Optional human-readable description.
            reviewer_role: Required reviewer role.
            metadata: Additional gate configuration.

        Returns:
            Newly created ComplianceGate.

        Raises:
            ConflictError: If regulation is invalid or no task_types provided.
        """
        if regulation not in VALID_REGULATIONS:
            raise ConflictError(
                message=f"Invalid regulation '{regulation}'. Valid: {VALID_REGULATIONS}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        if not task_types:
            raise ConflictError(
                message="At least one task_type must be specified for a compliance gate.",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        gate = await self._gates.create(
            tenant_id=tenant_id,
            gate_name=gate_name,
            regulation=regulation,
            task_types=task_types,
            description=description,
            reviewer_role=reviewer_role,
            metadata=metadata or {},
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_COMPLIANCE,
            {
                "event_type": "hac.compliance_gate.created",
                "tenant_id": str(tenant_id),
                "gate_id": str(gate.id),
                "gate_name": gate_name,
                "regulation": regulation,
                "task_types": task_types,
            },
        )

        logger.info(
            "Compliance gate created",
            tenant_id=str(tenant_id),
            gate_id=str(gate.id),
            regulation=regulation,
            task_types=task_types,
        )

        return gate

    async def get_gate(
        self, gate_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> ComplianceGate:
        """Retrieve a compliance gate by ID.

        Args:
            gate_id: ComplianceGate UUID.
            tenant_id: Requesting tenant.

        Returns:
            ComplianceGate.

        Raises:
            NotFoundError: If gate not found.
        """
        gate = await self._gates.get_by_id(gate_id, tenant_id)
        if gate is None:
            raise NotFoundError(
                message=f"Compliance gate {gate_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return gate

    async def evaluate_gate(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
    ) -> dict[str, Any]:
        """Evaluate whether a task type triggers any compliance gates.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to evaluate.

        Returns:
            Dict with triggered flag, matching gate IDs, and required reviewer roles.
        """
        matching_gates = await self._gates.find_matching_gates(tenant_id, task_type)
        triggered = len(matching_gates) > 0

        logger.info(
            "Compliance gate evaluation",
            tenant_id=str(tenant_id),
            task_type=task_type,
            triggered=triggered,
            gate_count=len(matching_gates),
        )

        return {
            "triggered": triggered,
            "gate_ids": [str(g.id) for g in matching_gates],
            "gate_names": [g.gate_name for g in matching_gates],
            "regulations": list({g.regulation for g in matching_gates}),
            "required_reviewer_roles": list(
                {g.reviewer_role for g in matching_gates if g.reviewer_role}
            ),
        }

    async def list_gates(
        self, tenant_id: uuid.UUID, active_only: bool = True
    ) -> list[ComplianceGate]:
        """List compliance gates for a tenant.

        Args:
            tenant_id: Requesting tenant.
            active_only: If True, exclude soft-deleted gates.

        Returns:
            List of ComplianceGate instances.
        """
        return await self._gates.list_by_tenant(tenant_id, active_only)


class HITLReviewService:
    """Manage human-in-the-loop review tasks.

    Creates, assigns, and records decisions for human review tasks
    triggered by confidence-based or compliance-gate routing.
    """

    def __init__(
        self,
        review_repo: IHITLReviewRepository,
        attribution_repo: IAttributionRepository,
        event_publisher: EventPublisher,
        review_timeout_hours: int = 24,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            review_repo: HITLReview persistence.
            attribution_repo: AttributionRecord persistence for post-review attribution.
            event_publisher: Kafka event publisher.
            review_timeout_hours: Deadline hours from review creation.
        """
        self._reviews = review_repo
        self._attribution = attribution_repo
        self._publisher = event_publisher
        self._timeout_hours = review_timeout_hours

    async def create_review(
        self,
        tenant_id: uuid.UUID,
        routing_decision_id: uuid.UUID,
        task_type: str,
        ai_output: dict[str, Any],
        reviewer_id: uuid.UUID | None = None,
        priority: int = 2,
    ) -> HITLReview:
        """Create a HITL review task for human assessment.

        Args:
            tenant_id: Owning tenant UUID.
            routing_decision_id: Parent RoutingDecision UUID.
            task_type: Task category.
            ai_output: AI-generated output for the reviewer to assess.
            reviewer_id: Optional pre-assigned reviewer UUID.
            priority: Review priority (1=low, 2=normal, 3=high, 4=critical).

        Returns:
            Newly created HITLReview in pending status.
        """
        from datetime import timedelta

        due_at = datetime.now(tz=timezone.utc) + timedelta(hours=self._timeout_hours)

        review = await self._reviews.create(
            tenant_id=tenant_id,
            routing_decision_id=routing_decision_id,
            task_type=task_type,
            ai_output=ai_output,
            reviewer_id=reviewer_id,
            due_at=due_at,
            priority=priority,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.hitl.review_created",
                "tenant_id": str(tenant_id),
                "review_id": str(review.id),
                "routing_decision_id": str(routing_decision_id),
                "task_type": task_type,
                "priority": priority,
                "due_at": due_at.isoformat(),
            },
        )

        logger.info(
            "HITL review created",
            tenant_id=str(tenant_id),
            review_id=str(review.id),
            task_type=task_type,
            priority=priority,
        )

        return review

    async def get_review(
        self, review_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> HITLReview:
        """Retrieve a HITL review by ID.

        Args:
            review_id: HITLReview UUID.
            tenant_id: Requesting tenant.

        Returns:
            HITLReview.

        Raises:
            NotFoundError: If review not found.
        """
        review = await self._reviews.get_by_id(review_id, tenant_id)
        if review is None:
            raise NotFoundError(
                message=f"HITL review {review_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return review

    async def submit_decision(
        self,
        review_id: uuid.UUID,
        tenant_id: uuid.UUID,
        decision: str,
        reviewer_output: dict[str, Any],
        reviewer_id: uuid.UUID,
        reviewer_notes: str | None = None,
    ) -> HITLReview:
        """Submit a reviewer decision on a HITL review task.

        Decision is immutable once submitted. Creates an attribution record
        after the decision is recorded.

        Args:
            review_id: HITLReview UUID.
            tenant_id: Requesting tenant.
            decision: approved | rejected | modified.
            reviewer_output: Final output from the reviewer.
            reviewer_id: Reviewer user UUID.
            reviewer_notes: Optional reviewer comments.

        Returns:
            Updated HITLReview with decision recorded.

        Raises:
            NotFoundError: If review not found.
            ConflictError: If review is already decided or decision is invalid.
        """
        if decision not in VALID_REVIEW_DECISIONS:
            raise ConflictError(
                message=f"Invalid decision '{decision}'. Valid: {VALID_REVIEW_DECISIONS}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        review = await self.get_review(review_id, tenant_id)

        if review.status in TERMINAL_REVIEW_STATUSES:
            raise ConflictError(
                message=(
                    f"Review {review_id} is already in terminal status '{review.status}'. "
                    "Review decisions are immutable."
                ),
                error_code=ErrorCode.INVALID_OPERATION,
            )

        now = datetime.now(tz=timezone.utc)
        review = await self._reviews.submit_decision(
            review_id=review_id,
            decision=decision,
            reviewer_output=reviewer_output,
            reviewer_notes=reviewer_notes,
            review_completed_at=now,
        )

        # Compute attribution based on decision
        ai_score = 0.0
        human_score = 1.0
        if decision == "approved":
            # Reviewer approved AI output — AI contributed
            ai_score = 1.0
            human_score = 0.0
        elif decision == "modified":
            # Hybrid: AI produced draft, human refined
            ai_score = 0.5
            human_score = 0.5

        await self._attribution.create(
            tenant_id=tenant_id,
            task_id=review.routing_decision_id,
            task_type=review.task_type,
            routing_decision_id=review.routing_decision_id,
            hitl_review_id=review.id,
            ai_contribution_score=ai_score,
            human_contribution_score=human_score,
            time_saved_seconds=None,
            attribution_method="review_decision",
            resolution=decision,
            attribution_metadata={"reviewer_id": str(reviewer_id), "decision": decision},
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.hitl.decision_submitted",
                "tenant_id": str(tenant_id),
                "review_id": str(review_id),
                "decision": decision,
                "reviewer_id": str(reviewer_id),
                "ai_contribution_score": ai_score,
                "human_contribution_score": human_score,
            },
        )

        logger.info(
            "HITL review decision submitted",
            review_id=str(review_id),
            decision=decision,
            reviewer_id=str(reviewer_id),
            ai_contribution_score=ai_score,
        )

        return review

    async def list_reviews(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        reviewer_id: uuid.UUID | None = None,
        task_type: str | None = None,
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
        return await self._reviews.list_by_tenant(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            status=status,
            reviewer_id=reviewer_id,
            task_type=task_type,
        )


class AttributionService:
    """Generate productivity attribution reports for AI vs. human contributions.

    Aggregates attribution records to measure how much value AI vs. human
    reviewers contribute across task types and time periods.
    """

    def __init__(
        self,
        attribution_repo: IAttributionRepository,
        event_publisher: EventPublisher,
        default_report_days: int = 30,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            attribution_repo: AttributionRecord persistence.
            event_publisher: Kafka event publisher.
            default_report_days: Default lookback window for reports.
        """
        self._attribution = attribution_repo
        self._publisher = event_publisher
        self._default_days = default_report_days

    async def create_attribution(
        self,
        tenant_id: uuid.UUID,
        task_id: uuid.UUID,
        task_type: str,
        routing_decision_id: uuid.UUID | None,
        hitl_review_id: uuid.UUID | None,
        ai_contribution_score: float,
        human_contribution_score: float,
        resolution: str,
        time_saved_seconds: int | None = None,
        attribution_method: str = "routing_outcome",
        attribution_metadata: dict[str, Any] | None = None,
    ) -> AttributionRecord:
        """Record an attribution for a completed task.

        Args:
            tenant_id: Owning tenant UUID.
            task_id: External task UUID.
            task_type: Task category.
            routing_decision_id: Linked routing decision.
            hitl_review_id: Linked HITL review.
            ai_contribution_score: AI fraction (0–1).
            human_contribution_score: Human fraction (0–1).
            resolution: Final resolution: ai | human | hybrid.
            time_saved_seconds: Optional estimated time saved.
            attribution_method: Computation method.
            attribution_metadata: Additional context.

        Returns:
            Newly created AttributionRecord.

        Raises:
            ConflictError: If scores don't sum to approximately 1.0.
        """
        total = ai_contribution_score + human_contribution_score
        if abs(total - 1.0) > 0.01:
            raise ConflictError(
                message=(
                    f"Attribution scores must sum to 1.0. Got ai={ai_contribution_score} "
                    f"+ human={human_contribution_score} = {total}"
                ),
                error_code=ErrorCode.INVALID_OPERATION,
            )

        if resolution not in VALID_ROUTING_OUTCOMES:
            raise ConflictError(
                message=f"Invalid resolution '{resolution}'. Valid: {VALID_ROUTING_OUTCOMES}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        record = await self._attribution.create(
            tenant_id=tenant_id,
            task_id=task_id,
            task_type=task_type,
            routing_decision_id=routing_decision_id,
            hitl_review_id=hitl_review_id,
            ai_contribution_score=ai_contribution_score,
            human_contribution_score=human_contribution_score,
            time_saved_seconds=time_saved_seconds,
            attribution_method=attribution_method,
            resolution=resolution,
            attribution_metadata=attribution_metadata or {},
        )

        logger.info(
            "Attribution record created",
            tenant_id=str(tenant_id),
            task_id=str(task_id),
            resolution=resolution,
            ai_score=ai_contribution_score,
            human_score=human_contribution_score,
        )

        return record

    async def get_report(
        self,
        tenant_id: uuid.UUID,
        days: int | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Generate an attribution analytics report.

        Args:
            tenant_id: Requesting tenant.
            days: Lookback window in days (defaults to service default).
            task_type: Optional task type filter.

        Returns:
            Attribution report with AI/human contribution breakdown.
        """
        lookback = days if days is not None else self._default_days
        return await self._attribution.get_report(
            tenant_id=tenant_id,
            days=lookback,
            task_type=task_type,
        )


class FeedbackService:
    """Process human corrections and trigger AI confidence recalibration.

    Collects feedback corrections from human reviewers and uses them
    to recalibrate the confidence scoring model for affected task types.
    """

    def __init__(
        self,
        feedback_repo: IFeedbackRepository,
        confidence_engine: IConfidenceEngineAdapter,
        event_publisher: EventPublisher,
        calibration_min_samples: int = 10,
        feedback_decay_factor: float = 0.9,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            feedback_repo: FeedbackCorrection persistence.
            confidence_engine: AI confidence scoring and recalibration adapter.
            event_publisher: Kafka event publisher.
            calibration_min_samples: Minimum corrections before recalibration triggers.
            feedback_decay_factor: Exponential decay for older samples.
        """
        self._feedback = feedback_repo
        self._confidence = confidence_engine
        self._publisher = event_publisher
        self._min_samples = calibration_min_samples
        self._decay_factor = feedback_decay_factor

    async def submit_correction(
        self,
        tenant_id: uuid.UUID,
        routing_decision_id: uuid.UUID,
        task_type: str,
        original_confidence: float,
        original_routing: str,
        corrected_routing: str,
        correction_type: str,
        hitl_review_id: uuid.UUID | None = None,
        correction_reason: str | None = None,
        submitted_by: uuid.UUID | None = None,
    ) -> FeedbackCorrection:
        """Submit a human correction for a routing decision.

        After recording the correction, checks if the sample threshold
        has been reached and triggers recalibration if so.

        Args:
            tenant_id: Owning tenant UUID.
            routing_decision_id: The routing decision being corrected.
            task_type: Task type for recalibration grouping.
            original_confidence: Confidence score at routing time.
            original_routing: Original routing outcome.
            corrected_routing: What the routing should have been.
            correction_type: Correction category.
            hitl_review_id: Optional linked HITL review.
            correction_reason: Human explanation.
            submitted_by: Submitting user UUID.

        Returns:
            Newly created FeedbackCorrection.

        Raises:
            ConflictError: If routing values or correction_type are invalid.
        """
        if original_routing not in VALID_ROUTING_OUTCOMES:
            raise ConflictError(
                message=f"Invalid original_routing '{original_routing}'.",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        if corrected_routing not in VALID_ROUTING_OUTCOMES:
            raise ConflictError(
                message=f"Invalid corrected_routing '{corrected_routing}'.",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        if correction_type not in VALID_CORRECTION_TYPES:
            raise ConflictError(
                message=f"Invalid correction_type '{correction_type}'. Valid: {VALID_CORRECTION_TYPES}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        correction = await self._feedback.create(
            tenant_id=tenant_id,
            routing_decision_id=routing_decision_id,
            hitl_review_id=hitl_review_id,
            task_type=task_type,
            original_confidence=original_confidence,
            original_routing=original_routing,
            corrected_routing=corrected_routing,
            correction_reason=correction_reason,
            correction_type=correction_type,
            submitted_by=submitted_by,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_FEEDBACK,
            {
                "event_type": "hac.feedback.correction_submitted",
                "tenant_id": str(tenant_id),
                "correction_id": str(correction.id),
                "routing_decision_id": str(routing_decision_id),
                "task_type": task_type,
                "original_routing": original_routing,
                "corrected_routing": corrected_routing,
                "correction_type": correction_type,
            },
        )

        logger.info(
            "Feedback correction submitted",
            tenant_id=str(tenant_id),
            correction_id=str(correction.id),
            task_type=task_type,
            original_routing=original_routing,
            corrected_routing=corrected_routing,
        )

        # Check if recalibration threshold reached
        await self._maybe_recalibrate(tenant_id, task_type)

        return correction

    async def _maybe_recalibrate(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> None:
        """Trigger recalibration if enough uncalibrated corrections exist.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to check and potentially recalibrate.
        """
        uncalibrated = await self._feedback.list_uncalibrated(
            tenant_id=tenant_id, task_type=task_type, limit=self._min_samples + 1
        )

        if len(uncalibrated) < self._min_samples:
            return

        corrections_data = [
            {
                "original_confidence": c.original_confidence,
                "original_routing": c.original_routing,
                "corrected_routing": c.corrected_routing,
                "correction_type": c.correction_type,
            }
            for c in uncalibrated[: self._min_samples]
        ]

        result = await self._confidence.recalibrate(
            task_type=task_type,
            corrections=corrections_data,
            decay_factor=self._decay_factor,
        )

        now = datetime.now(tz=timezone.utc)
        correction_ids = [c.id for c in uncalibrated[: self._min_samples]]
        await self._feedback.mark_calibrated(
            correction_ids=correction_ids,
            calibration_applied_at=now,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_FEEDBACK,
            {
                "event_type": "hac.feedback.recalibration_triggered",
                "tenant_id": str(tenant_id),
                "task_type": task_type,
                "sample_count": len(correction_ids),
                "recalibration_result": result,
            },
        )

        logger.info(
            "Confidence recalibration triggered",
            tenant_id=str(tenant_id),
            task_type=task_type,
            sample_count=len(correction_ids),
        )

    async def get_calibration_summary(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> dict[str, Any]:
        """Get feedback calibration summary for a task type.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to summarise.

        Returns:
            Calibration summary dict.
        """
        return await self._feedback.get_calibration_summary(tenant_id, task_type)


# ---------------------------------------------------------------------------
# New services wiring the 8 new domain adapters
# ---------------------------------------------------------------------------


class FeedbackAggregationService:
    """Aggregate and analyse user feedback for model and feature quality.

    Stores individual feedback records and produces trend reports, top-issue
    summaries, and volume statistics. Publishes events on aggregation milestones.
    """

    def __init__(
        self,
        feedback_aggregator: IFeedbackAggregator,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            feedback_aggregator: Feedback storage and aggregation adapter.
            event_publisher: Kafka event publisher.
        """
        self._aggregator = feedback_aggregator
        self._publisher = event_publisher

    async def submit_feedback(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        feature: str,
        rating: int,
        category: str,
        text: str | None = None,
        submitted_by: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Store feedback and emit a Kafka event.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Model being rated.
            feature: Feature being rated.
            rating: Integer rating 1–5.
            category: Feedback category.
            text: Optional free-text comment.
            submitted_by: Optional submitter UUID.

        Returns:
            Stored feedback record as a dict.

        Raises:
            ValueError: If rating or category is invalid.
        """
        record = await self._aggregator.store_feedback(
            tenant_id=tenant_id,
            model_id=model_id,
            feature=feature,
            rating=rating,
            category=category,
            text=text,
            submitted_by=submitted_by,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_FEEDBACK,
            {
                "event_type": "hac.feedback.submitted",
                "tenant_id": str(tenant_id),
                "model_id": model_id,
                "feature": feature,
                "rating": rating,
                "category": category,
            },
        )

        logger.info(
            "User feedback submitted",
            tenant_id=str(tenant_id),
            model_id=model_id,
            feature=feature,
            rating=rating,
        )

        return record.to_dict() if hasattr(record, "to_dict") else record

    async def get_model_trends(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Return aggregated feedback trends for a model.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            window_days: Lookback window in days.

        Returns:
            Aggregate trend dict.
        """
        return await self._aggregator.aggregate_by_model(
            tenant_id=tenant_id,
            model_id=model_id,
            window_days=window_days,
        )

    async def get_top_issues(
        self,
        tenant_id: uuid.UUID,
        window_days: int = 7,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top user-reported issues.

        Args:
            tenant_id: Requesting tenant.
            window_days: Lookback window in days.
            top_n: Maximum issues to return.

        Returns:
            List of issue dicts sorted by severity.
        """
        return await self._aggregator.get_top_issues(
            tenant_id=tenant_id,
            window_days=window_days,
            top_n=top_n,
        )

    async def export_feedback(
        self,
        tenant_id: uuid.UUID,
        model_id: str | None = None,
        feature: str | None = None,
        category: str | None = None,
        window_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Export raw feedback records for downstream analysis.

        Args:
            tenant_id: Requesting tenant.
            model_id: Optional model filter.
            feature: Optional feature filter.
            category: Optional category filter.
            window_days: Optional lookback window.

        Returns:
            List of feedback record dicts.
        """
        return await self._aggregator.export_records(
            tenant_id=tenant_id,
            model_id=model_id,
            feature=feature,
            category=category,
            window_days=window_days,
        )


class AnnotationService:
    """Manage crowdsourced annotation tasks and annotation collection.

    Orchestrates task creation, annotator assignment, annotation storage,
    progress monitoring, and export. Publishes events on task completion.
    """

    def __init__(
        self,
        annotation_engine: IAnnotationEngine,
        consensus_scorer: IConsensusScorer,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            annotation_engine: Annotation task management adapter.
            consensus_scorer: Inter-annotator agreement adapter.
            event_publisher: Kafka event publisher.
        """
        self._engine = annotation_engine
        self._consensus = consensus_scorer
        self._publisher = event_publisher

    async def create_annotation_task(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        name: str,
        items: list[dict[str, Any]],
        labels: list[str],
        gold_items: list[dict[str, Any]] | None = None,
        annotations_per_item: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create and publish an annotation task.

        Args:
            tenant_id: Owning tenant UUID.
            task_type: Type of labeling task.
            name: Human-readable name.
            items: Items to annotate.
            labels: Valid label strings.
            gold_items: Optional gold standard items.
            annotations_per_item: Override annotations-per-item default.
            metadata: Additional configuration.

        Returns:
            Task summary dict.
        """
        task = await self._engine.create_task(
            tenant_id=tenant_id,
            task_type=task_type,
            name=name,
            items=items,
            labels=labels,
            gold_items=gold_items,
            annotations_per_item=annotations_per_item,
            metadata=metadata,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.annotation.task_created",
                "tenant_id": str(tenant_id),
                "task_id": task.get("task_id"),
                "task_type": task_type,
                "item_count": len(items),
            },
        )

        logger.info(
            "Annotation task created",
            tenant_id=str(tenant_id),
            task_type=task_type,
            item_count=len(items),
        )

        return task

    async def submit_annotation(
        self,
        task_id: uuid.UUID,
        item_id: str,
        annotator_id: uuid.UUID,
        label: str,
        confidence: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Store an annotation and return the result including gold-check outcome.

        Args:
            task_id: Task UUID.
            item_id: Item being annotated.
            annotator_id: Annotator UUID.
            label: Chosen label.
            confidence: Optional confidence.
            notes: Optional notes.

        Returns:
            Annotation result dict with gold_check result if applicable.
        """
        return await self._engine.store_annotation(
            task_id=task_id,
            item_id=item_id,
            annotator_id=annotator_id,
            label=label,
            confidence=confidence,
            notes=notes,
        )

    async def get_consensus_report(
        self,
        task_id: uuid.UUID,
        annotations: list[dict[str, str]],
        labels: list[str],
        annotator_ids: list[uuid.UUID],
    ) -> dict[str, Any]:
        """Generate a consensus agreement report for a task.

        Args:
            task_id: Task UUID (used as string label in report).
            annotations: All task annotation dicts.
            labels: All possible labels.
            annotator_ids: All annotator UUIDs.

        Returns:
            Consensus report dict.
        """
        return self._consensus.generate_consensus_report(
            task_id=str(task_id),
            annotations=annotations,
            labels=labels,
            annotator_ids=annotator_ids,
        )

    async def export_task_annotations(
        self,
        task_id: uuid.UUID,
        format: str = "jsonl",
    ) -> str:
        """Export annotations for a completed task.

        Args:
            task_id: Task UUID.
            format: "jsonl" or "csv".

        Returns:
            Exported content string.
        """
        return await self._engine.export_annotations(
            task_id=task_id,
            format=format,
            exclude_gold=True,
        )


class ActiveLearningService:
    """Orchestrate active learning cycles for efficient labelling.

    Manages unlabelled pools, selects samples to label using uncertainty
    strategies, and tracks learning curve progress.
    """

    def __init__(
        self,
        active_learner: IActiveLearner,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            active_learner: Uncertainty sampling adapter.
            event_publisher: Kafka event publisher.
        """
        self._learner = active_learner
        self._publisher = event_publisher

    async def add_unlabelled_pool(
        self,
        session_id: uuid.UUID,
        samples: list[dict[str, Any]],
    ) -> int:
        """Add samples to the active learning pool.

        Args:
            session_id: Active learning session UUID.
            samples: Unlabelled sample dicts.

        Returns:
            Total pool size after adding.
        """
        pool_size = self._learner.add_to_pool(session_id, samples)
        logger.info(
            "Samples added to active learning pool",
            session_id=str(session_id),
            added=len(samples),
            pool_size=pool_size,
        )
        return pool_size

    async def request_labels(
        self,
        session_id: uuid.UUID,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Select the most informative samples for labelling.

        Args:
            session_id: Active learning session UUID.
            batch_size: Override default batch size.

        Returns:
            Selection result dict with selected_samples.
        """
        result = await self._learner.select_samples(
            session_id=session_id,
            batch_size=batch_size,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.active_learning.samples_selected",
                "session_id": str(session_id),
                "round_index": result.get("round_index"),
                "selected_count": result.get("total_selected"),
                "budget_remaining": result.get("budget_remaining"),
            },
        )

        return result

    async def record_round_outcome(
        self,
        session_id: uuid.UUID,
        labels_used: int,
        estimated_accuracy: float,
    ) -> dict[str, Any]:
        """Record the outcome of a labelling round.

        Args:
            session_id: Active learning session UUID.
            labels_used: Number of labels used.
            estimated_accuracy: Model accuracy after this round.

        Returns:
            LearningCurvePoint as a dict.
        """
        point = await self._learner.record_round_result(
            session_id=session_id,
            labels_used=labels_used,
            estimated_accuracy=estimated_accuracy,
        )
        return point.to_dict() if hasattr(point, "to_dict") else point

    def get_budget_status(self) -> dict[str, Any]:
        """Return current label budget status.

        Returns:
            Budget status dict.
        """
        return self._learner.get_budget_status()


class ExplainabilityService:
    """Provide AI explanations to human reviewers.

    Retrieves SHAP or LIME explanations from the aumos-explainability service,
    formats them for human readability, and delivers the explanation package
    for the reviewer interface.
    """

    def __init__(
        self,
        explainability_bridge: IExplainabilityBridge,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            explainability_bridge: Explanation retrieval adapter.
            event_publisher: Kafka event publisher.
        """
        self._bridge = explainability_bridge
        self._publisher = event_publisher

    async def get_review_explanation(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        method: str = "shap",
    ) -> dict[str, Any]:
        """Get a formatted explanation package for a reviewer.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            input_data: Prediction input data.
            prediction: Model output/prediction.
            method: Explanation method — "shap" or "lime".

        Returns:
            Reviewer explanation package dict.
        """
        package = await self._bridge.get_explanation_for_review(
            model_id=model_id,
            input_data=input_data,
            prediction=prediction,
            tenant_id=tenant_id,
            method=method,
        )

        logger.info(
            "Explanation package delivered to reviewer",
            tenant_id=str(tenant_id),
            model_id=model_id,
            method=method,
            explanation_confidence=package.get("explanation_confidence"),
        )

        return package


class ModelPerformanceService:
    """Track and report on model performance over time.

    Records metric snapshots, computes trends and velocity, performs A/B
    comparisons of model versions, and generates comprehensive performance reports.
    """

    def __init__(
        self,
        performance_tracker: IPerformanceTracker,
        event_publisher: EventPublisher,
        alert_threshold_delta: float = -0.02,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            performance_tracker: Metric tracking adapter.
            event_publisher: Kafka event publisher.
            alert_threshold_delta: Minimum acceptable delta before raising a degradation alert.
        """
        self._tracker = performance_tracker
        self._publisher = event_publisher
        self._alert_threshold = alert_threshold_delta

    async def record_metric(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        model_version: str,
        metric_name: str,
        metric_value: float,
        segment: str | None = None,
        data_type: str | None = None,
        feedback_round_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a model metric snapshot and emit an event.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Model identifier.
            model_version: Model version string.
            metric_name: Metric name.
            metric_value: Measured value.
            segment: Optional user segment.
            data_type: Optional data type.
            feedback_round_id: Optional feedback round identifier.

        Returns:
            Snapshot as dict.
        """
        snapshot = await self._tracker.record_metric(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version=model_version,
            metric_name=metric_name,
            metric_value=metric_value,
            segment=segment,
            data_type=data_type,
            feedback_round_id=feedback_round_id,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_FEEDBACK,
            {
                "event_type": "hac.performance.metric_recorded",
                "tenant_id": str(tenant_id),
                "model_id": model_id,
                "model_version": model_version,
                "metric_name": metric_name,
                "metric_value": metric_value,
            },
        )

        return snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot

    async def get_trend(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        metric_name: str = "accuracy",
        window_days: int = 30,
        segment: str | None = None,
    ) -> dict[str, Any]:
        """Return metric trend and raise a degradation alert if needed.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            metric_name: Metric to trend.
            window_days: Lookback window.
            segment: Optional segment filter.

        Returns:
            Trend dict.
        """
        trend = await self._tracker.get_accuracy_trend(
            tenant_id=tenant_id,
            model_id=model_id,
            metric_name=metric_name,
            window_days=window_days,
            segment=segment,
        )

        delta = trend.get("delta")
        if delta is not None and delta < self._alert_threshold:
            await self._publisher.publish(
                Topics.HUMAN_AI_FEEDBACK,
                {
                    "event_type": "hac.performance.degradation_alert",
                    "tenant_id": str(tenant_id),
                    "model_id": model_id,
                    "metric_name": metric_name,
                    "delta": delta,
                    "trend_direction": trend.get("trend_direction"),
                },
            )
            logger.warning(
                "Model performance degradation detected",
                tenant_id=str(tenant_id),
                model_id=model_id,
                metric_name=metric_name,
                delta=delta,
            )

        return trend

    async def compare_ab(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        version_a: str,
        version_b: str,
        metric_name: str = "accuracy",
        window_days: int = 14,
    ) -> dict[str, Any]:
        """Run an A/B performance comparison between two model versions.

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
        return await self._tracker.compare_versions(
            tenant_id=tenant_id,
            model_id=model_id,
            version_a=version_a,
            version_b=version_b,
            metric_name=metric_name,
            window_days=window_days,
        )

    async def generate_report(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        window_days: int = 30,
        include_versions: list[str] | None = None,
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
        return await self._tracker.generate_performance_report(
            tenant_id=tenant_id,
            model_id=model_id,
            window_days=window_days,
            include_versions=include_versions,
        )


class ConfidenceCalibrationService:
    """Drive confidence score recalibration from feedback-derived ground truth.

    Wraps the CalibrationEngine adapter with business logic: validates that
    enough samples exist before calibrating, computes ECE improvement, and
    emits recalibration events.
    """

    def __init__(
        self,
        calibration_engine: ICalibrationEngine,
        feedback_repo: IFeedbackRepository,
        event_publisher: EventPublisher,
        min_samples_for_calibration: int = 20,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            calibration_engine: Calibration method adapter.
            feedback_repo: FeedbackCorrection persistence.
            event_publisher: Kafka event publisher.
            min_samples_for_calibration: Minimum sample count before calibrating.
        """
        self._engine = calibration_engine
        self._feedback = feedback_repo
        self._publisher = event_publisher
        self._min_samples = min_samples_for_calibration

    async def run_calibration(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        confidences: list[float],
        labels: list[int],
        method: str = "temperature_scaling",
    ) -> dict[str, Any]:
        """Run confidence calibration and emit an event.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type to calibrate.
            confidences: Historical confidence scores.
            labels: Ground-truth binary labels.
            method: Calibration method to apply.

        Returns:
            CalibrationRecord as dict.

        Raises:
            ConflictError: If fewer samples than minimum threshold.
        """
        if len(confidences) < self._min_samples:
            raise ConflictError(
                message=(
                    f"Insufficient samples for calibration. "
                    f"Need {self._min_samples}, got {len(confidences)}."
                ),
                error_code=ErrorCode.INVALID_OPERATION,
            )

        record = await self._engine.recalibrate_from_feedback(
            tenant_id=tenant_id,
            task_type=task_type,
            confidences=confidences,
            labels=labels,
            method=method,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_FEEDBACK,
            {
                "event_type": "hac.calibration.applied",
                "tenant_id": str(tenant_id),
                "task_type": task_type,
                "method": method,
                "ece_before": record.ece_before if hasattr(record, "ece_before") else None,
                "ece_after": record.ece_after if hasattr(record, "ece_after") else None,
                "sample_count": len(confidences),
            },
        )

        logger.info(
            "Confidence calibration completed",
            tenant_id=str(tenant_id),
            task_type=task_type,
            method=method,
            sample_count=len(confidences),
        )

        return record.to_dict() if hasattr(record, "to_dict") else record

    def apply_to_confidence(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        raw_confidence: float,
    ) -> float:
        """Apply current calibration parameters to a raw confidence score.

        Returns raw_confidence unchanged if no calibration exists for the task type.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type.
            raw_confidence: Raw model confidence.

        Returns:
            Calibrated confidence score.
        """
        params = self._engine.get_calibration_history(
            tenant_id=tenant_id,
            task_type=task_type,
            limit=1,
        )
        if not params:
            return raw_confidence

        latest_params = params[0].get("params", {})
        return self._engine.apply_calibration(raw_confidence, latest_params)

    def get_calibration_history(
        self,
        tenant_id: uuid.UUID,
        task_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return calibration history for a tenant.

        Args:
            tenant_id: Requesting tenant.
            task_type: Optional task type filter.
            limit: Maximum records to return.

        Returns:
            List of CalibrationRecord dicts.
        """
        return self._engine.get_calibration_history(
            tenant_id=tenant_id,
            task_type=task_type,
            limit=limit,
        )


class ReviewQueueService:
    """Manage the HITL review priority queue.

    Handles enqueuing newly routed reviews, assigning them to reviewers,
    completing them, monitoring depth, and handling timeouts with reassignment.
    """

    def __init__(
        self,
        queue_manager: IReviewQueueManager,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            queue_manager: Priority queue adapter.
            event_publisher: Kafka event publisher.
        """
        self._queue = queue_manager
        self._publisher = event_publisher

    async def enqueue_review(
        self,
        tenant_id: uuid.UUID,
        review_id: uuid.UUID,
        task_type: str,
        confidence: float,
        urgency: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enqueue a new HITL review and publish an event.

        Args:
            tenant_id: Owning tenant UUID.
            review_id: HITLReview UUID.
            task_type: Task category.
            confidence: Model confidence score.
            urgency: Review urgency 1–4.
            metadata: Optional review context.

        Returns:
            QueueItem as dict.
        """
        item = await self._queue.enqueue(
            tenant_id=tenant_id,
            review_id=review_id,
            task_type=task_type,
            confidence=confidence,
            urgency=urgency,
            metadata=metadata,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.queue.review_enqueued",
                "tenant_id": str(tenant_id),
                "review_id": str(review_id),
                "task_type": task_type,
                "urgency": urgency,
                "priority_score": item.priority_score if hasattr(item, "priority_score") else None,
            },
        )

        return item.to_dict() if hasattr(item, "to_dict") else item

    async def claim_next_review(
        self,
        tenant_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Assign the next highest-priority review to a reviewer.

        Args:
            tenant_id: Requesting tenant.
            reviewer_id: Reviewer UUID.

        Returns:
            Assigned QueueItem as dict, or None if queue is empty.
        """
        item = await self._queue.assign_to_reviewer(
            tenant_id=tenant_id,
            reviewer_id=reviewer_id,
        )
        if item is None:
            return None

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.queue.review_assigned",
                "tenant_id": str(tenant_id),
                "reviewer_id": str(reviewer_id),
                "review_id": str(item.review_id) if hasattr(item, "review_id") else None,
                "wait_seconds": item.wait_seconds if hasattr(item, "wait_seconds") else None,
            },
        )

        return item.to_dict() if hasattr(item, "to_dict") else item

    async def complete_review(
        self,
        tenant_id: uuid.UUID,
        item_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Mark a queue item as completed.

        Args:
            tenant_id: Owning tenant UUID.
            item_id: Queue item UUID.

        Returns:
            Completed QueueItem as dict.
        """
        item = await self._queue.mark_completed(
            tenant_id=tenant_id,
            item_id=item_id,
        )

        await self._publisher.publish(
            Topics.HUMAN_AI_HITL,
            {
                "event_type": "hac.queue.review_completed",
                "tenant_id": str(tenant_id),
                "item_id": str(item_id),
            },
        )

        return item.to_dict() if hasattr(item, "to_dict") else item

    async def get_analytics(self, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Return queue analytics for a tenant.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Queue analytics dict.
        """
        return await self._queue.get_queue_analytics(tenant_id)

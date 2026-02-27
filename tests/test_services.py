"""Tests for the core business logic services.

Covers RoutingService, ComplianceGateService, HITLReviewService,
AttributionService, and FeedbackService. All external dependencies are
replaced with AsyncMocks so no database, Kafka, or Redis is required.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from aumos_common.errors import ConflictError, NotFoundError

from aumos_human_ai_collab.core.services import (
    AttributionService,
    ComplianceGateService,
    FeedbackService,
    HITLReviewService,
    RoutingService,
)
from tests.conftest import (
    DECISION_ID,
    GATE_ID,
    REVIEW_ID,
    REVIEWER_ID,
    TASK_ID,
    TENANT_ID,
    make_compliance_gate,
    make_feedback_correction,
    make_hitl_review,
    make_routing_decision,
)


# ---------------------------------------------------------------------------
# RoutingService tests
# ---------------------------------------------------------------------------


class TestRoutingServiceEvaluate:
    """Tests for RoutingService.evaluate_routing."""

    @pytest.fixture()
    def routing_service(
        self,
        mock_routing_repo: AsyncMock,
        mock_compliance_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> RoutingService:
        """Build a RoutingService with mocked dependencies.

        Args:
            mock_routing_repo: Mock routing decision repository.
            mock_compliance_repo: Mock compliance gate repository.
            mock_confidence_engine: Mock confidence engine adapter.
            mock_event_publisher: Mock Kafka event publisher.

        Returns:
            RoutingService configured with defaults (ai_threshold=0.85, hybrid_lower=0.65).
        """
        return RoutingService(
            routing_repo=mock_routing_repo,
            compliance_repo=mock_compliance_repo,
            confidence_engine=mock_confidence_engine,
            event_publisher=mock_event_publisher,
            ai_confidence_threshold=0.85,
            hybrid_confidence_lower=0.65,
        )

    @pytest.mark.asyncio()
    async def test_high_confidence_routes_to_ai(
        self,
        routing_service: RoutingService,
        mock_confidence_engine: AsyncMock,
        mock_routing_repo: AsyncMock,
    ) -> None:
        """Confidence above ai_threshold should produce an 'ai' routing outcome.

        Args:
            routing_service: Service under test.
            mock_confidence_engine: Pre-configured to return score 0.90.
            mock_routing_repo: Mock repository to inspect create call.
        """
        mock_confidence_engine.score_task.return_value = (0.90, "model-v1")
        mock_routing_repo.create.return_value = make_routing_decision(routing_outcome="ai")

        decision = await routing_service.evaluate_routing(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="document_analysis",
            task_payload={"text": "some content"},
        )

        assert decision.routing_outcome == "ai"
        _, kwargs = mock_routing_repo.create.call_args
        assert kwargs["routing_outcome"] == "ai"
        assert kwargs["compliance_gate_triggered"] is False

    @pytest.mark.asyncio()
    async def test_low_confidence_routes_to_human(
        self,
        routing_service: RoutingService,
        mock_confidence_engine: AsyncMock,
        mock_routing_repo: AsyncMock,
    ) -> None:
        """Confidence below hybrid_lower should produce a 'human' routing outcome.

        Args:
            routing_service: Service under test.
            mock_confidence_engine: Will be configured to return 0.50.
            mock_routing_repo: Mock repository to inspect create call.
        """
        mock_confidence_engine.score_task.return_value = (0.50, "model-v1")
        mock_routing_repo.create.return_value = make_routing_decision(routing_outcome="human")

        decision = await routing_service.evaluate_routing(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="medical_diagnosis",
            task_payload={},
        )

        assert decision.routing_outcome == "human"
        _, kwargs = mock_routing_repo.create.call_args
        assert kwargs["routing_outcome"] == "human"

    @pytest.mark.asyncio()
    async def test_mid_confidence_routes_to_hybrid(
        self,
        routing_service: RoutingService,
        mock_confidence_engine: AsyncMock,
        mock_routing_repo: AsyncMock,
    ) -> None:
        """Confidence between hybrid_lower and ai_threshold should produce 'hybrid'.

        Args:
            routing_service: Service under test.
            mock_confidence_engine: Will be configured to return 0.75.
            mock_routing_repo: Mock repository to inspect create call.
        """
        mock_confidence_engine.score_task.return_value = (0.75, "model-v1")
        mock_routing_repo.create.return_value = make_routing_decision(routing_outcome="hybrid")

        decision = await routing_service.evaluate_routing(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="contract_review",
            task_payload={"pages": 5},
        )

        assert decision.routing_outcome == "hybrid"

    @pytest.mark.asyncio()
    async def test_compliance_gate_overrides_high_confidence(
        self,
        routing_service: RoutingService,
        mock_confidence_engine: AsyncMock,
        mock_compliance_repo: AsyncMock,
        mock_routing_repo: AsyncMock,
    ) -> None:
        """Active compliance gate must force 'human' even when confidence is very high.

        This is a key invariant: compliance gates always override confidence scores.

        Args:
            routing_service: Service under test.
            mock_confidence_engine: Will return 0.99 (highest possible).
            mock_compliance_repo: Will return a matching gate.
            mock_routing_repo: Mock repository to inspect create call.
        """
        mock_confidence_engine.score_task.return_value = (0.99, "model-v1")
        mock_compliance_repo.find_matching_gates.return_value = [make_compliance_gate()]
        mock_routing_repo.create.return_value = make_routing_decision(
            routing_outcome="human", compliance_gate_triggered=True
        )

        decision = await routing_service.evaluate_routing(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="medical_diagnosis",
            task_payload={},
        )

        assert decision.routing_outcome == "human"
        _, kwargs = mock_routing_repo.create.call_args
        assert kwargs["compliance_gate_triggered"] is True
        assert kwargs["compliance_gate_id"] == GATE_ID

    @pytest.mark.asyncio()
    async def test_override_routing_is_respected(
        self,
        routing_service: RoutingService,
        mock_confidence_engine: AsyncMock,
        mock_routing_repo: AsyncMock,
    ) -> None:
        """A valid override_routing value should bypass threshold logic.

        Args:
            routing_service: Service under test.
            mock_confidence_engine: Returns a high score, but override takes precedence.
            mock_routing_repo: Mock repository to inspect create call.
        """
        mock_confidence_engine.score_task.return_value = (0.95, "model-v1")
        mock_routing_repo.create.return_value = make_routing_decision(routing_outcome="human")

        decision = await routing_service.evaluate_routing(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="document_analysis",
            task_payload={},
            override_routing="human",
        )

        assert decision.routing_outcome == "human"

    @pytest.mark.asyncio()
    async def test_invalid_override_routing_raises_conflict_error(
        self,
        routing_service: RoutingService,
    ) -> None:
        """An invalid override_routing value must raise ConflictError before any IO.

        Args:
            routing_service: Service under test.
        """
        with pytest.raises(ConflictError, match="Invalid override_routing"):
            await routing_service.evaluate_routing(
                tenant_id=TENANT_ID,
                task_id=TASK_ID,
                task_type="document_analysis",
                task_payload={},
                override_routing="invalid_value",
            )

    @pytest.mark.asyncio()
    async def test_event_published_after_routing_decision(
        self,
        routing_service: RoutingService,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """evaluate_routing must publish exactly one event after recording the decision.

        Args:
            routing_service: Service under test.
            mock_event_publisher: Event publisher mock to assert publish was called.
        """
        await routing_service.evaluate_routing(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="document_analysis",
            task_payload={},
        )

        mock_event_publisher.publish.assert_called_once()
        topic_arg = mock_event_publisher.publish.call_args[0][0]
        event_arg = mock_event_publisher.publish.call_args[0][1]
        assert event_arg["event_type"] == "hac.routing.evaluated"
        assert event_arg["task_type"] == "document_analysis"

    @pytest.mark.asyncio()
    async def test_get_decision_not_found_raises_not_found_error(
        self,
        routing_service: RoutingService,
        mock_routing_repo: AsyncMock,
    ) -> None:
        """get_decision must raise NotFoundError when the record does not exist.

        Args:
            routing_service: Service under test.
            mock_routing_repo: Configured to return None.
        """
        mock_routing_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await routing_service.get_decision(uuid.uuid4(), TENANT_ID)


# ---------------------------------------------------------------------------
# ComplianceGateService tests
# ---------------------------------------------------------------------------


class TestComplianceGateService:
    """Tests for ComplianceGateService."""

    @pytest.fixture()
    def compliance_service(
        self,
        mock_compliance_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> ComplianceGateService:
        """Build ComplianceGateService with mocked dependencies.

        Args:
            mock_compliance_repo: Mock compliance gate repository.
            mock_event_publisher: Mock Kafka event publisher.

        Returns:
            ComplianceGateService instance.
        """
        return ComplianceGateService(
            compliance_repo=mock_compliance_repo,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_create_gate_valid_regulation_succeeds(
        self,
        compliance_service: ComplianceGateService,
        mock_compliance_repo: AsyncMock,
    ) -> None:
        """Creating a gate with a valid regulation string should persist and return a gate.

        Args:
            compliance_service: Service under test.
            mock_compliance_repo: Mock repository to verify create was called.
        """
        gate = await compliance_service.create_gate(
            tenant_id=TENANT_ID,
            gate_name="hipaa-clinical-notes",
            regulation="hipaa",
            task_types=["medical_diagnosis"],
        )

        assert gate.regulation == "hipaa"
        mock_compliance_repo.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_gate_invalid_regulation_raises_conflict_error(
        self,
        compliance_service: ComplianceGateService,
    ) -> None:
        """Creating a gate with an unsupported regulation must raise ConflictError.

        Args:
            compliance_service: Service under test.
        """
        with pytest.raises(ConflictError, match="Invalid regulation"):
            await compliance_service.create_gate(
                tenant_id=TENANT_ID,
                gate_name="unknown-gate",
                regulation="unknown_regulation",
                task_types=["task_x"],
            )

    @pytest.mark.asyncio()
    async def test_create_gate_empty_task_types_raises_conflict_error(
        self,
        compliance_service: ComplianceGateService,
    ) -> None:
        """Creating a gate with an empty task_types list must raise ConflictError.

        Args:
            compliance_service: Service under test.
        """
        with pytest.raises(ConflictError, match="At least one task_type"):
            await compliance_service.create_gate(
                tenant_id=TENANT_ID,
                gate_name="empty-gate",
                regulation="gdpr",
                task_types=[],
            )

    @pytest.mark.asyncio()
    async def test_evaluate_gate_with_match_returns_triggered_true(
        self,
        compliance_service: ComplianceGateService,
        mock_compliance_repo: AsyncMock,
    ) -> None:
        """evaluate_gate must return triggered=True when matching gates exist.

        Args:
            compliance_service: Service under test.
            mock_compliance_repo: Configured to return one matching gate.
        """
        mock_compliance_repo.find_matching_gates.return_value = [make_compliance_gate()]

        result = await compliance_service.evaluate_gate(TENANT_ID, "medical_diagnosis")

        assert result["triggered"] is True
        assert len(result["gate_ids"]) == 1
        assert "hipaa" in result["regulations"]

    @pytest.mark.asyncio()
    async def test_evaluate_gate_no_match_returns_triggered_false(
        self,
        compliance_service: ComplianceGateService,
        mock_compliance_repo: AsyncMock,
    ) -> None:
        """evaluate_gate must return triggered=False when no gates match the task type.

        Args:
            compliance_service: Service under test.
            mock_compliance_repo: Configured to return empty list.
        """
        mock_compliance_repo.find_matching_gates.return_value = []

        result = await compliance_service.evaluate_gate(TENANT_ID, "simple_task")

        assert result["triggered"] is False
        assert result["gate_ids"] == []

    @pytest.mark.asyncio()
    async def test_get_gate_not_found_raises_not_found_error(
        self,
        compliance_service: ComplianceGateService,
        mock_compliance_repo: AsyncMock,
    ) -> None:
        """get_gate must raise NotFoundError when the gate does not exist.

        Args:
            compliance_service: Service under test.
            mock_compliance_repo: Configured to return None.
        """
        mock_compliance_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await compliance_service.get_gate(uuid.uuid4(), TENANT_ID)


# ---------------------------------------------------------------------------
# HITLReviewService tests
# ---------------------------------------------------------------------------


class TestHITLReviewService:
    """Tests for HITLReviewService."""

    @pytest.fixture()
    def hitl_service(
        self,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> HITLReviewService:
        """Build HITLReviewService with mocked dependencies.

        Args:
            mock_hitl_repo: Mock HITL review repository.
            mock_attribution_repo: Mock attribution repository.
            mock_event_publisher: Mock Kafka event publisher.

        Returns:
            HITLReviewService instance.
        """
        return HITLReviewService(
            review_repo=mock_hitl_repo,
            attribution_repo=mock_attribution_repo,
            event_publisher=mock_event_publisher,
            review_timeout_hours=24,
        )

    @pytest.mark.asyncio()
    async def test_create_review_returns_pending_review(
        self,
        hitl_service: HITLReviewService,
        mock_hitl_repo: AsyncMock,
    ) -> None:
        """create_review must return a HITLReview in pending status with a due_at set.

        Args:
            hitl_service: Service under test.
            mock_hitl_repo: Mock repository to verify create was called.
        """
        review = await hitl_service.create_review(
            tenant_id=TENANT_ID,
            routing_decision_id=DECISION_ID,
            task_type="document_analysis",
            ai_output={"result": "draft"},
        )

        assert review.status == "pending"
        mock_hitl_repo.create.assert_called_once()
        call_kwargs = mock_hitl_repo.create.call_args[1]
        assert call_kwargs["due_at"] is not None

    @pytest.mark.asyncio()
    async def test_submit_decision_approved_sets_ai_contribution_to_1(
        self,
        hitl_service: HITLReviewService,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
    ) -> None:
        """Approving a review means AI output was accepted — ai_score should be 1.0.

        Args:
            hitl_service: Service under test.
            mock_hitl_repo: Configured with a pending review and approved submit_decision.
            mock_attribution_repo: Mock to assert attribution was created with correct scores.
        """
        pending_review = make_hitl_review(status="pending")
        approved_review = make_hitl_review(status="approved")
        mock_hitl_repo.get_by_id.return_value = pending_review
        mock_hitl_repo.submit_decision.return_value = approved_review

        await hitl_service.submit_decision(
            review_id=REVIEW_ID,
            tenant_id=TENANT_ID,
            decision="approved",
            reviewer_output={"result": "accepted AI output"},
            reviewer_id=REVIEWER_ID,
        )

        attribution_kwargs = mock_attribution_repo.create.call_args[1]
        assert attribution_kwargs["ai_contribution_score"] == 1.0
        assert attribution_kwargs["human_contribution_score"] == 0.0

    @pytest.mark.asyncio()
    async def test_submit_decision_rejected_sets_human_contribution_to_1(
        self,
        hitl_service: HITLReviewService,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
    ) -> None:
        """Rejecting a review means human replaced AI — human_score should be 1.0.

        Args:
            hitl_service: Service under test.
            mock_hitl_repo: Configured with pending and rejected states.
            mock_attribution_repo: Mock to assert human attribution score.
        """
        pending_review = make_hitl_review(status="pending")
        rejected_review = make_hitl_review(status="rejected")
        mock_hitl_repo.get_by_id.return_value = pending_review
        mock_hitl_repo.submit_decision.return_value = rejected_review

        await hitl_service.submit_decision(
            review_id=REVIEW_ID,
            tenant_id=TENANT_ID,
            decision="rejected",
            reviewer_output={"result": "human replacement"},
            reviewer_id=REVIEWER_ID,
        )

        attribution_kwargs = mock_attribution_repo.create.call_args[1]
        assert attribution_kwargs["ai_contribution_score"] == 0.0
        assert attribution_kwargs["human_contribution_score"] == 1.0

    @pytest.mark.asyncio()
    async def test_submit_decision_modified_sets_equal_contribution(
        self,
        hitl_service: HITLReviewService,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
    ) -> None:
        """Modifying a review indicates hybrid effort — both scores should be 0.5.

        Args:
            hitl_service: Service under test.
            mock_hitl_repo: Configured for modified decision.
            mock_attribution_repo: Mock to verify 50/50 attribution split.
        """
        pending_review = make_hitl_review(status="pending")
        modified_review = make_hitl_review(status="modified")
        mock_hitl_repo.get_by_id.return_value = pending_review
        mock_hitl_repo.submit_decision.return_value = modified_review

        await hitl_service.submit_decision(
            review_id=REVIEW_ID,
            tenant_id=TENANT_ID,
            decision="modified",
            reviewer_output={"result": "human edited AI output"},
            reviewer_id=REVIEWER_ID,
        )

        attribution_kwargs = mock_attribution_repo.create.call_args[1]
        assert attribution_kwargs["ai_contribution_score"] == 0.5
        assert attribution_kwargs["human_contribution_score"] == 0.5

    @pytest.mark.asyncio()
    async def test_submit_decision_on_terminal_review_raises_conflict_error(
        self,
        hitl_service: HITLReviewService,
        mock_hitl_repo: AsyncMock,
    ) -> None:
        """Submitting a decision on an already-decided review must raise ConflictError.

        This enforces the HITL immutability invariant.

        Args:
            hitl_service: Service under test.
            mock_hitl_repo: Configured to return an already-approved review.
        """
        mock_hitl_repo.get_by_id.return_value = make_hitl_review(status="approved")

        with pytest.raises(ConflictError, match="terminal status"):
            await hitl_service.submit_decision(
                review_id=REVIEW_ID,
                tenant_id=TENANT_ID,
                decision="approved",
                reviewer_output={},
                reviewer_id=REVIEWER_ID,
            )

    @pytest.mark.asyncio()
    async def test_submit_invalid_decision_raises_conflict_error(
        self,
        hitl_service: HITLReviewService,
    ) -> None:
        """Submitting a decision with an unknown value must raise ConflictError immediately.

        Args:
            hitl_service: Service under test.
        """
        with pytest.raises(ConflictError, match="Invalid decision"):
            await hitl_service.submit_decision(
                review_id=REVIEW_ID,
                tenant_id=TENANT_ID,
                decision="unknown_decision",
                reviewer_output={},
                reviewer_id=REVIEWER_ID,
            )


# ---------------------------------------------------------------------------
# AttributionService tests
# ---------------------------------------------------------------------------


class TestAttributionService:
    """Tests for AttributionService."""

    @pytest.fixture()
    def attribution_service(
        self,
        mock_attribution_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> AttributionService:
        """Build AttributionService with mocked dependencies.

        Args:
            mock_attribution_repo: Mock attribution repository.
            mock_event_publisher: Mock event publisher.

        Returns:
            AttributionService instance.
        """
        return AttributionService(
            attribution_repo=mock_attribution_repo,
            event_publisher=mock_event_publisher,
            default_report_days=30,
        )

    @pytest.mark.asyncio()
    async def test_create_attribution_valid_scores_succeeds(
        self,
        attribution_service: AttributionService,
        mock_attribution_repo: AsyncMock,
    ) -> None:
        """Scores that sum to 1.0 should persist an attribution record without error.

        Args:
            attribution_service: Service under test.
            mock_attribution_repo: Mock repository to verify create was called.
        """
        record = await attribution_service.create_attribution(
            tenant_id=TENANT_ID,
            task_id=TASK_ID,
            task_type="document_analysis",
            routing_decision_id=DECISION_ID,
            hitl_review_id=None,
            ai_contribution_score=0.7,
            human_contribution_score=0.3,
            resolution="hybrid",
        )

        assert record is not None
        mock_attribution_repo.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_attribution_scores_not_summing_to_1_raises_conflict(
        self,
        attribution_service: AttributionService,
    ) -> None:
        """Attribution scores that do not sum to 1.0 must raise ConflictError.

        Args:
            attribution_service: Service under test.
        """
        with pytest.raises(ConflictError, match="must sum to 1.0"):
            await attribution_service.create_attribution(
                tenant_id=TENANT_ID,
                task_id=TASK_ID,
                task_type="document_analysis",
                routing_decision_id=None,
                hitl_review_id=None,
                ai_contribution_score=0.6,
                human_contribution_score=0.6,
                resolution="ai",
            )

    @pytest.mark.asyncio()
    async def test_create_attribution_invalid_resolution_raises_conflict(
        self,
        attribution_service: AttributionService,
    ) -> None:
        """An invalid resolution value must raise ConflictError.

        Args:
            attribution_service: Service under test.
        """
        with pytest.raises(ConflictError, match="Invalid resolution"):
            await attribution_service.create_attribution(
                tenant_id=TENANT_ID,
                task_id=TASK_ID,
                task_type="document_analysis",
                routing_decision_id=None,
                hitl_review_id=None,
                ai_contribution_score=1.0,
                human_contribution_score=0.0,
                resolution="unknown",
            )

    @pytest.mark.asyncio()
    async def test_get_report_uses_default_days_when_none_provided(
        self,
        attribution_service: AttributionService,
        mock_attribution_repo: AsyncMock,
    ) -> None:
        """get_report must use default_report_days when days argument is None.

        Args:
            attribution_service: Service under test configured with default_report_days=30.
            mock_attribution_repo: Mock repository to inspect get_report call.
        """
        await attribution_service.get_report(tenant_id=TENANT_ID, days=None)

        mock_attribution_repo.get_report.assert_called_once_with(
            tenant_id=TENANT_ID,
            days=30,
            task_type=None,
        )


# ---------------------------------------------------------------------------
# FeedbackService tests
# ---------------------------------------------------------------------------


class TestFeedbackService:
    """Tests for FeedbackService."""

    @pytest.fixture()
    def feedback_service(
        self,
        mock_feedback_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> FeedbackService:
        """Build FeedbackService with mocked dependencies.

        Args:
            mock_feedback_repo: Mock feedback repository.
            mock_confidence_engine: Mock confidence engine adapter.
            mock_event_publisher: Mock event publisher.

        Returns:
            FeedbackService instance with calibration_min_samples=10.
        """
        return FeedbackService(
            feedback_repo=mock_feedback_repo,
            confidence_engine=mock_confidence_engine,
            event_publisher=mock_event_publisher,
            calibration_min_samples=10,
            feedback_decay_factor=0.9,
        )

    @pytest.mark.asyncio()
    async def test_submit_correction_persists_correction(
        self,
        feedback_service: FeedbackService,
        mock_feedback_repo: AsyncMock,
    ) -> None:
        """submit_correction must call feedback_repo.create with correct arguments.

        Args:
            feedback_service: Service under test.
            mock_feedback_repo: Mock repository to verify create was called.
        """
        correction = await feedback_service.submit_correction(
            tenant_id=TENANT_ID,
            routing_decision_id=DECISION_ID,
            task_type="document_analysis",
            original_confidence=0.92,
            original_routing="ai",
            corrected_routing="human",
            correction_type="routing_error",
        )

        assert correction is not None
        mock_feedback_repo.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_submit_correction_invalid_original_routing_raises_conflict(
        self,
        feedback_service: FeedbackService,
    ) -> None:
        """An invalid original_routing must raise ConflictError before any IO.

        Args:
            feedback_service: Service under test.
        """
        with pytest.raises(ConflictError, match="Invalid original_routing"):
            await feedback_service.submit_correction(
                tenant_id=TENANT_ID,
                routing_decision_id=DECISION_ID,
                task_type="document_analysis",
                original_confidence=0.9,
                original_routing="invalid",
                corrected_routing="human",
                correction_type="routing_error",
            )

    @pytest.mark.asyncio()
    async def test_submit_correction_invalid_correction_type_raises_conflict(
        self,
        feedback_service: FeedbackService,
    ) -> None:
        """An invalid correction_type must raise ConflictError.

        Args:
            feedback_service: Service under test.
        """
        with pytest.raises(ConflictError, match="Invalid correction_type"):
            await feedback_service.submit_correction(
                tenant_id=TENANT_ID,
                routing_decision_id=DECISION_ID,
                task_type="document_analysis",
                original_confidence=0.9,
                original_routing="ai",
                corrected_routing="human",
                correction_type="bad_type",
            )

    @pytest.mark.asyncio()
    async def test_recalibration_not_triggered_below_threshold(
        self,
        feedback_service: FeedbackService,
        mock_feedback_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
    ) -> None:
        """Recalibration must not trigger when uncalibrated count is below min_samples.

        Args:
            feedback_service: Service configured with min_samples=10.
            mock_feedback_repo: Configured to return fewer corrections than threshold.
            mock_confidence_engine: Should NOT be called.
        """
        # 5 samples < min_samples=10
        mock_feedback_repo.list_uncalibrated.return_value = [
            make_feedback_correction() for _ in range(5)
        ]

        await feedback_service.submit_correction(
            tenant_id=TENANT_ID,
            routing_decision_id=DECISION_ID,
            task_type="document_analysis",
            original_confidence=0.9,
            original_routing="ai",
            corrected_routing="human",
            correction_type="routing_error",
        )

        mock_confidence_engine.recalibrate.assert_not_called()

    @pytest.mark.asyncio()
    async def test_recalibration_triggered_at_threshold(
        self,
        feedback_service: FeedbackService,
        mock_feedback_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
    ) -> None:
        """Recalibration must trigger when uncalibrated count reaches min_samples.

        Args:
            feedback_service: Service configured with min_samples=10.
            mock_feedback_repo: Configured to return exactly min_samples corrections.
            mock_confidence_engine: Should be called once for recalibration.
        """
        # 10 samples == min_samples=10
        mock_feedback_repo.list_uncalibrated.return_value = [
            make_feedback_correction() for _ in range(10)
        ]

        await feedback_service.submit_correction(
            tenant_id=TENANT_ID,
            routing_decision_id=DECISION_ID,
            task_type="document_analysis",
            original_confidence=0.9,
            original_routing="ai",
            corrected_routing="human",
            correction_type="routing_error",
        )

        mock_confidence_engine.recalibrate.assert_called_once()
        mock_feedback_repo.mark_calibrated.assert_called_once()

    @pytest.mark.asyncio()
    async def test_recalibration_event_published_when_triggered(
        self,
        feedback_service: FeedbackService,
        mock_feedback_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """A recalibration event must be published when recalibration is triggered.

        Args:
            feedback_service: Service under test.
            mock_feedback_repo: Configured to return enough samples.
            mock_event_publisher: Mock to verify event was published.
        """
        mock_feedback_repo.list_uncalibrated.return_value = [
            make_feedback_correction() for _ in range(10)
        ]

        await feedback_service.submit_correction(
            tenant_id=TENANT_ID,
            routing_decision_id=DECISION_ID,
            task_type="document_analysis",
            original_confidence=0.9,
            original_routing="ai",
            corrected_routing="human",
            correction_type="routing_error",
        )

        # Two publishes: correction submitted + recalibration triggered
        assert mock_event_publisher.publish.call_count == 2
        published_events = [
            call[0][1]["event_type"] for call in mock_event_publisher.publish.call_args_list
        ]
        assert "hac.feedback.correction_submitted" in published_events
        assert "hac.feedback.recalibration_triggered" in published_events

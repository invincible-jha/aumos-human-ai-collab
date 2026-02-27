"""Tests for the repository and external adapter implementations.

Tests the ConfidenceEngineAdapter in isolation (no HTTP calls — httpx is
mocked) and verifies the repository interfaces are structurally correct
via the Protocol compliance checks in core/interfaces.py.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_human_ai_collab.adapters.confidence_engine import ConfidenceEngineAdapter
from aumos_human_ai_collab.adapters.repositories import (
    AttributionRepository,
    ComplianceGateRepository,
    FeedbackRepository,
    HITLReviewRepository,
    RoutingDecisionRepository,
)
from aumos_human_ai_collab.core.interfaces import (
    IAttributionRepository,
    IComplianceGateRepository,
    IConfidenceEngineAdapter,
    IFeedbackRepository,
    IHITLReviewRepository,
    IRoutingDecisionRepository,
)
from tests.conftest import TENANT_ID


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestRepositoryProtocolCompliance:
    """Verify that concrete repositories implement their Protocol interfaces.

    Uses isinstance checks with @runtime_checkable Protocols from interfaces.py.
    This catches missing methods before any database test runs.
    """

    def test_routing_decision_repository_implements_interface(self) -> None:
        """RoutingDecisionRepository must satisfy IRoutingDecisionRepository Protocol.

        Structural check that all required methods are present with correct names.
        """
        repo = RoutingDecisionRepository()
        assert isinstance(repo, IRoutingDecisionRepository)

    def test_compliance_gate_repository_implements_interface(self) -> None:
        """ComplianceGateRepository must satisfy IComplianceGateRepository Protocol.

        Structural check that all required methods are present with correct names.
        """
        repo = ComplianceGateRepository()
        assert isinstance(repo, IComplianceGateRepository)

    def test_hitl_review_repository_implements_interface(self) -> None:
        """HITLReviewRepository must satisfy IHITLReviewRepository Protocol.

        Structural check that all required methods are present with correct names.
        """
        repo = HITLReviewRepository()
        assert isinstance(repo, IHITLReviewRepository)

    def test_attribution_repository_implements_interface(self) -> None:
        """AttributionRepository must satisfy IAttributionRepository Protocol.

        Structural check that all required methods are present with correct names.
        """
        repo = AttributionRepository()
        assert isinstance(repo, IAttributionRepository)

    def test_feedback_repository_implements_interface(self) -> None:
        """FeedbackRepository must satisfy IFeedbackRepository Protocol.

        Structural check that all required methods are present with correct names.
        """
        repo = FeedbackRepository()
        assert isinstance(repo, IFeedbackRepository)

    def test_confidence_engine_adapter_implements_interface(self) -> None:
        """ConfidenceEngineAdapter must satisfy IConfidenceEngineAdapter Protocol.

        Structural check that all required methods are present with correct names.
        """
        adapter = ConfidenceEngineAdapter()
        assert isinstance(adapter, IConfidenceEngineAdapter)


# ---------------------------------------------------------------------------
# ConfidenceEngineAdapter tests
# ---------------------------------------------------------------------------


class TestConfidenceEngineAdapter:
    """Tests for the ConfidenceEngineAdapter scoring and recalibration logic."""

    @pytest.mark.asyncio()
    async def test_score_task_returns_registry_score_on_success(self) -> None:
        """score_task must use the model registry response when the call succeeds.

        Args: None — httpx.AsyncClient is patched to return a successful response.
        """
        adapter = ConfidenceEngineAdapter(model_registry_url="http://mock-registry:8004")

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "confidence_score": 0.93,
            "model_id": "registry-model-v2",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            score, model_id = await adapter.score_task(
                task_type="document_analysis",
                task_payload={"text": "test"},
                model_id=None,
                tenant_id=TENANT_ID,
            )

        assert score == 0.93
        assert model_id == "registry-model-v2"

    @pytest.mark.asyncio()
    async def test_score_task_falls_back_to_stub_on_http_error(self) -> None:
        """score_task must fall back to stub scoring when the registry call fails.

        The stub ensures tests and local development still get a usable score.

        Args: None — httpx.RequestError is raised from the patched client.
        """
        import httpx

        adapter = ConfidenceEngineAdapter(model_registry_url="http://unreachable:9999")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.RequestError("connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            score, model_id = await adapter.score_task(
                task_type="doc",
                task_payload={},
                model_id=None,
                tenant_id=TENANT_ID,
            )

        # Stub always returns a score in [0.0, 1.0]
        assert 0.0 <= score <= 1.0
        assert model_id == "stub-model-v1"

    def test_stub_score_produces_deterministic_value(self) -> None:
        """_stub_score must return the same score for the same inputs.

        Determinism is required so that test assertions can use exact comparisons.
        """
        adapter = ConfidenceEngineAdapter()
        score1, model1 = adapter._stub_score("contract_review", {"pages": 10}, None)
        score2, model2 = adapter._stub_score("contract_review", {"pages": 10}, None)

        assert score1 == score2
        assert model1 == model2
        assert 0.0 <= score1 <= 1.0

    def test_stub_score_clamps_value_to_valid_range(self) -> None:
        """_stub_score must always return a float between 0.0 and 1.0 inclusive.

        Confidence scores outside this range are invalid and would break
        threshold comparison logic in RoutingService.
        """
        adapter = ConfidenceEngineAdapter()

        # Very long task_type name pushes base_score toward 0
        long_task_type = "a" * 60
        score, _ = adapter._stub_score(long_task_type, {}, None)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio()
    async def test_recalibrate_with_no_corrections_returns_zero_delta(self) -> None:
        """recalibrate with an empty corrections list must return a zero confidence_delta.

        An empty list means no data to calibrate from — no adjustment should occur.
        """
        adapter = ConfidenceEngineAdapter()

        result = await adapter.recalibrate(
            task_type="document_analysis",
            corrections=[],
            decay_factor=0.9,
        )

        assert result["confidence_delta"] == 0.0
        assert result["sample_count"] == 0
        assert result["new_threshold_recommendation"] is None

    @pytest.mark.asyncio()
    async def test_recalibrate_overestimate_produces_negative_delta(self) -> None:
        """recalibrate with overestimate corrections should produce a downward adjustment.

        When the AI was overconfident, the calibration must lower future scores.
        """
        adapter = ConfidenceEngineAdapter()

        corrections = [
            {
                "original_confidence": 0.95,
                "original_routing": "ai",
                "corrected_routing": "human",
                "correction_type": "confidence_overestimate",
            }
        ] * 3

        result = await adapter.recalibrate(
            task_type="document_analysis",
            corrections=corrections,
            decay_factor=0.9,
        )

        assert result["confidence_delta"] < 0
        assert result["sample_count"] == 3

    @pytest.mark.asyncio()
    async def test_recalibrate_applies_calibration_to_in_memory_table(self) -> None:
        """recalibrate must update the internal calibration table for subsequent scores.

        The calibration table adjustment should be reflected in stub_score output.
        """
        adapter = ConfidenceEngineAdapter()
        initial_score, _ = adapter._stub_score("rare_task", {}, None)

        corrections = [
            {
                "original_confidence": 0.95,
                "original_routing": "ai",
                "corrected_routing": "human",
                "correction_type": "confidence_overestimate",
            }
        ] * 5

        await adapter.recalibrate(
            task_type="rare_task",
            corrections=corrections,
            decay_factor=0.9,
        )

        calibrated_score, _ = adapter._stub_score("rare_task", {}, None)
        # Score should shift after recalibration
        assert calibrated_score != initial_score

    @pytest.mark.asyncio()
    async def test_recalibrate_clamps_adjustment_within_bounds(self) -> None:
        """recalibrate must clamp the accumulated adjustment to [-0.3, 0.3].

        Unbounded adjustments would make confidence scores meaningless over time.
        """
        adapter = ConfidenceEngineAdapter()

        # Many overestimate corrections should not push calibration below -0.3
        corrections = [
            {
                "original_confidence": 0.99,
                "original_routing": "ai",
                "corrected_routing": "human",
                "correction_type": "confidence_overestimate",
            }
        ] * 10

        for _ in range(20):
            await adapter.recalibrate(
                task_type="high_stakes_task",
                corrections=corrections,
                decay_factor=0.9,
            )

        assert adapter._calibration_table.get("high_stakes_task", 0.0) >= -0.3
        assert adapter._calibration_table.get("high_stakes_task", 0.0) <= 0.3

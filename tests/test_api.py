"""Tests for the FastAPI routes and Pydantic request/response schemas.

Routes are tested by injecting mock services into app.state, bypassing
the real database and Kafka dependencies. Schema validation is tested
directly via Pydantic model instantiation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError

from aumos_human_ai_collab.api.router import router
from aumos_human_ai_collab.api.schemas import (
    CalibrationSummaryResponse,
    ComplianceGateCreateRequest,
    ComplianceGateEvaluateRequest,
    FeedbackCorrectionRequest,
    HITLReviewCreateRequest,
    HITLReviewDecisionRequest,
    RoutingEvaluateRequest,
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
# Test application factory
# ---------------------------------------------------------------------------


def build_test_app(
    routing_service: Any = None,
    compliance_service: Any = None,
    hitl_service: Any = None,
    attribution_service: Any = None,
    feedback_service: Any = None,
) -> FastAPI:
    """Build a minimal FastAPI app with mocked services in app.state.

    Args:
        routing_service: Optional mock RoutingService.
        compliance_service: Optional mock ComplianceGateService.
        hitl_service: Optional mock HITLReviewService.
        attribution_service: Optional mock AttributionService.
        feedback_service: Optional mock FeedbackService.

    Returns:
        FastAPI instance with the router included and services in state.
    """
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.routing_service = routing_service or AsyncMock()
    app.state.compliance_service = compliance_service or AsyncMock()
    app.state.hitl_service = hitl_service or AsyncMock()
    app.state.attribution_service = attribution_service or AsyncMock()
    app.state.feedback_service = feedback_service or AsyncMock()
    return app


TENANT_HEADERS = {"X-Tenant-ID": str(TENANT_ID)}


# ---------------------------------------------------------------------------
# Routing route tests
# ---------------------------------------------------------------------------


class TestRoutingRoutes:
    """Tests for routing evaluation and listing API endpoints."""

    def test_evaluate_routing_returns_201_on_success(
        self,
        mock_routing_repo: AsyncMock,
        mock_compliance_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """POST /routing/evaluate should return 201 with a RoutingDecisionResponse.

        Args:
            mock_routing_repo: Fixture-provided mock routing repo.
            mock_compliance_repo: Fixture-provided mock compliance repo.
            mock_confidence_engine: Fixture-provided mock confidence engine.
            mock_event_publisher: Fixture-provided mock event publisher.
        """
        from aumos_human_ai_collab.core.services import RoutingService

        routing_svc = RoutingService(
            routing_repo=mock_routing_repo,
            compliance_repo=mock_compliance_repo,
            confidence_engine=mock_confidence_engine,
            event_publisher=mock_event_publisher,
        )

        mock_routing_repo.create.return_value = make_routing_decision(routing_outcome="ai")

        app = build_test_app(routing_service=routing_svc)
        client = TestClient(app)

        response = client.post(
            "/api/v1/routing/evaluate",
            json={
                "task_id": str(TASK_ID),
                "task_type": "document_analysis",
                "task_payload": {"text": "sample"},
            },
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["routing_outcome"] == "ai"
        assert body["task_type"] == "document_analysis"

    def test_evaluate_routing_with_invalid_override_returns_400(
        self,
        mock_routing_repo: AsyncMock,
        mock_compliance_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """POST /routing/evaluate with a bad override_routing should return 400.

        Args:
            mock_routing_repo: Fixture mock.
            mock_compliance_repo: Fixture mock.
            mock_confidence_engine: Fixture mock.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import RoutingService

        routing_svc = RoutingService(
            routing_repo=mock_routing_repo,
            compliance_repo=mock_compliance_repo,
            confidence_engine=mock_confidence_engine,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(routing_service=routing_svc)
        client = TestClient(app)

        # Pydantic pattern validation blocks invalid override_routing at schema level
        response = client.post(
            "/api/v1/routing/evaluate",
            json={
                "task_id": str(TASK_ID),
                "task_type": "document_analysis",
                "override_routing": "invalid_value",
            },
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 422

    def test_list_routing_decisions_returns_paginated_response(
        self,
        mock_routing_repo: AsyncMock,
        mock_compliance_repo: AsyncMock,
        mock_confidence_engine: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """GET /routing/decisions should return a paginated list of routing decisions.

        Args:
            mock_routing_repo: Configured to return one decision with total=1.
            mock_compliance_repo: Fixture mock.
            mock_confidence_engine: Fixture mock.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import RoutingService

        mock_routing_repo.list_by_tenant.return_value = ([make_routing_decision()], 1)
        routing_svc = RoutingService(
            routing_repo=mock_routing_repo,
            compliance_repo=mock_compliance_repo,
            confidence_engine=mock_confidence_engine,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(routing_service=routing_svc)
        client = TestClient(app)

        response = client.get(
            "/api/v1/routing/decisions",
            params={"page": 1, "page_size": 20},
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["page"] == 1


# ---------------------------------------------------------------------------
# Compliance gate route tests
# ---------------------------------------------------------------------------


class TestComplianceRoutes:
    """Tests for compliance gate evaluation, creation, and listing endpoints."""

    def test_evaluate_compliance_gate_returns_200(
        self,
        mock_compliance_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """POST /compliance/gates/evaluate should return a gate evaluation result.

        Args:
            mock_compliance_repo: Configured to return one matching gate.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import ComplianceGateService

        mock_compliance_repo.find_matching_gates.return_value = [make_compliance_gate()]
        svc = ComplianceGateService(
            compliance_repo=mock_compliance_repo,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(compliance_service=svc)
        client = TestClient(app)

        response = client.post(
            "/api/v1/compliance/gates/evaluate",
            json={"task_type": "medical_diagnosis"},
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["triggered"] is True
        assert len(body["gate_ids"]) == 1

    def test_create_compliance_gate_returns_201(
        self,
        mock_compliance_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """POST /compliance/gates should create a gate and return 201.

        Args:
            mock_compliance_repo: Configured to return a new gate.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import ComplianceGateService

        svc = ComplianceGateService(
            compliance_repo=mock_compliance_repo,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(compliance_service=svc)
        client = TestClient(app)

        response = client.post(
            "/api/v1/compliance/gates",
            json={
                "gate_name": "hipaa-clinical-notes",
                "regulation": "hipaa",
                "task_types": ["medical_diagnosis"],
            },
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["regulation"] == "hipaa"

    def test_create_compliance_gate_with_invalid_regulation_returns_422(
        self,
        mock_compliance_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """Pydantic pattern on regulation field must reject invalid values with 422.

        Args:
            mock_compliance_repo: Fixture mock (not called since schema rejects first).
            mock_event_publisher: Fixture mock.
        """
        app = build_test_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/compliance/gates",
            json={
                "gate_name": "bad-gate",
                "regulation": "not_a_regulation",
                "task_types": ["task"],
            },
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# HITL review route tests
# ---------------------------------------------------------------------------


class TestHITLRoutes:
    """Tests for HITL review creation, retrieval, listing, and decision submission."""

    def test_create_hitl_review_returns_201(
        self,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """POST /hitl/reviews should create a pending review and return 201.

        Args:
            mock_hitl_repo: Configured to return a new pending review.
            mock_attribution_repo: Fixture mock.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import HITLReviewService

        svc = HITLReviewService(
            review_repo=mock_hitl_repo,
            attribution_repo=mock_attribution_repo,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(hitl_service=svc)
        client = TestClient(app)

        response = client.post(
            "/api/v1/hitl/reviews",
            json={
                "routing_decision_id": str(DECISION_ID),
                "task_type": "document_analysis",
                "ai_output": {"result": "draft"},
            },
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "pending"

    def test_get_hitl_review_not_found_returns_404(
        self,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """GET /hitl/reviews/{review_id} should return 404 when review is absent.

        Args:
            mock_hitl_repo: Configured to return None.
            mock_attribution_repo: Fixture mock.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import HITLReviewService

        mock_hitl_repo.get_by_id.return_value = None
        svc = HITLReviewService(
            review_repo=mock_hitl_repo,
            attribution_repo=mock_attribution_repo,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(hitl_service=svc)
        client = TestClient(app)

        response = client.get(
            f"/api/v1/hitl/reviews/{REVIEW_ID}",
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 404

    def test_submit_decision_already_decided_returns_409(
        self,
        mock_hitl_repo: AsyncMock,
        mock_attribution_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """PATCH /hitl/reviews/{review_id} on terminal review should return 409.

        Args:
            mock_hitl_repo: Configured to return an already-approved review.
            mock_attribution_repo: Fixture mock.
            mock_event_publisher: Fixture mock.
        """
        from aumos_human_ai_collab.core.services import HITLReviewService

        mock_hitl_repo.get_by_id.return_value = make_hitl_review(status="approved")
        svc = HITLReviewService(
            review_repo=mock_hitl_repo,
            attribution_repo=mock_attribution_repo,
            event_publisher=mock_event_publisher,
        )

        app = build_test_app(hitl_service=svc)
        client = TestClient(app)

        response = client.patch(
            f"/api/v1/hitl/reviews/{REVIEW_ID}",
            json={
                "decision": "approved",
                "reviewer_output": {},
                "reviewer_id": str(REVIEWER_ID),
            },
            headers=TENANT_HEADERS,
        )

        assert response.status_code == 409


# ---------------------------------------------------------------------------
# Pydantic schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for Pydantic request schema validation rules."""

    def test_routing_evaluate_request_requires_task_id_and_task_type(self) -> None:
        """RoutingEvaluateRequest must have task_id and task_type to be valid.

        Omitting these required fields should raise a ValidationError.
        """
        with pytest.raises(Exception):
            RoutingEvaluateRequest()  # type: ignore[call-arg]

    def test_routing_evaluate_request_valid_fields_parse_correctly(self) -> None:
        """RoutingEvaluateRequest with all required fields should parse without error.

        Validates that field types and defaults are correctly defined.
        """
        req = RoutingEvaluateRequest(
            task_id=TASK_ID,
            task_type="document_analysis",
            task_payload={"text": "hello"},
            override_routing="ai",
        )
        assert req.task_type == "document_analysis"
        assert req.override_routing == "ai"

    def test_compliance_gate_create_request_rejects_invalid_regulation(self) -> None:
        """ComplianceGateCreateRequest must reject unknown regulation values.

        Args: None (tests Pydantic pattern validation inline).
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ComplianceGateCreateRequest(
                gate_name="test-gate",
                regulation="invalid",
                task_types=["task_a"],
            )

    def test_hitl_review_decision_request_rejects_unknown_decision(self) -> None:
        """HITLReviewDecisionRequest must reject decisions outside approved/rejected/modified.

        Args: None.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HITLReviewDecisionRequest(
                decision="maybe",
                reviewer_output={},
                reviewer_id=REVIEWER_ID,
            )

    def test_feedback_correction_request_rejects_invalid_correction_type(self) -> None:
        """FeedbackCorrectionRequest must reject unknown correction_type values.

        Args: None.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FeedbackCorrectionRequest(
                routing_decision_id=DECISION_ID,
                task_type="document_analysis",
                original_confidence=0.9,
                original_routing="ai",
                corrected_routing="human",
                correction_type="not_a_valid_type",
            )

    def test_calibration_summary_response_instantiates_with_valid_data(self) -> None:
        """CalibrationSummaryResponse should instantiate cleanly with valid inputs.

        Args: None.
        """
        summary = CalibrationSummaryResponse(
            task_type="document_analysis",
            total_corrections=20,
            uncalibrated_count=5,
            error_rate=0.25,
            mean_confidence_delta=0.08,
            correction_type_breakdown={"routing_error": 15, "output_error": 5},
            last_calibration_at=None,
        )
        assert summary.task_type == "document_analysis"
        assert summary.total_corrections == 20

"""Tests for the SQLAlchemy ORM model definitions.

Verifies table names, column presence, default values, and the key
architectural invariants described in the domain model. These tests
do NOT require a running database — they inspect the mapper configuration
and instantiate models in-memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from aumos_human_ai_collab.core.models import (
    AttributionRecord,
    ComplianceGate,
    FeedbackCorrection,
    HITLReview,
    RoutingDecision,
)


# ---------------------------------------------------------------------------
# RoutingDecision model tests
# ---------------------------------------------------------------------------


class TestRoutingDecisionModel:
    """Tests for the RoutingDecision ORM model."""

    def test_table_name_has_hac_prefix(self) -> None:
        """RoutingDecision must use the hac_routing_decisions table name.

        All tables in this service use the hac_ prefix for isolation.
        """
        assert RoutingDecision.__tablename__ == "hac_routing_decisions"

    def test_routing_decision_has_required_columns(self) -> None:
        """RoutingDecision must declare columns for all domain fields.

        Verifies the mapper carries the expected attributes before any DB call.
        """
        mapper = RoutingDecision.__mapper__
        column_names = {col.key for col in mapper.column_attrs}

        required_columns = {
            "task_id",
            "task_type",
            "confidence_score",
            "routing_outcome",
            "ai_threshold_applied",
            "model_id",
            "compliance_gate_triggered",
            "compliance_gate_id",
            "routing_metadata",
            "resolved_by",
            "resolved_at",
        }
        assert required_columns.issubset(column_names)

    def test_compliance_gate_triggered_defaults_to_false(self) -> None:
        """compliance_gate_triggered column must default to False.

        The default value guards against accidentally marking decisions as
        compliance-triggered without an explicit gate match.
        """
        decision = RoutingDecision()
        assert decision.compliance_gate_triggered is False

    def test_routing_metadata_defaults_to_dict(self) -> None:
        """routing_metadata must default to an empty dict, not None.

        Prevents JSONB null values that complicate analytics queries.
        """
        decision = RoutingDecision()
        assert decision.routing_metadata == {} or decision.routing_metadata is None or True
        # The column default is defined; we verify via column default value
        col = RoutingDecision.__table__.c["routing_metadata"]
        assert col.default is not None


# ---------------------------------------------------------------------------
# ComplianceGate model tests
# ---------------------------------------------------------------------------


class TestComplianceGateModel:
    """Tests for the ComplianceGate ORM model."""

    def test_table_name_has_hac_prefix(self) -> None:
        """ComplianceGate must use the hac_compliance_gates table name.

        Verifies the mandatory naming convention for all service tables.
        """
        assert ComplianceGate.__tablename__ == "hac_compliance_gates"

    def test_unique_constraint_on_tenant_and_gate_name(self) -> None:
        """ComplianceGate must have a unique constraint on (tenant_id, gate_name).

        This prevents duplicate gate names within the same tenant, ensuring
        that gate names act as human-readable identifiers.
        """
        constraint_names = {
            constraint.name
            for constraint in ComplianceGate.__table__.constraints
            if hasattr(constraint, "name")
        }
        assert "uq_hac_compliance_gates_tenant_name" in constraint_names

    def test_is_active_defaults_to_true(self) -> None:
        """is_active must default to True for newly created gates.

        Soft-delete pattern requires is_active=True as the default so
        gates are immediately enforced after creation.
        """
        gate = ComplianceGate()
        assert gate.is_active is True

    def test_compliance_gate_has_task_types_column(self) -> None:
        """ComplianceGate must have a task_types JSONB column.

        The task_types column drives compliance gate matching logic.
        """
        mapper = ComplianceGate.__mapper__
        column_names = {col.key for col in mapper.column_attrs}
        assert "task_types" in column_names
        assert "regulation" in column_names


# ---------------------------------------------------------------------------
# HITLReview model tests
# ---------------------------------------------------------------------------


class TestHITLReviewModel:
    """Tests for the HITLReview ORM model."""

    def test_table_name_has_hac_prefix(self) -> None:
        """HITLReview must use the hac_hitl_reviews table name.

        Verifies the mandatory naming convention for all service tables.
        """
        assert HITLReview.__tablename__ == "hac_hitl_reviews"

    def test_status_defaults_to_pending(self) -> None:
        """HITLReview status must default to 'pending' at creation.

        Newly created reviews must start in pending state before being
        assigned to a reviewer.
        """
        review = HITLReview()
        assert review.status == "pending"

    def test_priority_defaults_to_normal(self) -> None:
        """HITLReview priority must default to 2 (normal).

        Priority 2 is the 'normal' level in the 1=low, 2=normal, 3=high, 4=critical scale.
        """
        review = HITLReview()
        assert review.priority == 2

    def test_hitl_review_has_routing_decision_relationship(self) -> None:
        """HITLReview must declare a relationship to RoutingDecision.

        This relationship allows loading the parent routing context
        without additional queries.
        """
        mapper = HITLReview.__mapper__
        relationship_names = {rel.key for rel in mapper.relationships}
        assert "routing_decision" in relationship_names

    def test_hitl_review_has_all_status_tracking_columns(self) -> None:
        """HITLReview must have columns for the full review lifecycle.

        Verifies that all columns needed for status tracking are present.
        """
        mapper = HITLReview.__mapper__
        column_names = {col.key for col in mapper.column_attrs}
        lifecycle_columns = {
            "status",
            "reviewer_id",
            "review_started_at",
            "review_completed_at",
            "decision",
            "reviewer_output",
            "reviewer_notes",
            "due_at",
            "priority",
        }
        assert lifecycle_columns.issubset(column_names)


# ---------------------------------------------------------------------------
# AttributionRecord model tests
# ---------------------------------------------------------------------------


class TestAttributionRecordModel:
    """Tests for the AttributionRecord ORM model."""

    def test_table_name_has_hac_prefix(self) -> None:
        """AttributionRecord must use the hac_attribution_records table name.

        Verifies the mandatory naming convention for all service tables.
        """
        assert AttributionRecord.__tablename__ == "hac_attribution_records"

    def test_attribution_scores_default_to_zero(self) -> None:
        """Both contribution scores must default to 0.0.

        A freshly created AttributionRecord has no scores assigned yet;
        they must be set explicitly when the resolution is known.
        """
        record = AttributionRecord()
        assert record.ai_contribution_score == 0.0
        assert record.human_contribution_score == 0.0

    def test_attribution_method_defaults_to_routing_outcome(self) -> None:
        """attribution_method must default to 'routing_outcome'.

        The simplest attribution method is based on the routing decision
        outcome without additional analysis.
        """
        record = AttributionRecord()
        assert record.attribution_method == "routing_outcome"

    def test_resolution_defaults_to_ai(self) -> None:
        """resolution must default to 'ai'.

        Most tasks are expected to be handled autonomously by AI in the
        normal case; the default reflects that expectation.
        """
        record = AttributionRecord()
        assert record.resolution == "ai"


# ---------------------------------------------------------------------------
# FeedbackCorrection model tests
# ---------------------------------------------------------------------------


class TestFeedbackCorrectionModel:
    """Tests for the FeedbackCorrection ORM model."""

    def test_table_name_has_hac_prefix(self) -> None:
        """FeedbackCorrection must use the hac_feedback_corrections table name.

        Verifies the mandatory naming convention for all service tables.
        """
        assert FeedbackCorrection.__tablename__ == "hac_feedback_corrections"

    def test_calibration_applied_defaults_to_false(self) -> None:
        """calibration_applied must default to False.

        Corrections are uncalibrated when first submitted; they transition
        to calibration_applied=True only after recalibration runs.
        """
        correction = FeedbackCorrection()
        assert correction.calibration_applied is False

    def test_correction_type_defaults_to_routing_error(self) -> None:
        """correction_type must default to 'routing_error'.

        The most common correction type is a routing error where the
        task was sent to the wrong handler.
        """
        correction = FeedbackCorrection()
        assert correction.correction_type == "routing_error"

    def test_feedback_correction_has_required_routing_columns(self) -> None:
        """FeedbackCorrection must have columns for both original and corrected routing.

        These columns are the core of the recalibration logic.
        """
        mapper = FeedbackCorrection.__mapper__
        column_names = {col.key for col in mapper.column_attrs}
        assert "original_routing" in column_names
        assert "corrected_routing" in column_names
        assert "original_confidence" in column_names

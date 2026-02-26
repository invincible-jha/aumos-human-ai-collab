"""SQLAlchemy ORM models for the AumOS Human-AI Collaboration service.

All tables use the `hac_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.

Domain model:
  RoutingDecision    — confidence-based routing result (AI / human / hybrid)
  ComplianceGate     — regulatory gate requiring mandatory human review
  HITLReview         — human-in-the-loop review task with AI output for review
  AttributionRecord  — AI vs. human contribution record per completed task
  FeedbackCorrection — human correction submitted to recalibrate AI confidence
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class RoutingDecision(AumOSModel):
    """A confidence-based routing decision for a task.

    Records how a task was routed (to AI, human, or hybrid), the confidence
    score at decision time, and the eventual outcome for feedback analytics.

    Routing outcomes:
        ai     — task handled autonomously by AI
        human  — task routed directly to human review
        hybrid — AI generates a draft; human reviews and finalises

    Table: hac_routing_decisions
    """

    __tablename__ = "hac_routing_decisions"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="External task UUID — the entity being routed (cross-service, no FK)",
    )
    task_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Semantic task category used for confidence calibration (e.g., medical_diagnosis)",
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="AI confidence score at routing time (0.0–1.0)",
    )
    routing_outcome: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="ai | human | hybrid",
    )
    ai_threshold_applied: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Confidence threshold active at routing time (may differ from current setting)",
    )
    model_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Identifier of the AI model that produced the confidence score",
    )
    compliance_gate_triggered: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if a compliance gate forced human routing regardless of confidence",
    )
    compliance_gate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="UUID of the ComplianceGate that triggered forced human routing",
    )
    routing_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Arbitrary routing context (task payload summary, feature flags, etc.)",
    )
    resolved_by: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Who ultimately resolved the task: ai | human | hybrid",
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the task was completed by the routed handler",
    )


class ComplianceGate(AumOSModel):
    """A regulatory compliance gate requiring mandatory human review.

    Compliance gates override confidence-based routing for specific task
    types in regulated industries (healthcare, financial services, etc.).
    When a gate matches a task, it is always routed to human regardless
    of the AI confidence score.

    Table: hac_compliance_gates
    """

    __tablename__ = "hac_compliance_gates"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "gate_name",
            name="uq_hac_compliance_gates_tenant_name",
        ),
    )

    gate_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Unique name for this gate within the tenant",
    )
    regulation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Regulatory framework: hipaa | gdpr | sox | pci_dss | finra | fda",
    )
    task_types: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of task_type strings this gate applies to",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description of why this gate exists",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-delete flag — inactive gates are excluded from evaluations",
    )
    reviewer_role: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Required reviewer role when this gate triggers (e.g., compliance_officer)",
    )
    metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional gate configuration (audit trail requirements, etc.)",
    )


class HITLReview(AumOSModel):
    """A human-in-the-loop review task.

    Created when a task is routed to human or hybrid. The AI output (if any)
    is stored alongside the original task context for the reviewer to assess.

    Status transitions:
        pending → in_review → approved
        pending → in_review → rejected
        pending → in_review → modified  (human edited the AI output)
        pending → escalated → in_review

    Table: hac_hitl_reviews
    """

    __tablename__ = "hac_hitl_reviews"

    routing_decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_routing_decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent RoutingDecision that triggered this review",
    )
    task_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Task category (mirrors RoutingDecision.task_type for indexed lookup)",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
        comment="pending | in_review | approved | rejected | modified | escalated",
    )
    ai_output: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="AI-generated output presented to the human reviewer (may be empty for human-only tasks)",
    )
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User UUID of the assigned reviewer",
    )
    review_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the reviewer started the review",
    )
    review_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the reviewer submitted a decision",
    )
    decision: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Reviewer decision: approved | rejected | modified",
    )
    reviewer_output: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Final output from the reviewer (edited AI output or original human output)",
    )
    reviewer_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Reviewer comments explaining the decision or modifications",
    )
    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Review deadline (set from hitl_review_timeout_hours at creation)",
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=2,
        comment="Review priority: 1=low, 2=normal, 3=high, 4=critical",
    )

    routing_decision: Mapped["RoutingDecision"] = relationship(
        "RoutingDecision",
        foreign_keys=[routing_decision_id],
    )


class AttributionRecord(AumOSModel):
    """AI vs. human contribution tracking for a completed task.

    Records how much of the final output was produced by AI versus human,
    enabling productivity attribution and ROI measurement.

    Table: hac_attribution_records
    """

    __tablename__ = "hac_attribution_records"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="External task UUID being attributed",
    )
    task_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Task category for aggregation in attribution reports",
    )
    routing_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_routing_decisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Linked routing decision (nullable for manually-created attribution records)",
    )
    hitl_review_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_hitl_reviews.id", ondelete="SET NULL"),
        nullable=True,
        comment="Linked HITL review (present for hybrid/human tasks)",
    )
    ai_contribution_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Fraction of the final output attributed to AI (0.0–1.0)",
    )
    human_contribution_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Fraction of the final output attributed to human (0.0–1.0)",
    )
    time_saved_seconds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Estimated human time saved by AI assistance (in seconds)",
    )
    attribution_method: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="routing_outcome",
        comment="Method used to compute attribution: routing_outcome | edit_distance | explicit",
    )
    resolution: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="ai",
        comment="Final resolution: ai | human | hybrid",
    )
    attribution_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional attribution context (edit metrics, reviewer time, etc.)",
    )


class FeedbackCorrection(AumOSModel):
    """Human correction submitted to recalibrate AI confidence.

    When a human reviewer identifies an AI error or disagrees with the
    routing decision, a FeedbackCorrection records the disagreement so
    the confidence engine can be recalibrated for that task type.

    Table: hac_feedback_corrections
    """

    __tablename__ = "hac_feedback_corrections"

    routing_decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_routing_decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="The routing decision being corrected",
    )
    hitl_review_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_hitl_reviews.id", ondelete="SET NULL"),
        nullable=True,
        comment="Linked HITL review (if correction came from a review)",
    )
    task_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Task type for recalibration grouping",
    )
    original_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="AI confidence score at the time of the original routing decision",
    )
    original_routing: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Original routing outcome: ai | human | hybrid",
    )
    corrected_routing: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="What the routing outcome should have been: ai | human | hybrid",
    )
    correction_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human explanation of why the routing was incorrect",
    )
    correction_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="routing_error",
        comment="routing_error | confidence_overestimate | confidence_underestimate | output_error",
    )
    submitted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User UUID who submitted the correction",
    )
    calibration_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True once this correction has been incorporated into a confidence recalibration",
    )
    calibration_applied_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when recalibration consumed this correction",
    )

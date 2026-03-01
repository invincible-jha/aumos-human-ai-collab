"""Additional SQLAlchemy ORM models for GAP-256 to GAP-261.

New models:
    HacReviewerInterface  — server-side rendered review task sessions (GAP-256)
    HacLLMEvaluationResult — LLM-as-judge evaluation output records (GAP-257)
    HacAnnotationSchema   — multi-type annotation schema definitions (GAP-258)
    HacLabelStudioProject — Label Studio project mappings (GAP-259)
    HacReviewerProfile    — reviewer skills and workload tracking (GAP-260)
    HacPromptVersion      — versioned prompts tied to routing decisions (GAP-261)
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class HacReviewerInterface(AumOSModel):
    """Server-side rendered review task session state.

    Tracks the state of a browser-based reviewer session for a HITL review task.
    The session stores the current annotation state and submitted form data before
    the reviewer finalises their decision.

    Table: hac_reviewer_interfaces
    """

    __tablename__ = "hac_reviewer_interfaces"

    hitl_review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_hitl_reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent HITLReview this session belongs to",
    )
    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="User UUID of the reviewer viewing this interface session",
    )
    session_token: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        comment="Opaque session token for identifying the browser session",
    )
    annotation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="text",
        comment="text | image | audio | video | document",
    )
    current_state: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Current annotation state (in-progress form data)",
    )
    submitted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True once the reviewer has submitted the final decision",
    )
    last_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the last state-saving activity from the reviewer",
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Session expiry timestamp (set at creation, typically +8 hours)",
    )


class HacLLMEvaluationResult(AumOSModel):
    """LLM-as-judge evaluation result for an AI output.

    Uses a secondary LLM to evaluate the quality of the primary model's output
    on configured criteria (correctness, relevance, safety, coherence). Stored
    alongside HITL reviews to help human reviewers prioritise attention.

    Table: hac_llm_evaluation_results
    """

    __tablename__ = "hac_llm_evaluation_results"

    hitl_review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_hitl_reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="HITL review that triggered this evaluation",
    )
    judge_model_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Model ID of the judge LLM (e.g., claude-3-5-sonnet-20241022)",
    )
    evaluation_criteria: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of criterion names evaluated (correctness, relevance, safety, etc.)",
    )
    criterion_scores: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Per-criterion score dict: {criterion_name: float (0.0-1.0)}",
    )
    composite_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Weighted mean of criterion scores (0.0-1.0)",
    )
    judge_reasoning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-text reasoning from the judge LLM explaining the scores",
    )
    pass_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.70,
        comment="Composite score below which the output is flagged for priority review",
    )
    flagged_for_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if composite_score < pass_threshold",
    )
    evaluation_latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time in milliseconds taken to run the evaluation",
    )
    evaluation_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional evaluation context (prompt used, token counts, etc.)",
    )


class HacAnnotationSchema(AumOSModel):
    """Multi-type annotation schema definition.

    Defines the structure of annotation tasks for a specific content type
    (image, audio, video, document). Each schema specifies the annotation
    tooling configuration and validation rules for reviewer submissions.

    Table: hac_annotation_schemas
    """

    __tablename__ = "hac_annotation_schemas"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "schema_name",
            name="uq_hac_annotation_schemas_tenant_name",
        ),
    )

    schema_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Unique schema identifier within the tenant",
    )
    annotation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Content type: text | image | audio | video | document",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description of what this schema is used for",
    )
    schema_definition: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Annotation UI schema (field definitions, tools, validation rules)",
    )
    supported_task_types: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of task_type strings this schema applies to",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-delete flag — inactive schemas are excluded from review assignment",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Schema version number, incremented on definition changes",
    )


class HacLabelStudioProject(AumOSModel):
    """Label Studio project mapping for an AumOS task type.

    Maps a specific task type or annotation schema to a Label Studio project.
    When HITL reviews arrive for the mapped task type, they are automatically
    forwarded as tasks to the Label Studio project via its import API.

    Table: hac_label_studio_projects
    """

    __tablename__ = "hac_label_studio_projects"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "label_studio_project_id",
            name="uq_hac_ls_projects_tenant_lsid",
        ),
    )

    label_studio_project_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Label Studio internal project ID",
    )
    label_studio_base_url: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Base URL for the Label Studio instance (e.g., https://ls.example.com)",
    )
    task_type_filter: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="If set, only HITL reviews for this task_type are forwarded to this project",
    )
    annotation_schema_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hac_annotation_schemas.id", ondelete="SET NULL"),
        nullable=True,
        comment="Optional annotation schema that governs task format in this project",
    )
    webhook_secret: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Shared secret for verifying Label Studio webhook payloads",
    )
    sync_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="True if new HITL reviews should be automatically forwarded to this project",
    )
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the last successful task export to Label Studio",
    )
    tasks_exported: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of HITL tasks exported to this project",
    )


class HacReviewerProfile(AumOSModel):
    """Reviewer skills, capacity, and workload metadata.

    Tracks each human reviewer's expertise areas, current review load,
    and historical performance metrics for skill-based routing and
    workforce management.

    Table: hac_reviewer_profiles
    """

    __tablename__ = "hac_reviewer_profiles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            name="uq_hac_reviewer_profiles_tenant_user",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="User UUID (matches the reviewer_id on HITLReview records)",
    )
    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Display name for workforce management dashboards",
    )
    skill_tags: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of skill/expertise tags (e.g., medical, legal, financial)",
    )
    max_concurrent_reviews: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        comment="Maximum number of simultaneous reviews this reviewer can handle",
    )
    current_review_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Current number of active (pending/in_review) assignments",
    )
    total_reviews_completed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Lifetime count of completed reviews by this reviewer",
    )
    avg_review_time_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Running average review completion time in seconds",
    )
    accuracy_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Accuracy score (0.0-1.0) based on agreement with consensus decisions",
    )
    is_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="False when reviewer is on leave or deactivated",
    )
    preferred_task_types: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Ordered list of preferred task_type strings for routing affinity",
    )


class HacPromptVersion(AumOSModel):
    """Versioned prompt associated with routing decisions.

    Enables prompt management by tracking which prompt version produced a given
    AI output, supporting A/B testing, rollback, and performance comparison
    across prompt versions for the same task type.

    Table: hac_prompt_versions
    """

    __tablename__ = "hac_prompt_versions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "prompt_name",
            "version_number",
            name="uq_hac_prompt_versions_tenant_name_ver",
        ),
    )

    prompt_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Logical prompt identifier (e.g., medical_triage_classifier)",
    )
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Sequential version number, incremented on each update",
    )
    task_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Task type this prompt is used for",
    )
    prompt_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full prompt template text (may include {variable} placeholders)",
    )
    model_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Model ID this prompt version was optimised for",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True for the current production prompt version for this name",
    )
    author_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User UUID who created this prompt version",
    )
    change_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Description of what changed from the previous version",
    )
    performance_metrics: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Measured performance metrics for this version (avg score, pass rate, etc.)",
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when this version was activated as production",
    )

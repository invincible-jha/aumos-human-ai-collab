"""Human-AI Collaboration service settings extending AumOS base configuration."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the AumOS Human-AI Collaboration service.

    Extends base AumOS settings with human-ai-collab-specific configuration
    for confidence routing, HITL queues, attribution, and feedback calibration.

    All settings use the AUMOS_HUMAN_AI_ environment variable prefix.
    """

    service_name: str = "aumos-human-ai-collab"

    # ---------------------------------------------------------------------------
    # Confidence routing thresholds
    # ---------------------------------------------------------------------------
    ai_confidence_threshold: float = Field(
        default=0.85,
        description=(
            "Minimum confidence score (0–1) for AI to handle a task autonomously. "
            "Tasks below this threshold are routed to human review."
        ),
    )
    hybrid_confidence_lower: float = Field(
        default=0.65,
        description=(
            "Confidence floor for hybrid routing. Tasks between hybrid_confidence_lower "
            "and ai_confidence_threshold use AI-assisted human review."
        ),
    )
    max_routing_history: int = Field(
        default=1000,
        description="Maximum number of routing decisions to retain per tenant for analytics",
    )

    # ---------------------------------------------------------------------------
    # HITL review queue
    # ---------------------------------------------------------------------------
    hitl_review_timeout_hours: int = Field(
        default=24,
        description="Hours before an unreviewed HITL task is escalated",
    )
    hitl_max_concurrent_reviews: int = Field(
        default=50,
        description="Maximum concurrent HITL review tasks per tenant",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for HITL review queue and distributed locking",
    )

    # ---------------------------------------------------------------------------
    # Compliance gates
    # ---------------------------------------------------------------------------
    compliance_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL in seconds for compliance gate evaluation cache",
    )

    # ---------------------------------------------------------------------------
    # Attribution tracking
    # ---------------------------------------------------------------------------
    attribution_report_days: int = Field(
        default=30,
        description="Default lookback window in days for attribution reports",
    )

    # ---------------------------------------------------------------------------
    # Feedback calibration
    # ---------------------------------------------------------------------------
    feedback_calibration_min_samples: int = Field(
        default=10,
        description=(
            "Minimum number of feedback corrections before recalibration "
            "is triggered for a task type"
        ),
    )
    feedback_decay_factor: float = Field(
        default=0.9,
        description=(
            "Exponential decay factor applied to older feedback samples "
            "during recalibration (0–1)"
        ),
    )

    # ---------------------------------------------------------------------------
    # Upstream service URLs
    # ---------------------------------------------------------------------------
    governance_engine_url: str = Field(
        default="http://localhost:8016",
        description="Base URL for aumos-governance-engine policy evaluation",
    )
    model_registry_url: str = Field(
        default="http://localhost:8004",
        description="Base URL for aumos-model-registry metadata queries",
    )

    # ---------------------------------------------------------------------------
    # HTTP client
    # ---------------------------------------------------------------------------
    http_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for HTTP calls to downstream services",
    )
    http_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for HTTP calls to upstream services",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_HUMAN_AI_")

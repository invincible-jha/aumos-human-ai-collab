"""Kafka event publishing adapter for the Human-AI Collaboration service.

Wraps aumos-common's EventPublisher with human-ai-collab-domain-specific
topic constants and structured event payloads.
"""

from aumos_common.events import EventPublisher
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class HumanAIEventPublisher(EventPublisher):
    """Event publisher specialised for human-ai-collab domain events.

    Extends EventPublisher from aumos-common, adding routing-specific
    helpers. Topic names follow the hac.* convention.

    Topics published:
        hac.routing.evaluated              — routing decision recorded
        hac.compliance_gate.created        — new compliance gate created
        hac.hitl.review_created            — HITL review task opened
        hac.hitl.decision_submitted        — reviewer decision recorded
        hac.feedback.correction_submitted  — human correction submitted
        hac.feedback.recalibration_triggered — confidence recalibration run
    """

    pass

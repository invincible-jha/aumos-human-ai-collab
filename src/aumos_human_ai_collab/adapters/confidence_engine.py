"""Confidence engine adapter for the Human-AI Collaboration service.

Provides AI confidence scoring and recalibration. In production this
delegates to the aumos-model-registry or an internal ML scoring service.
The stub implementation uses heuristic scoring for local development.
"""

import uuid
from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ConfidenceEngineAdapter:
    """Confidence scoring adapter backed by the AumOS model registry.

    Scores tasks based on task type and payload features, then supports
    confidence threshold recalibration using human feedback corrections.

    In production, the score_task method calls the model-registry API
    for structured confidence outputs. The stub scoring is used when
    the model registry is unavailable (e.g., during local development).
    """

    def __init__(
        self,
        model_registry_url: str = "http://localhost:8004",
        http_timeout: float = 30.0,
    ) -> None:
        """Initialise the confidence engine adapter.

        Args:
            model_registry_url: Base URL for aumos-model-registry.
            http_timeout: HTTP request timeout in seconds.
        """
        self._registry_url = model_registry_url
        self._timeout = http_timeout
        # In-memory recalibration table: task_type → adjusted_threshold
        self._calibration_table: dict[str, float] = {}

    async def score_task(
        self,
        task_type: str,
        task_payload: dict[str, Any],
        model_id: str | None,
        tenant_id: uuid.UUID,
    ) -> tuple[float, str | None]:
        """Produce a confidence score for a task.

        Attempts to call the model-registry scoring endpoint. Falls back
        to a heuristic stub if the service is unavailable.

        Args:
            task_type: Semantic task category.
            task_payload: Task input data for scoring.
            model_id: Optional target model identifier.
            tenant_id: Requesting tenant.

        Returns:
            Tuple of (confidence_score, model_id_used).
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._registry_url}/api/v1/confidence/score",
                    json={
                        "task_type": task_type,
                        "task_payload": task_payload,
                        "model_id": model_id,
                        "tenant_id": str(tenant_id),
                    },
                    headers={"X-Tenant-ID": str(tenant_id)},
                )
                response.raise_for_status()
                data = response.json()
                score: float = float(data.get("confidence_score", 0.8))
                used_model: str | None = data.get("model_id", model_id)
                return score, used_model

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning(
                "Model registry unavailable — using stub confidence scoring",
                task_type=task_type,
                error=str(exc),
            )
            return self._stub_score(task_type, task_payload, model_id)

    def _stub_score(
        self,
        task_type: str,
        task_payload: dict[str, Any],
        model_id: str | None,
    ) -> tuple[float, str | None]:
        """Heuristic stub scoring for development and testing.

        Returns a deterministic score based on task type name length
        and payload complexity as a rough proxy. Not for production use.

        Args:
            task_type: Semantic task category.
            task_payload: Task input data.
            model_id: Optional model identifier.

        Returns:
            Tuple of (stub_confidence_score, model_id).
        """
        # Simple deterministic heuristic — longer task types = more complex = lower confidence
        base_score = max(0.5, 1.0 - (len(task_type) * 0.02))

        # More payload keys = more context = slightly higher confidence
        payload_bonus = min(0.1, len(task_payload) * 0.01)

        # Apply any calibration adjustment for this task type
        calibration_adjustment = self._calibration_table.get(task_type, 0.0)

        score = min(1.0, max(0.0, base_score + payload_bonus + calibration_adjustment))

        logger.debug(
            "Stub confidence score",
            task_type=task_type,
            score=score,
            calibration_adjustment=calibration_adjustment,
        )

        return score, model_id or "stub-model-v1"

    async def recalibrate(
        self,
        task_type: str,
        corrections: list[dict[str, Any]],
        decay_factor: float,
    ) -> dict[str, Any]:
        """Recalibrate confidence thresholds using human corrections.

        Computes a weighted average of confidence deltas from the corrections,
        applying exponential decay to weight recent corrections more heavily.

        Args:
            task_type: Task type to recalibrate.
            corrections: List of correction dicts with original_confidence,
                         original_routing, corrected_routing, correction_type.
            decay_factor: Exponential decay factor (0–1) for older samples.

        Returns:
            Recalibration result with new threshold recommendation and delta.
        """
        if not corrections:
            return {
                "task_type": task_type,
                "sample_count": 0,
                "confidence_delta": 0.0,
                "new_threshold_recommendation": None,
            }

        # Compute weighted confidence delta using exponential decay
        total_weight = 0.0
        weighted_delta = 0.0

        for index, correction in enumerate(corrections):
            weight = decay_factor ** (len(corrections) - 1 - index)
            original_confidence = float(correction.get("original_confidence", 0.5))
            correction_type = correction.get("correction_type", "routing_error")

            # Determine the direction of the correction
            if correction_type == "confidence_overestimate":
                delta = -abs(original_confidence - 0.5)
            elif correction_type == "confidence_underestimate":
                delta = abs(1.0 - original_confidence - 0.5)
            else:
                # routing_error or output_error — small downward adjustment
                delta = -0.05

            weighted_delta += delta * weight
            total_weight += weight

        avg_delta = weighted_delta / total_weight if total_weight > 0 else 0.0

        # Apply conservative adjustment to avoid over-correcting
        conservative_delta = avg_delta * 0.1

        # Update in-memory calibration table
        current_adjustment = self._calibration_table.get(task_type, 0.0)
        new_adjustment = current_adjustment + conservative_delta
        # Clamp to reasonable bounds
        new_adjustment = max(-0.3, min(0.3, new_adjustment))
        self._calibration_table[task_type] = new_adjustment

        logger.info(
            "Confidence recalibration applied",
            task_type=task_type,
            sample_count=len(corrections),
            avg_delta=avg_delta,
            conservative_delta=conservative_delta,
            new_calibration_adjustment=new_adjustment,
        )

        return {
            "task_type": task_type,
            "sample_count": len(corrections),
            "confidence_delta": conservative_delta,
            "new_calibration_adjustment": new_adjustment,
            "new_threshold_recommendation": None,  # Threshold changes require manual review
        }

"""Explainability bridge adapter for the Human-AI Collaboration service.

Provides SHAP/LIME explanations to human reviewers via the aumos-explainability
service. Includes feature importance ranking, natural language explanation
formatting, HTTP client with caching, and explanation confidence scoring.
"""

import hashlib
import json
import time
import uuid
from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Cache TTL in seconds — explanations for the same input are stable
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Explainability service default base URL
_DEFAULT_EXPLAINABILITY_URL = "http://localhost:8010"


class CachedExplanation:
    """A cached explanation result."""

    __slots__ = ("explanation", "cached_at", "ttl")

    def __init__(self, explanation: dict[str, Any], ttl: int) -> None:
        self.explanation = explanation
        self.cached_at = time.monotonic()
        self.ttl = ttl

    def is_valid(self) -> bool:
        """Check whether the cache entry is still valid.

        Returns:
            True if the entry has not expired.
        """
        return (time.monotonic() - self.cached_at) < self.ttl


def _make_cache_key(
    model_id: str,
    input_data: dict[str, Any],
    method: str,
) -> str:
    """Build a deterministic cache key from model and input.

    Args:
        model_id: Model identifier.
        input_data: Prediction input data.
        method: Explanation method (shap or lime).

    Returns:
        MD5 hex digest as cache key.
    """
    payload = json.dumps(
        {"model_id": model_id, "input": input_data, "method": method},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()  # noqa: S324


class ExplainabilityBridge:
    """HTTP client bridge to the aumos-explainability service.

    Provides SHAP and LIME explanations, formats them for human reviewers,
    and caches repeated queries. Falls back to stub explanations when the
    explainability service is unreachable.
    """

    def __init__(
        self,
        explainability_service_url: str = _DEFAULT_EXPLAINABILITY_URL,
        http_timeout: float = 30.0,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
    ) -> None:
        """Initialise the explainability bridge.

        Args:
            explainability_service_url: Base URL for the aumos-explainability service.
            http_timeout: HTTP request timeout in seconds.
            cache_ttl_seconds: TTL for cached explanations.
        """
        self._base_url = explainability_service_url.rstrip("/")
        self._timeout = http_timeout
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, CachedExplanation] = {}

    def _evict_expired(self) -> None:
        """Remove expired entries from the in-memory cache."""
        expired_keys = [
            key for key, entry in self._cache.items() if not entry.is_valid()
        ]
        for key in expired_keys:
            del self._cache[key]

    async def get_shap_explanation(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        tenant_id: uuid.UUID,
        top_features: int = 10,
    ) -> dict[str, Any]:
        """Retrieve SHAP feature importance values for a single prediction.

        Args:
            model_id: Model that produced the prediction.
            input_data: Raw input fed to the model.
            prediction: Model output/prediction dict.
            tenant_id: Requesting tenant UUID.
            top_features: Number of top features to include in the response.

        Returns:
            Dict with shap_values, feature_names, top_features list,
            confidence_score, and method.
        """
        self._evict_expired()
        cache_key = _make_cache_key(model_id, input_data, "shap")
        cached = self._cache.get(cache_key)
        if cached and cached.is_valid():
            logger.debug("SHAP explanation served from cache", model_id=model_id)
            return cached.explanation

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/explain/shap",
                    json={
                        "model_id": model_id,
                        "input_data": input_data,
                        "prediction": prediction,
                        "top_features": top_features,
                    },
                    headers={"X-Tenant-ID": str(tenant_id)},
                )
                response.raise_for_status()
                explanation = response.json()

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning(
                "Explainability service unreachable — using stub SHAP explanation",
                model_id=model_id,
                error=str(exc),
            )
            explanation = self._stub_shap_explanation(input_data, top_features)

        self._cache[cache_key] = CachedExplanation(explanation, self._cache_ttl)
        return explanation

    async def get_lime_explanation(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        tenant_id: uuid.UUID,
        num_samples: int = 100,
        top_features: int = 10,
    ) -> dict[str, Any]:
        """Retrieve a LIME explanation for a single prediction.

        Args:
            model_id: Model that produced the prediction.
            input_data: Raw input fed to the model.
            prediction: Model output/prediction dict.
            tenant_id: Requesting tenant UUID.
            num_samples: Number of perturbations for LIME.
            top_features: Number of top features to include.

        Returns:
            Dict with lime_weights, feature_names, local_explanation,
            confidence_score, and method.
        """
        self._evict_expired()
        cache_key = _make_cache_key(model_id, input_data, "lime")
        cached = self._cache.get(cache_key)
        if cached and cached.is_valid():
            logger.debug("LIME explanation served from cache", model_id=model_id)
            return cached.explanation

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/explain/lime",
                    json={
                        "model_id": model_id,
                        "input_data": input_data,
                        "prediction": prediction,
                        "num_samples": num_samples,
                        "top_features": top_features,
                    },
                    headers={"X-Tenant-ID": str(tenant_id)},
                )
                response.raise_for_status()
                explanation = response.json()

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning(
                "Explainability service unreachable — using stub LIME explanation",
                model_id=model_id,
                error=str(exc),
            )
            explanation = self._stub_lime_explanation(input_data, top_features)

        self._cache[cache_key] = CachedExplanation(explanation, self._cache_ttl)
        return explanation

    def rank_features(
        self,
        explanation: dict[str, Any],
        method: str = "shap",
    ) -> list[dict[str, Any]]:
        """Rank features by their importance magnitude from an explanation.

        Args:
            explanation: Explanation dict from get_shap_explanation or get_lime_explanation.
            method: Explanation method ("shap" or "lime").

        Returns:
            List of feature importance dicts sorted by absolute importance descending.
        """
        if method == "shap":
            shap_values = explanation.get("shap_values", {})
            feature_names = explanation.get("feature_names", list(shap_values.keys()))
            ranked = [
                {
                    "feature": feature,
                    "importance": shap_values.get(feature, 0.0),
                    "abs_importance": abs(shap_values.get(feature, 0.0)),
                    "direction": "positive" if shap_values.get(feature, 0.0) >= 0 else "negative",
                }
                for feature in feature_names
            ]
        else:
            lime_weights = explanation.get("lime_weights", {})
            feature_names = explanation.get("feature_names", list(lime_weights.keys()))
            ranked = [
                {
                    "feature": feature,
                    "importance": lime_weights.get(feature, 0.0),
                    "abs_importance": abs(lime_weights.get(feature, 0.0)),
                    "direction": "positive" if lime_weights.get(feature, 0.0) >= 0 else "negative",
                }
                for feature in feature_names
            ]

        ranked.sort(key=lambda item: item["abs_importance"], reverse=True)
        return ranked

    def format_for_reviewer(
        self,
        explanation: dict[str, Any],
        method: str = "shap",
        max_features: int = 5,
    ) -> str:
        """Format an explanation as natural language for human reviewers.

        Args:
            explanation: Explanation dict from get_shap_explanation or get_lime_explanation.
            method: Explanation method ("shap" or "lime").
            max_features: Maximum number of features to mention.

        Returns:
            Natural language explanation string.
        """
        ranked = self.rank_features(explanation, method)[:max_features]
        if not ranked:
            return "No feature importance data available for this prediction."

        confidence = explanation.get("confidence_score", None)
        confidence_str = (
            f" (confidence: {confidence:.1%})" if confidence is not None else ""
        )

        lines = [f"Explanation{confidence_str}:"]
        for rank, feature_info in enumerate(ranked, start=1):
            feature = feature_info["feature"]
            importance = feature_info["importance"]
            direction = feature_info["direction"]
            direction_phrase = "increased" if direction == "positive" else "decreased"
            lines.append(
                f"  {rank}. '{feature}' {direction_phrase} the prediction "
                f"by {abs(importance):.4f}"
            )

        return "\n".join(lines)

    def score_explanation_confidence(
        self,
        explanation: dict[str, Any],
        method: str = "shap",
    ) -> float:
        """Estimate how reliable/confident the explanation itself is.

        Args:
            explanation: Explanation dict.
            method: Explanation method.

        Returns:
            Explanation confidence score in [0.0, 1.0].
        """
        # If the service returned an explicit confidence_score, trust it
        if "confidence_score" in explanation:
            return float(explanation["confidence_score"])

        # Heuristic: more features with meaningful importance = higher confidence
        if method == "shap":
            values = list(explanation.get("shap_values", {}).values())
        else:
            values = list(explanation.get("lime_weights", {}).values())

        if not values:
            return 0.0

        non_zero = sum(1 for value in values if abs(value) > 1e-6)
        coverage = non_zero / len(values)
        # Penalise if too few features explain the prediction
        return round(min(1.0, coverage * 1.2), 4)

    def _stub_shap_explanation(
        self,
        input_data: dict[str, Any],
        top_features: int,
    ) -> dict[str, Any]:
        """Generate a stub SHAP explanation for development/fallback use.

        Args:
            input_data: Input data dict used to derive feature names.
            top_features: Number of features to include.

        Returns:
            Stub explanation dict.
        """
        features = list(input_data.keys())[:top_features]
        # Assign small stub SHAP values based on feature index
        shap_values = {
            feature: round(0.1 * (index + 1) * (-1 if index % 2 else 1), 4)
            for index, feature in enumerate(features)
        }
        return {
            "method": "shap",
            "model_id": "stub",
            "shap_values": shap_values,
            "feature_names": features,
            "confidence_score": 0.5,
            "is_stub": True,
        }

    def _stub_lime_explanation(
        self,
        input_data: dict[str, Any],
        top_features: int,
    ) -> dict[str, Any]:
        """Generate a stub LIME explanation for development/fallback use.

        Args:
            input_data: Input data dict used to derive feature names.
            top_features: Number of features to include.

        Returns:
            Stub explanation dict.
        """
        features = list(input_data.keys())[:top_features]
        lime_weights = {
            feature: round(0.05 * (index + 1) * (-1 if index % 3 else 1), 4)
            for index, feature in enumerate(features)
        }
        return {
            "method": "lime",
            "model_id": "stub",
            "lime_weights": lime_weights,
            "feature_names": features,
            "local_explanation": "Stub LIME explanation (service unavailable).",
            "confidence_score": 0.4,
            "is_stub": True,
        }

    async def get_explanation_for_review(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction: dict[str, Any],
        tenant_id: uuid.UUID,
        method: str = "shap",
    ) -> dict[str, Any]:
        """Get a complete explanation package ready for a reviewer interface.

        Fetches the explanation, ranks features, formats natural language
        summary, and scores explanation confidence.

        Args:
            model_id: Model identifier.
            input_data: Prediction input data.
            prediction: Model output.
            tenant_id: Requesting tenant UUID.
            method: Explanation method — "shap" or "lime".

        Returns:
            Dict with raw_explanation, ranked_features, reviewer_summary,
            explanation_confidence.

        Raises:
            ValueError: If method is unsupported.
        """
        if method not in {"shap", "lime"}:
            raise ValueError(f"Unsupported explanation method '{method}'. Use 'shap' or 'lime'.")

        if method == "shap":
            raw_explanation = await self.get_shap_explanation(
                model_id=model_id,
                input_data=input_data,
                prediction=prediction,
                tenant_id=tenant_id,
            )
        else:
            raw_explanation = await self.get_lime_explanation(
                model_id=model_id,
                input_data=input_data,
                prediction=prediction,
                tenant_id=tenant_id,
            )

        ranked_features = self.rank_features(raw_explanation, method)
        reviewer_summary = self.format_for_reviewer(raw_explanation, method)
        explanation_confidence = self.score_explanation_confidence(raw_explanation, method)

        logger.info(
            "Explanation package prepared for reviewer",
            model_id=model_id,
            method=method,
            explanation_confidence=explanation_confidence,
        )

        return {
            "raw_explanation": raw_explanation,
            "ranked_features": ranked_features,
            "reviewer_summary": reviewer_summary,
            "explanation_confidence": explanation_confidence,
            "method": method,
        }

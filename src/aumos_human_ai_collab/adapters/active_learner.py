"""Active learner adapter for the Human-AI Collaboration service.

Implements uncertainty sampling strategies for prioritising which unlabelled
samples to send to annotators next. Supports least-confident, margin, entropy,
query-by-committee, and diverse batch selection with budget management.
"""

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Supported uncertainty sampling strategies
VALID_STRATEGIES = frozenset(
    {"least_confident", "margin", "entropy", "committee", "diverse"}
)

# Default budget in labels per round
DEFAULT_LABELS_PER_ROUND = 50


class LearningCurvePoint:
    """A single measurement on the active learning curve."""

    __slots__ = ("round_index", "labels_used", "estimated_accuracy", "timestamp")

    def __init__(
        self,
        round_index: int,
        labels_used: int,
        estimated_accuracy: float,
        timestamp: datetime,
    ) -> None:
        self.round_index = round_index
        self.labels_used = labels_used
        self.estimated_accuracy = estimated_accuracy
        self.timestamp = timestamp

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            Dict representation.
        """
        return {
            "round_index": self.round_index,
            "labels_used": self.labels_used,
            "estimated_accuracy": self.estimated_accuracy,
            "timestamp": self.timestamp.isoformat(),
        }


class ActiveLearner:
    """Uncertainty-based sample selection for active learning pipelines.

    Maintains an unlabelled pool per session and selects the most informative
    samples to label each round, tracking budget and learning curve progress.
    """

    def __init__(
        self,
        strategy: str = "entropy",
        labels_per_round: int = DEFAULT_LABELS_PER_ROUND,
        max_budget: int | None = None,
    ) -> None:
        """Initialise the active learner.

        Args:
            strategy: Sampling strategy to use.
            labels_per_round: Maximum labels to select per round.
            max_budget: Total label budget across all rounds. None = unlimited.

        Raises:
            ValueError: If strategy is not supported.
        """
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Valid: {VALID_STRATEGIES}"
            )

        self._strategy = strategy
        self._labels_per_round = labels_per_round
        self._max_budget = max_budget
        self._total_labels_used = 0
        self._round_index = 0
        self._learning_curve: list[LearningCurvePoint] = []
        # session_id -> pool of candidate samples
        self._pools: dict[uuid.UUID, list[dict[str, Any]]] = {}

    def add_to_pool(
        self,
        session_id: uuid.UUID,
        samples: list[dict[str, Any]],
    ) -> int:
        """Add unlabelled samples to the active learning pool.

        Each sample must contain a 'sample_id' field. Probability distributions
        should be provided under 'probabilities' (list of floats summing to 1)
        for entropy/margin/least-confident strategies. For committee strategy,
        'committee_predictions' (list of list[float]) is expected.

        Args:
            session_id: Active learning session UUID.
            samples: List of sample dicts.

        Returns:
            Total pool size after adding samples.
        """
        if session_id not in self._pools:
            self._pools[session_id] = []
        self._pools[session_id].extend(samples)

        logger.debug(
            "Samples added to pool",
            session_id=str(session_id),
            added=len(samples),
            pool_size=len(self._pools[session_id]),
        )

        return len(self._pools[session_id])

    def _compute_entropy(self, probabilities: list[float]) -> float:
        """Compute Shannon entropy of a probability distribution.

        Args:
            probabilities: Class probability distribution.

        Returns:
            Entropy value (higher = more uncertain).
        """
        entropy = 0.0
        for probability in probabilities:
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return entropy

    def _compute_margin(self, probabilities: list[float]) -> float:
        """Compute the margin score (difference between top-2 probabilities).

        Args:
            probabilities: Class probability distribution.

        Returns:
            Negative margin (higher score = smaller margin = more uncertain).
        """
        sorted_probs = sorted(probabilities, reverse=True)
        if len(sorted_probs) < 2:
            return 0.0
        return -(sorted_probs[0] - sorted_probs[1])

    def _compute_least_confident(self, probabilities: list[float]) -> float:
        """Compute least-confident score (1 - max probability).

        Args:
            probabilities: Class probability distribution.

        Returns:
            1 minus the maximum probability (higher = less confident).
        """
        if not probabilities:
            return 0.0
        return 1.0 - max(probabilities)

    def _compute_committee_disagreement(
        self, committee_predictions: list[list[float]]
    ) -> float:
        """Compute vote entropy across committee member predictions.

        Args:
            committee_predictions: List of probability distributions, one per model.

        Returns:
            Average entropy across committee members.
        """
        if not committee_predictions:
            return 0.0
        return sum(
            self._compute_entropy(probs) for probs in committee_predictions
        ) / len(committee_predictions)

    def _score_sample(self, sample: dict[str, Any]) -> float:
        """Score a single sample using the configured strategy.

        Args:
            sample: Sample dict with strategy-specific fields.

        Returns:
            Uncertainty score (higher = more informative to label).
        """
        if self._strategy == "entropy":
            probabilities = sample.get("probabilities", [])
            return self._compute_entropy(probabilities)

        if self._strategy == "margin":
            probabilities = sample.get("probabilities", [])
            return self._compute_margin(probabilities)

        if self._strategy == "least_confident":
            probabilities = sample.get("probabilities", [])
            return self._compute_least_confident(probabilities)

        if self._strategy == "committee":
            committee_predictions = sample.get("committee_predictions", [])
            return self._compute_committee_disagreement(committee_predictions)

        # "diverse" strategy: score by uncertainty then apply diversity
        probabilities = sample.get("probabilities", [])
        return self._compute_entropy(probabilities)

    def _select_diverse(
        self,
        scored_samples: list[tuple[float, dict[str, Any]]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """Select a diverse batch by combining uncertainty with feature diversity.

        Uses greedy farthest-first selection on a scalar feature proxy
        (the uncertainty score itself is used as a 1D feature proxy when
        embeddings are not available, ensuring spread across uncertainty levels).

        Args:
            scored_samples: List of (score, sample) tuples sorted by score descending.
            batch_size: Number of samples to select.

        Returns:
            Selected diverse sample list.
        """
        if not scored_samples:
            return []

        selected: list[dict[str, Any]] = []
        remaining = list(scored_samples)
        # Seed with the most uncertain sample
        selected.append(remaining.pop(0)[1])

        while len(selected) < batch_size and remaining:
            # Choose the sample most different from already-selected
            # (maximise min distance in score space)
            best_index = 0
            best_min_dist = float("-inf")

            for candidate_index, (candidate_score, _) in enumerate(remaining):
                min_dist = min(
                    abs(candidate_score - selected_sample.get("_score", 0.0))
                    for selected_sample in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_index = candidate_index

            score, sample = remaining.pop(best_index)
            sample["_score"] = score
            selected.append(sample)

        return selected

    async def select_samples(
        self,
        session_id: uuid.UUID,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Select the most informative samples from the pool for labelling.

        Args:
            session_id: Active learning session UUID.
            batch_size: Override the per-round batch size.

        Returns:
            Dict with selected_samples, round_index, budget_remaining, strategy.

        Raises:
            KeyError: If session_id has no pool.
            RuntimeError: If budget is exhausted.
        """
        pool = self._pools.get(session_id)
        if pool is None:
            raise KeyError(f"No pool found for session {session_id}.")

        effective_batch = batch_size or self._labels_per_round

        # Budget check
        if self._max_budget is not None:
            remaining_budget = self._max_budget - self._total_labels_used
            if remaining_budget <= 0:
                raise RuntimeError(
                    f"Label budget exhausted. Total labels used: {self._total_labels_used}"
                )
            effective_batch = min(effective_batch, remaining_budget)

        effective_batch = min(effective_batch, len(pool))

        # Score all samples
        scored = sorted(
            [(self._score_sample(sample), sample) for sample in pool],
            key=lambda scored_tuple: scored_tuple[0],
            reverse=True,
        )

        if self._strategy == "diverse":
            # Attach scores to samples for diversity selection
            for score, sample in scored:
                sample["_score"] = score
            selected = self._select_diverse(scored, effective_batch)
        else:
            selected = [sample for _, sample in scored[:effective_batch]]

        # Remove selected samples from pool
        selected_ids = {sample.get("sample_id") for sample in selected}
        self._pools[session_id] = [
            sample for sample in pool if sample.get("sample_id") not in selected_ids
        ]

        self._total_labels_used += len(selected)
        self._round_index += 1

        budget_remaining = (
            self._max_budget - self._total_labels_used
            if self._max_budget is not None
            else None
        )

        logger.info(
            "Samples selected for labelling",
            session_id=str(session_id),
            selected_count=len(selected),
            round_index=self._round_index,
            strategy=self._strategy,
            budget_remaining=budget_remaining,
        )

        # Clean up internal _score metadata before returning
        for sample in selected:
            sample.pop("_score", None)

        return {
            "selected_samples": selected,
            "round_index": self._round_index,
            "total_selected": len(selected),
            "pool_remaining": len(self._pools[session_id]),
            "budget_used": self._total_labels_used,
            "budget_remaining": budget_remaining,
            "strategy": self._strategy,
        }

    async def record_round_result(
        self,
        session_id: uuid.UUID,
        labels_used: int,
        estimated_accuracy: float,
    ) -> LearningCurvePoint:
        """Record the outcome of a labelling round for learning curve tracking.

        Args:
            session_id: Active learning session UUID.
            labels_used: Number of labels used in this round.
            estimated_accuracy: Model accuracy estimate after this round's labels.

        Returns:
            LearningCurvePoint recorded.
        """
        point = LearningCurvePoint(
            round_index=self._round_index,
            labels_used=labels_used,
            estimated_accuracy=estimated_accuracy,
            timestamp=datetime.now(tz=timezone.utc),
        )
        self._learning_curve.append(point)

        logger.info(
            "Learning curve point recorded",
            session_id=str(session_id),
            round_index=self._round_index,
            labels_used=labels_used,
            estimated_accuracy=estimated_accuracy,
        )

        return point

    def get_learning_curve(self) -> list[dict[str, Any]]:
        """Return the full learning curve history.

        Returns:
            List of learning curve point dicts.
        """
        return [point.to_dict() for point in self._learning_curve]

    def get_budget_status(self) -> dict[str, Any]:
        """Return current budget usage statistics.

        Returns:
            Dict with total_labels_used, max_budget, budget_remaining, rounds_completed.
        """
        return {
            "total_labels_used": self._total_labels_used,
            "max_budget": self._max_budget,
            "budget_remaining": (
                self._max_budget - self._total_labels_used
                if self._max_budget is not None
                else None
            ),
            "rounds_completed": self._round_index,
            "strategy": self._strategy,
        }

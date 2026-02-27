"""Consensus scorer adapter for the Human-AI Collaboration service.

Computes inter-annotator agreement metrics: Fleiss' kappa, Cohen's kappa,
Krippendorff's alpha, majority voting label resolution, annotator reliability
scoring, and consensus report generation.
"""

import math
import uuid
from collections import Counter, defaultdict
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Kappa interpretation thresholds (Landis & Koch scale)
_KAPPA_THRESHOLDS = [
    (0.81, "almost_perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (float("-inf"), "poor"),
]


def _interpret_kappa(kappa: float) -> str:
    """Map a kappa value to a Landis-Koch agreement label.

    Args:
        kappa: Kappa score.

    Returns:
        Human-readable agreement level.
    """
    for threshold, label in _KAPPA_THRESHOLDS:
        if kappa >= threshold:
            return label
    return "poor"


class ConsensusScorer:
    """Multi-annotator agreement computation and label resolution.

    Accepts annotation matrices (item x annotator) and computes various
    inter-annotator agreement metrics. Provides majority voting for final
    label resolution and per-annotator reliability tracking.
    """

    def __init__(self) -> None:
        """Initialise with empty reliability history."""
        # annotator_id -> list of (expected_label, actual_label) for gold items
        self._reliability_history: dict[uuid.UUID, list[tuple[str, str]]] = defaultdict(list)

    def compute_fleiss_kappa(
        self,
        annotations: list[dict[str, str]],
        labels: list[str],
    ) -> dict[str, Any]:
        """Compute Fleiss' kappa for multiple annotators and multiple items.

        Args:
            annotations: List of item annotation dicts.
                         Each dict: {"item_id": str, "annotator_id": str, "label": str}.
            labels: All possible label values.

        Returns:
            Dict with kappa, agreement_level, observed_agreement, expected_agreement.

        Raises:
            ValueError: If annotations are empty or malformed.
        """
        if not annotations:
            raise ValueError("Cannot compute kappa on empty annotations.")

        # Group by item
        items: dict[str, list[str]] = defaultdict(list)
        for ann in annotations:
            items[ann["item_id"]].append(ann["label"])

        n_items = len(items)
        if n_items < 2:
            raise ValueError(
                "Fleiss' kappa requires at least 2 annotated items."
            )

        # Require each item to have the same number of annotators
        annotators_per_item_counts = [len(labels_list) for labels_list in items.values()]
        if len(set(annotators_per_item_counts)) != 1:
            raise ValueError(
                "Fleiss' kappa requires the same number of annotators per item."
            )

        n_annotators = annotators_per_item_counts[0]
        label_index = {label: index for index, label in enumerate(labels)}

        # Build count matrix: n_items x n_labels
        count_matrix: list[list[int]] = []
        for item_labels in items.values():
            row = [0] * len(labels)
            for label in item_labels:
                if label in label_index:
                    row[label_index[label]] += 1
            count_matrix.append(row)

        # Compute observed agreement P_i per item
        total = n_items * n_annotators * (n_annotators - 1)
        if total == 0:
            return {
                "kappa": 1.0,
                "agreement_level": "almost_perfect",
                "observed_agreement": 1.0,
                "expected_agreement": 1.0,
                "n_items": n_items,
                "n_annotators": n_annotators,
            }

        p_i_values = []
        for row in count_matrix:
            row_total = sum(row)
            if row_total < 2:
                p_i_values.append(0.0)
                continue
            numerator = sum(count * (count - 1) for count in row)
            p_i_values.append(numerator / (row_total * (row_total - 1)))

        observed_agreement = sum(p_i_values) / n_items

        # Compute expected agreement p_j (marginal proportions) per label
        total_annotations = n_items * n_annotators
        p_j_values = [
            sum(row[j] for row in count_matrix) / total_annotations
            for j in range(len(labels))
        ]
        expected_agreement = sum(p_j ** 2 for p_j in p_j_values)

        if expected_agreement >= 1.0:
            kappa = 1.0
        else:
            kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

        logger.debug(
            "Fleiss kappa computed",
            kappa=kappa,
            n_items=n_items,
            n_annotators=n_annotators,
        )

        return {
            "kappa": round(kappa, 4),
            "agreement_level": _interpret_kappa(kappa),
            "observed_agreement": round(observed_agreement, 4),
            "expected_agreement": round(expected_agreement, 4),
            "n_items": n_items,
            "n_annotators": n_annotators,
        }

    def compute_cohen_kappa(
        self,
        annotator_a_labels: list[str],
        annotator_b_labels: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """Compute Cohen's kappa for pairwise agreement between two annotators.

        Args:
            annotator_a_labels: Labels from annotator A (one per item).
            annotator_b_labels: Labels from annotator B (one per item).
            labels: All possible label values.

        Returns:
            Dict with kappa, agreement_level, observed_agreement, expected_agreement,
            confusion_matrix.

        Raises:
            ValueError: If label lists differ in length or are empty.
        """
        if len(annotator_a_labels) != len(annotator_b_labels):
            raise ValueError(
                "Both annotators must have the same number of labels."
            )
        if not annotator_a_labels:
            raise ValueError("Cannot compute Cohen's kappa on empty label lists.")

        n = len(annotator_a_labels)
        label_index = {label: index for index, label in enumerate(labels)}
        n_labels = len(labels)

        # Confusion matrix
        confusion_matrix = [[0] * n_labels for _ in range(n_labels)]
        for label_a, label_b in zip(annotator_a_labels, annotator_b_labels):
            index_a = label_index.get(label_a)
            index_b = label_index.get(label_b)
            if index_a is not None and index_b is not None:
                confusion_matrix[index_a][index_b] += 1

        # Observed agreement
        observed_agreement = sum(
            confusion_matrix[i][i] for i in range(n_labels)
        ) / n

        # Expected agreement
        row_sums = [sum(row) / n for row in confusion_matrix]
        col_sums = [sum(confusion_matrix[row][col] for row in range(n_labels)) / n
                    for col in range(n_labels)]
        expected_agreement = sum(row_sums[i] * col_sums[i] for i in range(n_labels))

        if expected_agreement >= 1.0:
            kappa = 1.0
        else:
            kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

        return {
            "kappa": round(kappa, 4),
            "agreement_level": _interpret_kappa(kappa),
            "observed_agreement": round(observed_agreement, 4),
            "expected_agreement": round(expected_agreement, 4),
            "n_items": n,
            "confusion_matrix": {
                "labels": labels,
                "matrix": confusion_matrix,
            },
        }

    def compute_krippendorff_alpha(
        self,
        data: list[list[float | None]],
        level_of_measurement: str = "nominal",
    ) -> dict[str, Any]:
        """Compute Krippendorff's alpha for ordinal or nominal data.

        Args:
            data: Matrix of shape [n_annotators x n_items]. None means missing value.
            level_of_measurement: "nominal" or "ordinal".

        Returns:
            Dict with alpha, agreement_level, and level_of_measurement.

        Raises:
            ValueError: If level_of_measurement is unsupported or data is empty.
        """
        if level_of_measurement not in {"nominal", "ordinal"}:
            raise ValueError(
                f"Unsupported level_of_measurement '{level_of_measurement}'. "
                "Use 'nominal' or 'ordinal'."
            )
        if not data or not data[0]:
            raise ValueError("Data matrix must be non-empty.")

        n_annotators = len(data)
        n_items = len(data[0])

        # Collect pairable observations per unit
        observed_disagreement = 0.0
        expected_disagreement = 0.0
        n_pairs = 0

        # All observed values (excluding None)
        all_values: list[float] = [
            value
            for row in data
            for value in row
            if value is not None
        ]

        if len(all_values) < 2:
            return {
                "alpha": 1.0,
                "agreement_level": "almost_perfect",
                "level_of_measurement": level_of_measurement,
            }

        n_total = len(all_values)
        all_values_count = Counter(all_values)

        def metric_diff(value_a: float, value_b: float) -> float:
            """Compute metric difference based on measurement level.

            Args:
                value_a: First value.
                value_b: Second value.

            Returns:
                Squared difference.
            """
            if level_of_measurement == "nominal":
                return 0.0 if value_a == value_b else 1.0
            # ordinal: (rank_a - rank_b)^2
            return (value_a - value_b) ** 2

        # Observed disagreement: within-unit pairs
        for item_index in range(n_items):
            unit_values = [
                data[annotator_index][item_index]
                for annotator_index in range(n_annotators)
                if data[annotator_index][item_index] is not None
            ]
            n_u = len(unit_values)
            if n_u < 2:
                continue
            for i in range(n_u):
                for j in range(i + 1, n_u):
                    observed_disagreement += metric_diff(unit_values[i], unit_values[j])
                    n_pairs += 1

        # Expected disagreement: global pairs
        all_values_list = list(all_values_count.keys())
        global_pairs = 0.0
        for i, value_i in enumerate(all_values_list):
            for j, value_j in enumerate(all_values_list):
                count_i = all_values_count[value_i]
                count_j = all_values_count[value_j]
                if i != j:
                    global_pairs += count_i * count_j * metric_diff(value_i, value_j)
                else:
                    global_pairs += count_i * (count_i - 1) * metric_diff(value_i, value_j)

        if global_pairs > 0 and n_pairs > 0:
            n_c = n_total * (n_total - 1)
            alpha = 1.0 - (observed_disagreement / n_pairs) / (global_pairs / n_c)
        else:
            alpha = 1.0

        return {
            "alpha": round(alpha, 4),
            "agreement_level": _interpret_kappa(alpha),
            "level_of_measurement": level_of_measurement,
        }

    def majority_vote(
        self,
        item_id: str,
        annotations: list[dict[str, str]],
        min_agreement_pct: float = 0.5,
    ) -> dict[str, Any]:
        """Resolve a label for an item via majority voting.

        Args:
            item_id: Item being resolved.
            annotations: List of annotation dicts with 'annotator_id' and 'label'.
            min_agreement_pct: Minimum fraction of annotators that must agree.

        Returns:
            Dict with resolved_label (or None), agreement_pct,
            vote_counts, and is_consensus_reached.
        """
        if not annotations:
            return {
                "item_id": item_id,
                "resolved_label": None,
                "agreement_pct": 0.0,
                "vote_counts": {},
                "is_consensus_reached": False,
            }

        vote_counts = Counter(ann["label"] for ann in annotations)
        total_votes = len(annotations)
        winner_label, winner_count = vote_counts.most_common(1)[0]
        agreement_pct = winner_count / total_votes
        is_consensus_reached = agreement_pct >= min_agreement_pct

        return {
            "item_id": item_id,
            "resolved_label": winner_label if is_consensus_reached else None,
            "agreement_pct": round(agreement_pct, 4),
            "vote_counts": dict(vote_counts),
            "is_consensus_reached": is_consensus_reached,
        }

    def analyze_disagreements(
        self,
        annotations: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Identify items with the highest annotator disagreement.

        Args:
            annotations: List of annotation dicts with 'item_id', 'annotator_id', 'label'.

        Returns:
            List of disagreement records sorted by entropy (highest first).
        """
        # Group by item
        items: dict[str, list[str]] = defaultdict(list)
        for ann in annotations:
            items[ann["item_id"]].append(ann["label"])

        disagreements = []
        for item_id, item_labels in items.items():
            n = len(item_labels)
            if n < 2:
                continue
            counts = Counter(item_labels)
            n_unique = len(counts)
            # Entropy as disagreement measure
            entropy = 0.0
            for count in counts.values():
                probability = count / n
                if probability > 0:
                    entropy -= probability * math.log2(probability)

            disagreements.append(
                {
                    "item_id": item_id,
                    "n_annotations": n,
                    "n_unique_labels": n_unique,
                    "entropy": round(entropy, 4),
                    "label_distribution": dict(counts),
                }
            )

        disagreements.sort(key=lambda record: record["entropy"], reverse=True)
        return disagreements

    def record_gold_result(
        self,
        annotator_id: uuid.UUID,
        expected_label: str,
        actual_label: str,
    ) -> None:
        """Record a gold standard check result for annotator reliability tracking.

        Args:
            annotator_id: Annotator UUID.
            expected_label: The correct gold standard label.
            actual_label: The label the annotator chose.
        """
        self._reliability_history[annotator_id].append((expected_label, actual_label))

    def get_annotator_reliability(self, annotator_id: uuid.UUID) -> dict[str, Any]:
        """Compute reliability score for an annotator from gold standard history.

        Args:
            annotator_id: Annotator UUID.

        Returns:
            Dict with accuracy, total_gold_checks, and reliability_tier.
        """
        history = self._reliability_history.get(annotator_id, [])
        if not history:
            return {
                "annotator_id": str(annotator_id),
                "accuracy": None,
                "total_gold_checks": 0,
                "reliability_tier": "unverified",
            }

        correct = sum(1 for expected, actual in history if expected == actual)
        accuracy = correct / len(history)

        if accuracy >= 0.95:
            tier = "expert"
        elif accuracy >= 0.80:
            tier = "reliable"
        elif accuracy >= 0.65:
            tier = "moderate"
        else:
            tier = "low_reliability"

        return {
            "annotator_id": str(annotator_id),
            "accuracy": round(accuracy, 4),
            "total_gold_checks": len(history),
            "reliability_tier": tier,
        }

    def generate_consensus_report(
        self,
        task_id: str,
        annotations: list[dict[str, str]],
        labels: list[str],
        annotator_ids: list[uuid.UUID],
    ) -> dict[str, Any]:
        """Generate a comprehensive consensus report for a completed annotation task.

        Includes Fleiss' kappa, item-level majority vote resolutions, top disagreements,
        and per-annotator reliability scores.

        Args:
            task_id: Task identifier (for labeling the report).
            annotations: All task annotations.
            labels: All possible labels.
            annotator_ids: All annotators on the task.

        Returns:
            Comprehensive consensus report dict.
        """
        report: dict[str, Any] = {
            "task_id": task_id,
            "total_annotations": len(annotations),
            "n_labels": len(labels),
        }

        if annotations:
            try:
                report["fleiss_kappa"] = self.compute_fleiss_kappa(annotations, labels)
            except ValueError as exc:
                report["fleiss_kappa"] = {"error": str(exc)}

        # Majority vote per item
        items: dict[str, list[dict[str, str]]] = defaultdict(list)
        for ann in annotations:
            items[ann["item_id"]].append(ann)

        resolved_items = []
        for item_id, item_annotations in items.items():
            resolved_items.append(
                self.majority_vote(item_id, item_annotations)
            )

        consensus_reached = sum(
            1 for item in resolved_items if item["is_consensus_reached"]
        )
        report["resolved_items"] = len(resolved_items)
        report["consensus_reached_count"] = consensus_reached
        report["consensus_rate_pct"] = round(
            consensus_reached / len(resolved_items) * 100
            if resolved_items else 0.0,
            2,
        )

        # Top disagreements
        report["top_disagreements"] = self.analyze_disagreements(annotations)[:5]

        # Annotator reliability
        report["annotator_reliability"] = [
            self.get_annotator_reliability(annotator_id)
            for annotator_id in annotator_ids
        ]

        logger.info(
            "Consensus report generated",
            task_id=task_id,
            total_annotations=len(annotations),
            consensus_rate_pct=report.get("consensus_rate_pct"),
        )

        return report

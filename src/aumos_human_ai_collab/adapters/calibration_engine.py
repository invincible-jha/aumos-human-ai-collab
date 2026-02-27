"""Calibration engine adapter for the Human-AI Collaboration service.

Implements confidence score recalibration using Platt scaling, temperature
scaling, and isotonic regression. Computes calibration curves (reliability
diagram data), Expected Calibration Error (ECE), drives feedback-based
recalibration, and maintains calibration history per task type.
"""

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Number of bins for ECE and calibration curve computation
_DEFAULT_BINS = 10


def _sigmoid(value: float) -> float:
    """Compute the logistic sigmoid function.

    Args:
        value: Input value.

    Returns:
        Sigmoid of the input clamped to [0, 1].
    """
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, value))))


class CalibrationRecord:
    """An entry in the calibration history."""

    __slots__ = (
        "record_id",
        "tenant_id",
        "task_type",
        "method",
        "params",
        "ece_before",
        "ece_after",
        "sample_count",
        "applied_at",
    )

    def __init__(
        self,
        record_id: uuid.UUID,
        tenant_id: uuid.UUID,
        task_type: str,
        method: str,
        params: dict[str, Any],
        ece_before: float | None,
        ece_after: float | None,
        sample_count: int,
        applied_at: datetime,
    ) -> None:
        self.record_id = record_id
        self.tenant_id = tenant_id
        self.task_type = task_type
        self.method = method
        self.params = params
        self.ece_before = ece_before
        self.ece_after = ece_after
        self.sample_count = sample_count
        self.applied_at = applied_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            Dict representation.
        """
        return {
            "record_id": str(self.record_id),
            "tenant_id": str(self.tenant_id),
            "task_type": self.task_type,
            "method": self.method,
            "params": self.params,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "sample_count": self.sample_count,
            "applied_at": self.applied_at.isoformat(),
        }


class CalibrationEngine:
    """Confidence score recalibration with multiple calibration methods.

    Supports Platt scaling (logistic regression on raw logits), temperature
    scaling (single-parameter softmax adjustment), and isotonic regression
    (non-parametric monotone mapping). Tracks calibration history and computes
    ECE for evaluation.
    """

    def __init__(self) -> None:
        """Initialise with empty calibration state and history."""
        # tenant_id -> task_type -> current calibration params
        self._params: dict[uuid.UUID, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # tenant_id -> list of CalibrationRecord
        self._history: dict[uuid.UUID, list[CalibrationRecord]] = defaultdict(list)

    def compute_ece(
        self,
        confidences: list[float],
        labels: list[int],
        n_bins: int = _DEFAULT_BINS,
    ) -> dict[str, Any]:
        """Compute the Expected Calibration Error (ECE).

        Args:
            confidences: List of predicted confidence scores in [0, 1].
            labels: List of binary ground-truth labels (0 or 1).
            n_bins: Number of histogram bins for ECE computation.

        Returns:
            Dict with ece, bin_data (for reliability diagram), and n_samples.

        Raises:
            ValueError: If confidences and labels differ in length or are empty.
        """
        if len(confidences) != len(labels):
            raise ValueError("confidences and labels must have the same length.")
        if not confidences:
            raise ValueError("Cannot compute ECE on empty inputs.")

        n = len(confidences)
        bin_width = 1.0 / n_bins
        bin_totals = [0] * n_bins
        bin_correct = [0] * n_bins
        bin_confidence_sum = [0.0] * n_bins

        for confidence, label in zip(confidences, labels):
            bin_idx = min(int(confidence / bin_width), n_bins - 1)
            bin_totals[bin_idx] += 1
            bin_correct[bin_idx] += int(label)
            bin_confidence_sum[bin_idx] += confidence

        ece = 0.0
        bin_data = []
        for bin_idx in range(n_bins):
            total = bin_totals[bin_idx]
            if total == 0:
                continue
            accuracy = bin_correct[bin_idx] / total
            avg_confidence = bin_confidence_sum[bin_idx] / total
            ece += (total / n) * abs(accuracy - avg_confidence)
            bin_data.append(
                {
                    "bin_lower": round(bin_idx * bin_width, 4),
                    "bin_upper": round((bin_idx + 1) * bin_width, 4),
                    "avg_confidence": round(avg_confidence, 4),
                    "accuracy": round(accuracy, 4),
                    "count": total,
                }
            )

        return {
            "ece": round(ece, 6),
            "n_bins": n_bins,
            "n_samples": n,
            "bin_data": bin_data,
        }

    def compute_calibration_curve(
        self,
        confidences: list[float],
        labels: list[int],
        n_bins: int = _DEFAULT_BINS,
    ) -> dict[str, Any]:
        """Compute reliability diagram data (calibration curve).

        Args:
            confidences: Predicted confidence scores.
            labels: Binary ground-truth labels.
            n_bins: Number of histogram bins.

        Returns:
            Dict with mean_predicted_confidence, fraction_of_positives per bin,
            and perfect_calibration_line.
        """
        ece_result = self.compute_ece(confidences, labels, n_bins)
        bin_data = ece_result["bin_data"]

        mean_predicted = [b["avg_confidence"] for b in bin_data]
        fraction_positive = [b["accuracy"] for b in bin_data]
        perfect_line = [b["avg_confidence"] for b in bin_data]

        return {
            "mean_predicted_confidence": mean_predicted,
            "fraction_of_positives": fraction_positive,
            "perfect_calibration_line": perfect_line,
            "ece": ece_result["ece"],
            "n_samples": ece_result["n_samples"],
        }

    def platt_scale(
        self,
        logits: list[float],
        labels: list[int],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        """Fit Platt scaling parameters (logistic regression on logits).

        Trains a bias and scale parameter (intercept a, slope b) so that
        sigmoid(a * logit + b) is calibrated. Uses gradient descent.

        Args:
            logits: Raw model log-odds or logit outputs.
            labels: Binary ground-truth labels (0 or 1).
            n_iterations: Number of gradient descent steps.
            learning_rate: Step size for gradient descent.

        Returns:
            Dict with slope (a), intercept (b), and training_loss_final.

        Raises:
            ValueError: If logits and labels differ in length or are empty.
        """
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have the same length.")
        if not logits:
            raise ValueError("Cannot fit Platt scaling on empty data.")

        n = len(logits)
        scale = 1.0  # coefficient a
        intercept = 0.0  # bias b

        for _ in range(n_iterations):
            grad_scale = 0.0
            grad_intercept = 0.0
            total_loss = 0.0

            for logit, label in zip(logits, labels):
                probability = _sigmoid(scale * logit + intercept)
                error = probability - label
                grad_scale += error * logit
                grad_intercept += error
                # Binary cross-entropy
                probability_clipped = max(1e-9, min(1 - 1e-9, probability))
                total_loss -= label * math.log(probability_clipped) + (
                    1 - label
                ) * math.log(1 - probability_clipped)

            scale -= learning_rate * grad_scale / n
            intercept -= learning_rate * grad_intercept / n

        return {
            "method": "platt_scaling",
            "slope": round(scale, 6),
            "intercept": round(intercept, 6),
            "training_loss_final": round(total_loss / n, 6),
        }

    def temperature_scale(
        self,
        logits: list[float],
        labels: list[int],
        n_iterations: int = 50,
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        """Fit a temperature scaling parameter for confidence calibration.

        Divides logits by temperature T; temperature > 1 reduces overconfidence,
        temperature < 1 increases confidence.

        Args:
            logits: Raw model logit outputs.
            labels: Binary ground-truth labels (0 or 1).
            n_iterations: Number of gradient descent steps.
            learning_rate: Step size.

        Returns:
            Dict with temperature, and training_loss_final.

        Raises:
            ValueError: If logits and labels differ in length or are empty.
        """
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have the same length.")
        if not logits:
            raise ValueError("Cannot fit temperature scaling on empty data.")

        n = len(logits)
        temperature = 1.0

        for _ in range(n_iterations):
            grad = 0.0
            total_loss = 0.0

            for logit, label in zip(logits, labels):
                scaled_logit = logit / max(temperature, 1e-6)
                probability = _sigmoid(scaled_logit)
                error = probability - label
                # Gradient of loss w.r.t. temperature: d/dT sigmoid(logit/T)
                grad += error * (-logit / max(temperature ** 2, 1e-12))
                probability_clipped = max(1e-9, min(1 - 1e-9, probability))
                total_loss -= label * math.log(probability_clipped) + (
                    1 - label
                ) * math.log(1 - probability_clipped)

            temperature -= learning_rate * grad / n
            temperature = max(0.1, temperature)  # Clamp to avoid degenerate solutions

        return {
            "method": "temperature_scaling",
            "temperature": round(temperature, 6),
            "training_loss_final": round(total_loss / n, 6),
        }

    def isotonic_regression(
        self,
        confidences: list[float],
        labels: list[int],
    ) -> dict[str, Any]:
        """Fit isotonic regression calibration (pool adjacent violators algorithm).

        Maps raw confidence scores to calibrated probabilities using a
        non-parametric, monotone-increasing mapping.

        Args:
            confidences: Predicted confidence scores (not necessarily sorted).
            labels: Binary ground-truth labels.

        Returns:
            Dict with calibration_mapping (list of (confidence, calibrated_prob) pairs)
            sorted by confidence ascending.

        Raises:
            ValueError: If confidences and labels differ in length or are empty.
        """
        if len(confidences) != len(labels):
            raise ValueError("confidences and labels must have the same length.")
        if not confidences:
            raise ValueError("Cannot fit isotonic regression on empty data.")

        # Sort by confidence ascending
        paired = sorted(zip(confidences, labels), key=lambda pair: pair[0])
        sorted_confidences, sorted_labels = zip(*paired)

        # Pool adjacent violators algorithm
        blocks: list[list[float]] = [[float(sorted_labels[0])]]

        for label in sorted_labels[1:]:
            blocks.append([float(label)])
            # Merge violating blocks (enforce monotone non-decreasing)
            while len(blocks) > 1:
                prev_mean = sum(blocks[-2]) / len(blocks[-2])
                curr_mean = sum(blocks[-1]) / len(blocks[-1])
                if prev_mean > curr_mean:
                    merged = blocks[-2] + blocks[-1]
                    blocks = blocks[:-2] + [merged]
                else:
                    break

        # Build calibrated probability sequence
        calibrated: list[float] = []
        for block in blocks:
            block_mean = sum(block) / len(block)
            calibrated.extend([block_mean] * len(block))

        mapping = [
            {"confidence": round(conf, 6), "calibrated_prob": round(cal, 6)}
            for conf, cal in zip(sorted_confidences, calibrated)
        ]

        return {
            "method": "isotonic_regression",
            "calibration_mapping": mapping,
            "n_samples": len(calibrated),
        }

    def apply_calibration(
        self,
        raw_confidence: float,
        params: dict[str, Any],
    ) -> float:
        """Apply stored calibration parameters to a raw confidence score.

        Args:
            raw_confidence: Raw model confidence.
            params: Calibration params dict (as returned by platt_scale or temperature_scale).

        Returns:
            Calibrated confidence score in [0, 1].
        """
        method = params.get("method", "")

        if method == "platt_scaling":
            slope = params.get("slope", 1.0)
            intercept = params.get("intercept", 0.0)
            # Treat confidence as logit proxy
            logit = math.log(max(1e-9, raw_confidence) / max(1e-9, 1.0 - raw_confidence))
            return round(_sigmoid(slope * logit + intercept), 6)

        if method == "temperature_scaling":
            temperature = params.get("temperature", 1.0)
            logit = math.log(max(1e-9, raw_confidence) / max(1e-9, 1.0 - raw_confidence))
            return round(_sigmoid(logit / max(temperature, 1e-6)), 6)

        if method == "isotonic_regression":
            mapping = params.get("calibration_mapping", [])
            if not mapping:
                return raw_confidence
            # Linear interpolation using nearest neighbours
            sorted_mapping = sorted(mapping, key=lambda entry: entry["confidence"])
            for idx, entry in enumerate(sorted_mapping):
                if raw_confidence <= entry["confidence"]:
                    if idx == 0:
                        return entry["calibrated_prob"]
                    prev_entry = sorted_mapping[idx - 1]
                    # Interpolate
                    frac = (raw_confidence - prev_entry["confidence"]) / max(
                        entry["confidence"] - prev_entry["confidence"], 1e-9
                    )
                    return round(
                        prev_entry["calibrated_prob"]
                        + frac * (entry["calibrated_prob"] - prev_entry["calibrated_prob"]),
                        6,
                    )
            return sorted_mapping[-1]["calibrated_prob"]

        return raw_confidence

    async def recalibrate_from_feedback(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        confidences: list[float],
        labels: list[int],
        method: str = "temperature_scaling",
    ) -> CalibrationRecord:
        """Recalibrate confidence scores from feedback-derived ground truth.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type being calibrated.
            confidences: Historical confidence scores.
            labels: Corresponding ground truth labels (1=correct routing, 0=incorrect).
            method: Calibration method to use.

        Returns:
            CalibrationRecord documenting the recalibration.

        Raises:
            ValueError: If method is not supported.
        """
        valid_methods = {"platt_scaling", "temperature_scaling", "isotonic_regression"}
        if method not in valid_methods:
            raise ValueError(
                f"Unsupported calibration method '{method}'. Valid: {valid_methods}"
            )

        # Compute ECE before
        ece_before_result = self.compute_ece(confidences, labels)
        ece_before = ece_before_result["ece"]

        # Fit calibration params
        if method == "platt_scaling":
            logits = [
                math.log(max(1e-9, conf) / max(1e-9, 1.0 - conf))
                for conf in confidences
            ]
            params = self.platt_scale(logits, labels)
        elif method == "temperature_scaling":
            logits = [
                math.log(max(1e-9, conf) / max(1e-9, 1.0 - conf))
                for conf in confidences
            ]
            params = self.temperature_scale(logits, labels)
        else:
            params = self.isotonic_regression(confidences, labels)

        # Store params
        self._params[tenant_id][task_type] = params

        # Compute ECE after calibration
        calibrated_confidences = [
            self.apply_calibration(conf, params) for conf in confidences
        ]
        ece_after_result = self.compute_ece(calibrated_confidences, labels)
        ece_after = ece_after_result["ece"]

        record = CalibrationRecord(
            record_id=uuid.uuid4(),
            tenant_id=tenant_id,
            task_type=task_type,
            method=method,
            params=params,
            ece_before=ece_before,
            ece_after=ece_after,
            sample_count=len(confidences),
            applied_at=datetime.now(tz=timezone.utc),
        )
        self._history[tenant_id].append(record)

        logger.info(
            "Confidence recalibration applied",
            tenant_id=str(tenant_id),
            task_type=task_type,
            method=method,
            ece_before=ece_before,
            ece_after=ece_after,
            sample_count=len(confidences),
        )

        return record

    def get_current_params(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> dict[str, Any] | None:
        """Retrieve the current calibration parameters for a task type.

        Args:
            tenant_id: Requesting tenant.
            task_type: Task type.

        Returns:
            Calibration params dict or None if not yet calibrated.
        """
        return self._params[tenant_id].get(task_type)

    def get_calibration_history(
        self,
        tenant_id: uuid.UUID,
        task_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return calibration history for a tenant.

        Args:
            tenant_id: Requesting tenant.
            task_type: Optional filter by task type.
            limit: Maximum records to return (newest first).

        Returns:
            List of CalibrationRecord dicts.
        """
        records = self._history.get(tenant_id, [])
        if task_type is not None:
            records = [record for record in records if record.task_type == task_type]
        records_sorted = sorted(records, key=lambda record: record.applied_at, reverse=True)
        return [record.to_dict() for record in records_sorted[:limit]]

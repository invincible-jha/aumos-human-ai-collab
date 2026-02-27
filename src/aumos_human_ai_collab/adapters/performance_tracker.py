"""Performance tracker adapter for the Human-AI Collaboration service.

Tracks continuous model accuracy over time, attributes performance improvements
to feedback rounds, compares model version A/B performance, slices metrics by
user segment and data type, computes improvement velocity, and generates reports.
"""

import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Supported metric names
VALID_METRICS = frozenset(
    {"accuracy", "precision", "recall", "f1", "auc", "error_rate", "latency_ms"}
)


class MetricSnapshot:
    """A single model metric measurement at a point in time."""

    __slots__ = (
        "snapshot_id",
        "model_id",
        "model_version",
        "metric_name",
        "metric_value",
        "segment",
        "data_type",
        "feedback_round_id",
        "recorded_at",
    )

    def __init__(
        self,
        snapshot_id: uuid.UUID,
        model_id: str,
        model_version: str,
        metric_name: str,
        metric_value: float,
        segment: str | None,
        data_type: str | None,
        feedback_round_id: str | None,
        recorded_at: datetime,
    ) -> None:
        self.snapshot_id = snapshot_id
        self.model_id = model_id
        self.model_version = model_version
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.segment = segment
        self.data_type = data_type
        self.feedback_round_id = feedback_round_id
        self.recorded_at = recorded_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            Dict representation.
        """
        return {
            "snapshot_id": str(self.snapshot_id),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "segment": self.segment,
            "data_type": self.data_type,
            "feedback_round_id": self.feedback_round_id,
            "recorded_at": self.recorded_at.isoformat(),
        }


class PerformanceTracker:
    """Continuous model performance tracking with attribution and A/B comparison.

    Stores metric snapshots per tenant and provides aggregation, trend analysis,
    and improvement velocity computation. Supports per-segment and per-data-type
    slicing for granular performance visibility.
    """

    def __init__(self) -> None:
        """Initialise with empty metric storage."""
        # tenant_id -> list of MetricSnapshot
        self._snapshots: dict[uuid.UUID, list[MetricSnapshot]] = defaultdict(list)
        # tenant_id -> {feedback_round_id -> list of snapshot_ids}
        self._feedback_attribution: dict[uuid.UUID, dict[str, list[uuid.UUID]]] = defaultdict(
            lambda: defaultdict(list)
        )

    async def record_metric(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        model_version: str,
        metric_name: str,
        metric_value: float,
        segment: str | None = None,
        data_type: str | None = None,
        feedback_round_id: str | None = None,
    ) -> MetricSnapshot:
        """Record a model metric snapshot.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Model identifier.
            model_version: Model version string.
            metric_name: Metric name (accuracy, precision, etc.).
            metric_value: Measured value.
            segment: Optional user segment or group filter.
            data_type: Optional data type label.
            feedback_round_id: Optional ID of the feedback round that preceded this measurement.

        Returns:
            Created MetricSnapshot.

        Raises:
            ValueError: If metric_name is not in the valid set.
        """
        if metric_name not in VALID_METRICS:
            raise ValueError(
                f"Invalid metric_name '{metric_name}'. Valid: {VALID_METRICS}"
            )

        snapshot = MetricSnapshot(
            snapshot_id=uuid.uuid4(),
            model_id=model_id,
            model_version=model_version,
            metric_name=metric_name,
            metric_value=metric_value,
            segment=segment,
            data_type=data_type,
            feedback_round_id=feedback_round_id,
            recorded_at=datetime.now(tz=timezone.utc),
        )
        self._snapshots[tenant_id].append(snapshot)

        if feedback_round_id is not None:
            self._feedback_attribution[tenant_id][feedback_round_id].append(
                snapshot.snapshot_id
            )

        logger.info(
            "Metric snapshot recorded",
            tenant_id=str(tenant_id),
            model_id=model_id,
            model_version=model_version,
            metric_name=metric_name,
            metric_value=metric_value,
            segment=segment,
            feedback_round_id=feedback_round_id,
        )

        return snapshot

    async def get_accuracy_trend(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        metric_name: str = "accuracy",
        window_days: int = 30,
        segment: str | None = None,
    ) -> dict[str, Any]:
        """Return metric trend over a time window.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            metric_name: Metric to trend.
            window_days: Lookback window in days.
            segment: Optional segment filter.

        Returns:
            Dict with data_points (list), trend_direction, current_value, delta.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        snapshots = [
            snap
            for snap in self._snapshots.get(tenant_id, [])
            if snap.model_id == model_id
            and snap.metric_name == metric_name
            and snap.recorded_at >= cutoff
            and (segment is None or snap.segment == segment)
        ]
        snapshots.sort(key=lambda snap: snap.recorded_at)

        if not snapshots:
            return {
                "model_id": model_id,
                "metric_name": metric_name,
                "window_days": window_days,
                "data_points": [],
                "trend_direction": "insufficient_data",
                "current_value": None,
                "delta": None,
            }

        data_points = [
            {
                "timestamp": snap.recorded_at.isoformat(),
                "value": snap.metric_value,
                "version": snap.model_version,
                "feedback_round_id": snap.feedback_round_id,
            }
            for snap in snapshots
        ]

        current_value = snapshots[-1].metric_value
        first_value = snapshots[0].metric_value
        delta = current_value - first_value

        if len(snapshots) < 3:
            trend_direction = "insufficient_data"
        elif delta > 0.01:
            trend_direction = "improving"
        elif delta < -0.01:
            trend_direction = "degrading"
        else:
            trend_direction = "stable"

        return {
            "model_id": model_id,
            "metric_name": metric_name,
            "window_days": window_days,
            "segment": segment,
            "data_points": data_points,
            "trend_direction": trend_direction,
            "current_value": round(current_value, 6),
            "first_value": round(first_value, 6),
            "delta": round(delta, 6),
        }

    async def attribute_improvement(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        feedback_round_id: str,
        metric_name: str = "accuracy",
    ) -> dict[str, Any]:
        """Attribute a metric improvement to a specific feedback round.

        Compares the metric value immediately before and after the feedback round.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model to analyse.
            feedback_round_id: Feedback round identifier to attribute.
            metric_name: Metric to measure improvement on.

        Returns:
            Dict with pre_round_value, post_round_value, delta, and attribution.
        """
        all_snapshots = sorted(
            [
                snap
                for snap in self._snapshots.get(tenant_id, [])
                if snap.model_id == model_id and snap.metric_name == metric_name
            ],
            key=lambda snap: snap.recorded_at,
        )

        # Find the first snapshot with this feedback_round_id
        round_snapshot_indices = [
            idx
            for idx, snap in enumerate(all_snapshots)
            if snap.feedback_round_id == feedback_round_id
        ]

        if not round_snapshot_indices:
            return {
                "feedback_round_id": feedback_round_id,
                "model_id": model_id,
                "metric_name": metric_name,
                "attribution": "no_data_for_round",
                "pre_round_value": None,
                "post_round_value": None,
                "delta": None,
            }

        first_round_idx = round_snapshot_indices[0]
        pre_round_snapshots = all_snapshots[:first_round_idx]
        post_round_snapshots = [
            snap
            for snap in all_snapshots
            if snap.feedback_round_id == feedback_round_id
        ]

        pre_round_value = pre_round_snapshots[-1].metric_value if pre_round_snapshots else None
        post_round_value = (
            sum(snap.metric_value for snap in post_round_snapshots) / len(post_round_snapshots)
        )

        if pre_round_value is None:
            delta = None
            attribution = "no_baseline"
        else:
            delta = post_round_value - pre_round_value
            attribution = "improvement" if delta > 0 else ("regression" if delta < 0 else "neutral")

        return {
            "feedback_round_id": feedback_round_id,
            "model_id": model_id,
            "metric_name": metric_name,
            "pre_round_value": round(pre_round_value, 6) if pre_round_value is not None else None,
            "post_round_value": round(post_round_value, 6),
            "delta": round(delta, 6) if delta is not None else None,
            "attribution": attribution,
            "post_round_samples": len(post_round_snapshots),
        }

    async def compare_versions(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        version_a: str,
        version_b: str,
        metric_name: str = "accuracy",
        window_days: int = 14,
    ) -> dict[str, Any]:
        """Compare performance between two model versions (A/B analysis).

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            version_a: First version string.
            version_b: Second version string.
            metric_name: Metric to compare.
            window_days: Lookback window in days.

        Returns:
            Dict with version_a_stats, version_b_stats, winner, and delta.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)

        def _stats(version: str) -> dict[str, Any]:
            snaps = [
                snap
                for snap in self._snapshots.get(tenant_id, [])
                if snap.model_id == model_id
                and snap.model_version == version
                and snap.metric_name == metric_name
                and snap.recorded_at >= cutoff
            ]
            if not snaps:
                return {"version": version, "n_snapshots": 0, "mean": None, "min": None, "max": None}
            values = [snap.metric_value for snap in snaps]
            mean_val = sum(values) / len(values)
            return {
                "version": version,
                "n_snapshots": len(values),
                "mean": round(mean_val, 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }

        stats_a = _stats(version_a)
        stats_b = _stats(version_b)

        mean_a = stats_a.get("mean")
        mean_b = stats_b.get("mean")

        if mean_a is None or mean_b is None:
            winner = "insufficient_data"
            delta = None
        else:
            delta = round(mean_b - mean_a, 6)
            # For error_rate, lower is better
            if metric_name == "error_rate":
                winner = version_b if mean_b < mean_a else version_a
            else:
                winner = version_b if mean_b > mean_a else version_a

        return {
            "model_id": model_id,
            "metric_name": metric_name,
            "window_days": window_days,
            "version_a": stats_a,
            "version_b": stats_b,
            "winner": winner,
            "delta_b_minus_a": delta,
        }

    async def get_segment_breakdown(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        metric_name: str = "accuracy",
        window_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Return performance metrics broken down by user segment and data type.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            metric_name: Metric to slice.
            window_days: Lookback window in days.

        Returns:
            List of segment performance dicts sorted by mean metric value descending.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        snapshots = [
            snap
            for snap in self._snapshots.get(tenant_id, [])
            if snap.model_id == model_id
            and snap.metric_name == metric_name
            and snap.recorded_at >= cutoff
        ]

        group_values: dict[tuple[str | None, str | None], list[float]] = defaultdict(list)
        for snap in snapshots:
            group_values[(snap.segment, snap.data_type)].append(snap.metric_value)

        breakdown = []
        for (segment, data_type), values in group_values.items():
            mean_val = sum(values) / len(values)
            breakdown.append(
                {
                    "segment": segment,
                    "data_type": data_type,
                    "n_snapshots": len(values),
                    "mean": round(mean_val, 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                }
            )

        breakdown.sort(key=lambda segment: segment["mean"], reverse=True)
        return breakdown

    async def compute_improvement_velocity(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        metric_name: str = "accuracy",
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Compute the rate of improvement per day over the observation window.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            metric_name: Metric to compute velocity for.
            window_days: Lookback window in days.

        Returns:
            Dict with improvement_per_day, total_delta, trend_direction, n_samples.
        """
        trend = await self.get_accuracy_trend(
            tenant_id=tenant_id,
            model_id=model_id,
            metric_name=metric_name,
            window_days=window_days,
        )

        data_points = trend.get("data_points", [])
        total_delta = trend.get("delta")

        if total_delta is None or len(data_points) < 2:
            return {
                "model_id": model_id,
                "metric_name": metric_name,
                "improvement_per_day": None,
                "total_delta": total_delta,
                "trend_direction": trend.get("trend_direction"),
                "n_samples": len(data_points),
            }

        improvement_per_day = total_delta / window_days
        return {
            "model_id": model_id,
            "metric_name": metric_name,
            "improvement_per_day": round(improvement_per_day, 8),
            "total_delta": total_delta,
            "trend_direction": trend.get("trend_direction"),
            "n_samples": len(data_points),
            "window_days": window_days,
        }

    async def generate_performance_report(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        window_days: int = 30,
        include_versions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive performance report for a model.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier.
            window_days: Lookback window in days.
            include_versions: Optional list of specific versions to compare.

        Returns:
            Comprehensive performance report dict.
        """
        report: dict[str, Any] = {
            "model_id": model_id,
            "tenant_id": str(tenant_id),
            "window_days": window_days,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        # Trend for each metric
        metric_summaries = {}
        for metric_name in VALID_METRICS:
            trend = await self.get_accuracy_trend(
                tenant_id=tenant_id,
                model_id=model_id,
                metric_name=metric_name,
                window_days=window_days,
            )
            if trend["data_points"]:
                metric_summaries[metric_name] = {
                    "current_value": trend["current_value"],
                    "delta": trend["delta"],
                    "trend_direction": trend["trend_direction"],
                    "n_snapshots": len(trend["data_points"]),
                }

        report["metrics"] = metric_summaries

        # Improvement velocity for primary metrics
        velocity = await self.compute_improvement_velocity(
            tenant_id=tenant_id,
            model_id=model_id,
            metric_name="accuracy",
            window_days=window_days,
        )
        report["improvement_velocity"] = velocity

        # Segment breakdown
        report["segment_breakdown"] = await self.get_segment_breakdown(
            tenant_id=tenant_id,
            model_id=model_id,
            metric_name="accuracy",
            window_days=window_days,
        )

        logger.info(
            "Performance report generated",
            tenant_id=str(tenant_id),
            model_id=model_id,
            window_days=window_days,
            metrics_tracked=list(metric_summaries.keys()),
        )

        return report

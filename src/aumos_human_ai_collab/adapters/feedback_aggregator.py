"""Feedback aggregator adapter for the Human-AI Collaboration service.

Collects user feedback records (rating, text, category) and aggregates
them into trend reports per model, feature, and time window. Sentiment
analysis on text feedback drives trend direction detection.
"""

import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Sentiment word lists for basic polarity scoring
_POSITIVE_WORDS = frozenset(
    {
        "good", "great", "excellent", "accurate", "helpful", "correct",
        "perfect", "love", "useful", "clear", "fast", "reliable", "better",
        "improved", "works", "right", "nice", "smooth", "happy", "satisfied",
    }
)
_NEGATIVE_WORDS = frozenset(
    {
        "bad", "wrong", "incorrect", "slow", "confusing", "error", "broken",
        "terrible", "awful", "worse", "failed", "missing", "inaccurate",
        "unhelpful", "useless", "frustrating", "buggy", "poor", "hate",
    }
)

# Valid feedback categories
VALID_CATEGORIES = frozenset(
    {"accuracy", "speed", "usability", "relevance", "safety", "general"}
)

# Minimum samples required to compute a trend direction
_TREND_MIN_SAMPLES = 3


class FeedbackRecord:
    """In-memory feedback record produced by FeedbackAggregator."""

    __slots__ = (
        "record_id",
        "tenant_id",
        "model_id",
        "feature",
        "rating",
        "text",
        "category",
        "sentiment_score",
        "submitted_by",
        "created_at",
    )

    def __init__(
        self,
        record_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_id: str,
        feature: str,
        rating: int,
        text: str | None,
        category: str,
        sentiment_score: float,
        submitted_by: uuid.UUID | None,
        created_at: datetime,
    ) -> None:
        self.record_id = record_id
        self.tenant_id = tenant_id
        self.model_id = model_id
        self.feature = feature
        self.rating = rating
        self.text = text
        self.category = category
        self.sentiment_score = sentiment_score
        self.submitted_by = submitted_by
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for export.

        Returns:
            Dict representation of this feedback record.
        """
        return {
            "record_id": str(self.record_id),
            "tenant_id": str(self.tenant_id),
            "model_id": self.model_id,
            "feature": self.feature,
            "rating": self.rating,
            "text": self.text,
            "category": self.category,
            "sentiment_score": self.sentiment_score,
            "submitted_by": str(self.submitted_by) if self.submitted_by else None,
            "created_at": self.created_at.isoformat(),
        }


class FeedbackAggregator:
    """User feedback collection, aggregation, sentiment analysis, and trend detection.

    Stores feedback in-memory and provides aggregation by model, feature, and
    time window. Sentiment is scored using a lightweight word-polarity heuristic.
    Top issues are identified from the lowest-rated, highest-volume categories.
    """

    def __init__(self) -> None:
        """Initialise with empty storage."""
        # tenant_id -> list of FeedbackRecord
        self._records: dict[uuid.UUID, list[FeedbackRecord]] = defaultdict(list)

    def _score_sentiment(self, text: str | None) -> float:
        """Compute a polarity score from free-text feedback.

        Uses a simple word-matching heuristic. Returns a value in [-1.0, 1.0]
        where -1.0 is strongly negative, 0.0 is neutral, 1.0 is strongly positive.

        Args:
            text: Raw feedback text.

        Returns:
            Sentiment score in [-1.0, 1.0].
        """
        if not text:
            return 0.0

        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return 0.0

        positive_hits = sum(1 for word in words if word in _POSITIVE_WORDS)
        negative_hits = sum(1 for word in words if word in _NEGATIVE_WORDS)
        total = positive_hits + negative_hits

        if total == 0:
            return 0.0

        return (positive_hits - negative_hits) / total

    async def store_feedback(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        feature: str,
        rating: int,
        category: str,
        text: str | None = None,
        submitted_by: uuid.UUID | None = None,
    ) -> FeedbackRecord:
        """Store a single feedback record.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Model or component the feedback targets.
            feature: Specific feature being rated.
            rating: Integer rating (1–5).
            category: Feedback category (accuracy, speed, usability, etc.).
            text: Optional free-text comment.
            submitted_by: Optional submitter UUID.

        Returns:
            Stored FeedbackRecord.

        Raises:
            ValueError: If rating is outside 1–5 or category is invalid.
        """
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5. Got: {rating}")

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Valid: {VALID_CATEGORIES}"
            )

        sentiment_score = self._score_sentiment(text)
        record = FeedbackRecord(
            record_id=uuid.uuid4(),
            tenant_id=tenant_id,
            model_id=model_id,
            feature=feature,
            rating=rating,
            text=text,
            category=category,
            sentiment_score=sentiment_score,
            submitted_by=submitted_by,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._records[tenant_id].append(record)

        logger.info(
            "Feedback stored",
            tenant_id=str(tenant_id),
            model_id=model_id,
            feature=feature,
            rating=rating,
            category=category,
            sentiment_score=sentiment_score,
        )

        return record

    async def aggregate_by_model(
        self,
        tenant_id: uuid.UUID,
        model_id: str,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Aggregate feedback metrics for a specific model over a time window.

        Args:
            tenant_id: Requesting tenant.
            model_id: Model identifier to aggregate.
            window_days: Lookback window in days.

        Returns:
            Dict with average_rating, total_count, sentiment_avg,
            category_breakdown, and trend_direction.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        records = [
            record
            for record in self._records.get(tenant_id, [])
            if record.model_id == model_id and record.created_at >= cutoff
        ]

        return self._build_aggregate(records, window_days)

    async def aggregate_by_feature(
        self,
        tenant_id: uuid.UUID,
        feature: str,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Aggregate feedback metrics for a specific feature over a time window.

        Args:
            tenant_id: Requesting tenant.
            feature: Feature name to aggregate.
            window_days: Lookback window in days.

        Returns:
            Dict with average_rating, total_count, sentiment_avg,
            category_breakdown, and trend_direction.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        records = [
            record
            for record in self._records.get(tenant_id, [])
            if record.feature == feature and record.created_at >= cutoff
        ]

        return self._build_aggregate(records, window_days)

    def _build_aggregate(
        self, records: list[FeedbackRecord], window_days: int
    ) -> dict[str, Any]:
        """Compute aggregate statistics from a list of records.

        Args:
            records: Filtered feedback records.
            window_days: Lookback window used (informational).

        Returns:
            Aggregate statistics dict.
        """
        if not records:
            return {
                "total_count": 0,
                "average_rating": None,
                "sentiment_avg": None,
                "category_breakdown": {},
                "trend_direction": "insufficient_data",
                "window_days": window_days,
            }

        total_count = len(records)
        average_rating = sum(r.rating for r in records) / total_count
        sentiment_avg = sum(r.sentiment_score for r in records) / total_count

        category_breakdown: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "rating_sum": 0}
        )
        for record in records:
            category_breakdown[record.category]["count"] += 1
            category_breakdown[record.category]["rating_sum"] += record.rating

        category_stats: dict[str, Any] = {}
        for category, data in category_breakdown.items():
            count = data["count"]
            category_stats[category] = {
                "count": count,
                "average_rating": data["rating_sum"] / count,
            }

        trend_direction = self._compute_trend(records)

        return {
            "total_count": total_count,
            "average_rating": round(average_rating, 3),
            "sentiment_avg": round(sentiment_avg, 3),
            "category_breakdown": category_stats,
            "trend_direction": trend_direction,
            "window_days": window_days,
        }

    def _compute_trend(self, records: list[FeedbackRecord]) -> str:
        """Determine trend direction by comparing first-half vs second-half average ratings.

        Args:
            records: Feedback records sorted by recency (newest last).

        Returns:
            "improving", "degrading", "stable", or "insufficient_data".
        """
        sorted_records = sorted(records, key=lambda r: r.created_at)
        if len(sorted_records) < _TREND_MIN_SAMPLES:
            return "insufficient_data"

        midpoint = len(sorted_records) // 2
        first_half = sorted_records[:midpoint]
        second_half = sorted_records[midpoint:]

        first_avg = sum(r.rating for r in first_half) / len(first_half)
        second_avg = sum(r.rating for r in second_half) / len(second_half)

        delta = second_avg - first_avg
        if delta > 0.2:
            return "improving"
        if delta < -0.2:
            return "degrading"
        return "stable"

    async def get_top_issues(
        self,
        tenant_id: uuid.UUID,
        window_days: int = 7,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Identify the top issues by lowest average rating and highest volume.

        Args:
            tenant_id: Requesting tenant.
            window_days: Lookback window in days.
            top_n: Number of top issues to return.

        Returns:
            List of issue dicts sorted by severity (low rating, high volume).
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        records = [
            record
            for record in self._records.get(tenant_id, [])
            if record.created_at >= cutoff
        ]

        # Group by (model_id, feature, category)
        groups: dict[tuple[str, str, str], list[FeedbackRecord]] = defaultdict(list)
        for record in records:
            key = (record.model_id, record.feature, record.category)
            groups[key].append(record)

        issues = []
        for (model_id, feature, category), group_records in groups.items():
            count = len(group_records)
            avg_rating = sum(r.rating for r in group_records) / count
            avg_sentiment = sum(r.sentiment_score for r in group_records) / count
            # Severity: lower rating + higher volume = worse
            severity_score = (5.0 - avg_rating) * (1.0 + (count / max(len(records), 1)))
            issues.append(
                {
                    "model_id": model_id,
                    "feature": feature,
                    "category": category,
                    "count": count,
                    "average_rating": round(avg_rating, 3),
                    "average_sentiment": round(avg_sentiment, 3),
                    "severity_score": round(severity_score, 3),
                }
            )

        issues.sort(key=lambda issue: issue["severity_score"], reverse=True)
        return issues[:top_n]

    async def get_volume_stats(
        self,
        tenant_id: uuid.UUID,
        model_id: str | None = None,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Return feedback volume statistics over a time window.

        Args:
            tenant_id: Requesting tenant.
            model_id: Optional filter to a specific model.
            window_days: Lookback window in days.

        Returns:
            Dict with total volume, daily_avg, peak_day, and daily breakdown.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        records = [
            record
            for record in self._records.get(tenant_id, [])
            if record.created_at >= cutoff
            and (model_id is None or record.model_id == model_id)
        ]

        if not records:
            return {
                "total_volume": 0,
                "daily_avg": 0.0,
                "peak_day": None,
                "daily_breakdown": {},
            }

        daily_counts: dict[str, int] = defaultdict(int)
        for record in records:
            day_key = record.created_at.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1

        total_volume = len(records)
        daily_avg = total_volume / window_days
        peak_day = max(daily_counts, key=lambda day: daily_counts[day])

        return {
            "total_volume": total_volume,
            "daily_avg": round(daily_avg, 2),
            "peak_day": peak_day,
            "daily_breakdown": dict(daily_counts),
        }

    async def export_records(
        self,
        tenant_id: uuid.UUID,
        model_id: str | None = None,
        feature: str | None = None,
        category: str | None = None,
        window_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Export feedback records as a list of dicts for analysis.

        Args:
            tenant_id: Requesting tenant.
            model_id: Optional model filter.
            feature: Optional feature filter.
            category: Optional category filter.
            window_days: Optional lookback window. None returns all records.

        Returns:
            List of serialised FeedbackRecord dicts.
        """
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=window_days)
            if window_days is not None
            else None
        )

        records = [
            record
            for record in self._records.get(tenant_id, [])
            if (model_id is None or record.model_id == model_id)
            and (feature is None or record.feature == feature)
            and (category is None or record.category == category)
            and (cutoff is None or record.created_at >= cutoff)
        ]

        records.sort(key=lambda r: r.created_at, reverse=True)
        return [record.to_dict() for record in records]

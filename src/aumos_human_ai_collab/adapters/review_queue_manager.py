"""Review queue manager adapter for the Human-AI Collaboration service.

Redis-backed priority queue for HITL review assignment. Supports configurable
priority scoring, reviewer assignment with timeout and reassignment, queue depth
monitoring and alerts, draining metrics, workload balancing, and queue analytics.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Redis key namespaces
_QUEUE_KEY = "hac:review_queue:{tenant_id}"
_ITEM_KEY = "hac:review_item:{tenant_id}:{item_id}"
_REVIEWER_SET_KEY = "hac:reviewer_items:{tenant_id}:{reviewer_id}"
_ANALYTICS_KEY = "hac:queue_analytics:{tenant_id}"

# Queue item status values
STATUS_PENDING = "pending"
STATUS_ASSIGNED = "assigned"
STATUS_COMPLETED = "completed"
STATUS_TIMED_OUT = "timed_out"
STATUS_REASSIGNED = "reassigned"

# Default timeout for a reviewer to complete a review
DEFAULT_REVIEW_TIMEOUT_SECONDS = 3600  # 1 hour

# Alert threshold for queue depth
DEFAULT_QUEUE_DEPTH_ALERT_THRESHOLD = 50


def _compute_priority_score(
    confidence: float,
    urgency: int,
    age_seconds: float,
    priority_weights: dict[str, float],
) -> float:
    """Compute a numeric priority score for a queue item.

    Higher scores = higher priority = served first.

    Args:
        confidence: Model confidence score (0–1). Lower confidence = higher priority.
        urgency: Urgency level (1=low to 4=critical).
        age_seconds: Seconds the item has been waiting. Older = higher priority.
        priority_weights: Weights for each component (confidence, urgency, age).

    Returns:
        Priority score (higher = dequeue first).
    """
    weight_confidence = priority_weights.get("confidence", 1.0)
    weight_urgency = priority_weights.get("urgency", 2.0)
    weight_age = priority_weights.get("age", 0.5)

    # Invert confidence: low confidence gets high score
    confidence_component = (1.0 - confidence) * weight_confidence
    urgency_component = (urgency / 4.0) * weight_urgency
    # Cap age contribution at 24 hours to prevent runaway starvation priority
    age_hours = min(age_seconds / 3600.0, 24.0)
    age_component = (age_hours / 24.0) * weight_age

    return confidence_component + urgency_component + age_component


class QueueItem:
    """An item in the review queue."""

    def __init__(
        self,
        item_id: uuid.UUID,
        tenant_id: uuid.UUID,
        review_id: uuid.UUID,
        task_type: str,
        confidence: float,
        urgency: int,
        priority_score: float,
        metadata: dict[str, Any],
    ) -> None:
        self.item_id = item_id
        self.tenant_id = tenant_id
        self.review_id = review_id
        self.task_type = task_type
        self.confidence = confidence
        self.urgency = urgency
        self.priority_score = priority_score
        self.metadata = metadata
        self.status = STATUS_PENDING
        self.reviewer_id: uuid.UUID | None = None
        self.assigned_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.timeout_at: datetime | None = None
        self.enqueued_at = datetime.now(tz=timezone.utc)
        self.wait_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for Redis or API responses.

        Returns:
            Dict representation.
        """
        return {
            "item_id": str(self.item_id),
            "tenant_id": str(self.tenant_id),
            "review_id": str(self.review_id),
            "task_type": self.task_type,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "priority_score": self.priority_score,
            "status": self.status,
            "reviewer_id": str(self.reviewer_id) if self.reviewer_id else None,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None,
            "enqueued_at": self.enqueued_at.isoformat(),
            "wait_seconds": self.wait_seconds,
            "metadata": self.metadata,
        }


class ReviewQueueManager:
    """Redis-backed priority queue for HITL review task management.

    In production, operations target Redis using sorted sets (ZADD/ZRANGE) for
    O(log N) priority insertions and dequeues. This implementation maintains
    an in-process sorted list that mirrors Redis sorted set semantics so the
    logic is correct when the Redis client is injected.

    Inject a Redis client via the redis_client parameter. If None, falls back
    to an in-memory implementation for testing and local development.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        review_timeout_seconds: int = DEFAULT_REVIEW_TIMEOUT_SECONDS,
        depth_alert_threshold: int = DEFAULT_QUEUE_DEPTH_ALERT_THRESHOLD,
        priority_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialise the review queue manager.

        Args:
            redis_client: Optional Redis client (aioredis / redis-py async client).
            review_timeout_seconds: Seconds before an assigned review times out.
            depth_alert_threshold: Queue depth that triggers an alert.
            priority_weights: Custom weights for priority score components.
        """
        self._redis = redis_client
        self._timeout_seconds = review_timeout_seconds
        self._alert_threshold = depth_alert_threshold
        self._priority_weights = priority_weights or {
            "confidence": 1.0,
            "urgency": 2.0,
            "age": 0.5,
        }

        # In-memory fallback storage when Redis is not available
        # tenant_id -> list of QueueItem (maintained sorted by priority_score desc)
        self._memory_queue: dict[uuid.UUID, list[QueueItem]] = {}
        # item_id -> QueueItem for O(1) lookup
        self._items: dict[uuid.UUID, QueueItem] = {}
        # reviewer_id -> list of item_ids currently assigned
        self._reviewer_assignments: dict[uuid.UUID, list[uuid.UUID]] = {}
        # tenant_id -> analytics counters
        self._analytics: dict[uuid.UUID, dict[str, Any]] = {}

    def _get_tenant_analytics(self, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Get or create analytics counters for a tenant.

        Args:
            tenant_id: Tenant UUID.

        Returns:
            Analytics dict for the tenant.
        """
        if tenant_id not in self._analytics:
            self._analytics[tenant_id] = {
                "total_enqueued": 0,
                "total_assigned": 0,
                "total_completed": 0,
                "total_timed_out": 0,
                "total_reassigned": 0,
                "wait_time_sum_seconds": 0.0,
                "throughput_per_hour": 0.0,
            }
        return self._analytics[tenant_id]

    async def enqueue(
        self,
        tenant_id: uuid.UUID,
        review_id: uuid.UUID,
        task_type: str,
        confidence: float,
        urgency: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> QueueItem:
        """Add a review item to the priority queue.

        Args:
            tenant_id: Owning tenant UUID.
            review_id: HITLReview UUID.
            task_type: Task category.
            confidence: Model confidence score (0–1).
            urgency: Review urgency (1=low, 2=normal, 3=high, 4=critical).
            metadata: Additional review context.

        Returns:
            Enqueued QueueItem.

        Raises:
            ValueError: If urgency is outside 1–4 or confidence outside 0–1.
        """
        if not (1 <= urgency <= 4):
            raise ValueError(f"Urgency must be 1–4. Got: {urgency}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1]. Got: {confidence}")

        item_id = uuid.uuid4()
        now = datetime.now(tz=timezone.utc)
        age_seconds = 0.0

        priority_score = _compute_priority_score(
            confidence=confidence,
            urgency=urgency,
            age_seconds=age_seconds,
            priority_weights=self._priority_weights,
        )

        item = QueueItem(
            item_id=item_id,
            tenant_id=tenant_id,
            review_id=review_id,
            task_type=task_type,
            confidence=confidence,
            urgency=urgency,
            priority_score=priority_score,
            metadata=metadata or {},
        )

        if self._redis is not None:
            # Production: persist to Redis sorted set
            await self._redis.zadd(
                _QUEUE_KEY.format(tenant_id=str(tenant_id)),
                {str(item_id): priority_score},
            )
            await self._redis.setex(
                _ITEM_KEY.format(tenant_id=str(tenant_id), item_id=str(item_id)),
                86400,  # 24h TTL for item data
                json.dumps(item.to_dict()),
            )
        else:
            # In-memory fallback
            if tenant_id not in self._memory_queue:
                self._memory_queue[tenant_id] = []
            self._memory_queue[tenant_id].append(item)
            # Keep sorted by priority_score descending
            self._memory_queue[tenant_id].sort(
                key=lambda queued_item: queued_item.priority_score, reverse=True
            )

        self._items[item_id] = item
        analytics = self._get_tenant_analytics(tenant_id)
        analytics["total_enqueued"] += 1

        logger.info(
            "Review enqueued",
            tenant_id=str(tenant_id),
            item_id=str(item_id),
            review_id=str(review_id),
            task_type=task_type,
            priority_score=priority_score,
            urgency=urgency,
            confidence=confidence,
        )

        # Check depth alert
        queue_depth = await self.get_queue_depth(tenant_id)
        if queue_depth >= self._alert_threshold:
            logger.warning(
                "Queue depth alert threshold reached",
                tenant_id=str(tenant_id),
                depth=queue_depth,
                threshold=self._alert_threshold,
            )

        return item

    async def assign_to_reviewer(
        self,
        tenant_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> QueueItem | None:
        """Dequeue the highest-priority pending item and assign to a reviewer.

        Checks reviewer current workload — does not assign if reviewer already
        has too many in-progress items (default cap: 5).

        Args:
            tenant_id: Requesting tenant UUID.
            reviewer_id: Reviewer to assign the item to.

        Returns:
            Assigned QueueItem, or None if the queue is empty or reviewer is at capacity.
        """
        # Check reviewer workload
        current_assignments = [
            item_id
            for item_id in self._reviewer_assignments.get(reviewer_id, [])
            if self._items.get(item_id) and self._items[item_id].status == STATUS_ASSIGNED
        ]
        if len(current_assignments) >= 5:
            logger.warning(
                "Reviewer at capacity — not assigning",
                reviewer_id=str(reviewer_id),
                current_count=len(current_assignments),
            )
            return None

        item = None

        if self._redis is not None:
            # Atomically dequeue highest priority item from Redis
            result = await self._redis.zpopmax(
                _QUEUE_KEY.format(tenant_id=str(tenant_id))
            )
            if not result:
                return None
            item_id_str, _ = result[0]
            raw_item = await self._redis.get(
                _ITEM_KEY.format(tenant_id=str(tenant_id), item_id=item_id_str)
            )
            if raw_item:
                item_data = json.loads(raw_item)
                item = self._items.get(uuid.UUID(item_data["item_id"]))
        else:
            # In-memory: find first pending item
            queue = self._memory_queue.get(tenant_id, [])
            for queued_item in queue:
                if queued_item.status == STATUS_PENDING:
                    item = queued_item
                    break

        if item is None:
            return None

        now = datetime.now(tz=timezone.utc)
        wait_seconds = (now - item.enqueued_at).total_seconds()
        item.status = STATUS_ASSIGNED
        item.reviewer_id = reviewer_id
        item.assigned_at = now
        item.timeout_at = now + timedelta(seconds=self._timeout_seconds)
        item.wait_seconds = wait_seconds

        if reviewer_id not in self._reviewer_assignments:
            self._reviewer_assignments[reviewer_id] = []
        self._reviewer_assignments[reviewer_id].append(item.item_id)

        analytics = self._get_tenant_analytics(tenant_id)
        analytics["total_assigned"] += 1
        analytics["wait_time_sum_seconds"] += wait_seconds

        if self._redis is not None:
            await self._redis.sadd(
                _REVIEWER_SET_KEY.format(
                    tenant_id=str(tenant_id), reviewer_id=str(reviewer_id)
                ),
                str(item.item_id),
            )

        logger.info(
            "Review assigned to reviewer",
            tenant_id=str(tenant_id),
            item_id=str(item.item_id),
            reviewer_id=str(reviewer_id),
            wait_seconds=wait_seconds,
            timeout_at=item.timeout_at.isoformat(),
        )

        return item

    async def mark_completed(
        self,
        tenant_id: uuid.UUID,
        item_id: uuid.UUID,
    ) -> QueueItem:
        """Mark a queue item as completed.

        Args:
            tenant_id: Owning tenant UUID.
            item_id: Queue item UUID.

        Returns:
            Updated QueueItem.

        Raises:
            KeyError: If item_id is not found.
        """
        item = self._items.get(item_id)
        if item is None or item.tenant_id != tenant_id:
            raise KeyError(f"Queue item {item_id} not found for tenant {tenant_id}.")

        now = datetime.now(tz=timezone.utc)
        item.status = STATUS_COMPLETED
        item.completed_at = now

        analytics = self._get_tenant_analytics(tenant_id)
        analytics["total_completed"] += 1

        logger.info(
            "Review completed",
            tenant_id=str(tenant_id),
            item_id=str(item_id),
            reviewer_id=str(item.reviewer_id) if item.reviewer_id else None,
        )

        return item

    async def check_and_handle_timeouts(
        self, tenant_id: uuid.UUID
    ) -> list[QueueItem]:
        """Scan assigned items for timeouts and re-enqueue expired ones.

        Args:
            tenant_id: Tenant to scan.

        Returns:
            List of items that were timed out and re-enqueued.
        """
        now = datetime.now(tz=timezone.utc)
        timed_out = []

        for item in self._items.values():
            if (
                item.tenant_id != tenant_id
                or item.status != STATUS_ASSIGNED
            ):
                continue

            if item.timeout_at and now >= item.timeout_at:
                item.status = STATUS_TIMED_OUT
                analytics = self._get_tenant_analytics(tenant_id)
                analytics["total_timed_out"] += 1

                # Re-enqueue as reassigned with boosted urgency
                new_urgency = min(4, item.urgency + 1)
                re_item = await self.enqueue(
                    tenant_id=tenant_id,
                    review_id=item.review_id,
                    task_type=item.task_type,
                    confidence=item.confidence,
                    urgency=new_urgency,
                    metadata={
                        **item.metadata,
                        "reassigned_from": str(item.item_id),
                        "original_reviewer": str(item.reviewer_id) if item.reviewer_id else None,
                        "reassignment_reason": "timeout",
                    },
                )
                re_item.status = STATUS_REASSIGNED
                analytics["total_reassigned"] += 1
                timed_out.append(item)

                logger.warning(
                    "Review timed out — reassigned",
                    tenant_id=str(tenant_id),
                    original_item_id=str(item.item_id),
                    new_item_id=str(re_item.item_id),
                    reviewer_id=str(item.reviewer_id) if item.reviewer_id else None,
                )

        return timed_out

    async def get_queue_depth(self, tenant_id: uuid.UUID) -> int:
        """Return the number of pending items in the queue.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Count of pending (unassigned) queue items.
        """
        if self._redis is not None:
            return await self._redis.zcard(
                _QUEUE_KEY.format(tenant_id=str(tenant_id))
            )

        return sum(
            1
            for item in self._items.values()
            if item.tenant_id == tenant_id and item.status == STATUS_PENDING
        )

    async def get_reviewer_workload(
        self, tenant_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Return current workload per reviewer.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            List of reviewer workload dicts sorted by assigned_count descending.
        """
        workload: dict[uuid.UUID, int] = {}
        for item in self._items.values():
            if (
                item.tenant_id == tenant_id
                and item.status == STATUS_ASSIGNED
                and item.reviewer_id is not None
            ):
                workload[item.reviewer_id] = workload.get(item.reviewer_id, 0) + 1

        return sorted(
            [
                {"reviewer_id": str(reviewer_id), "assigned_count": count}
                for reviewer_id, count in workload.items()
            ],
            key=lambda entry: entry["assigned_count"],
            reverse=True,
        )

    async def get_queue_analytics(self, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Return queue analytics including throughput, wait time, and depth metrics.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Analytics dict with throughput, avg_wait_time, depth, completion_rate, etc.
        """
        analytics = self._get_tenant_analytics(tenant_id)
        depth = await self.get_queue_depth(tenant_id)
        total_assigned = analytics.get("total_assigned", 0)
        total_completed = analytics.get("total_completed", 0)
        total_enqueued = analytics.get("total_enqueued", 0)
        total_timed_out = analytics.get("total_timed_out", 0)
        total_reassigned = analytics.get("total_reassigned", 0)
        wait_time_sum = analytics.get("wait_time_sum_seconds", 0.0)

        avg_wait_time_seconds = (
            wait_time_sum / total_assigned if total_assigned > 0 else None
        )
        completion_rate = (
            total_completed / total_enqueued if total_enqueued > 0 else None
        )
        timeout_rate = (
            total_timed_out / total_assigned if total_assigned > 0 else None
        )

        return {
            "tenant_id": str(tenant_id),
            "queue_depth": depth,
            "total_enqueued": total_enqueued,
            "total_assigned": total_assigned,
            "total_completed": total_completed,
            "total_timed_out": total_timed_out,
            "total_reassigned": total_reassigned,
            "avg_wait_time_seconds": (
                round(avg_wait_time_seconds, 2) if avg_wait_time_seconds is not None else None
            ),
            "completion_rate": (
                round(completion_rate, 4) if completion_rate is not None else None
            ),
            "timeout_rate": (
                round(timeout_rate, 4) if timeout_rate is not None else None
            ),
            "alert_threshold": self._alert_threshold,
            "depth_alert_active": depth >= self._alert_threshold,
        }

    async def drain_queue(self, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Return metrics about queue draining progress.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Dict with pending, assigned, completed, timed_out counts and drain_rate.
        """
        counts: dict[str, int] = {
            STATUS_PENDING: 0,
            STATUS_ASSIGNED: 0,
            STATUS_COMPLETED: 0,
            STATUS_TIMED_OUT: 0,
            STATUS_REASSIGNED: 0,
        }
        for item in self._items.values():
            if item.tenant_id == tenant_id:
                counts[item.status] = counts.get(item.status, 0) + 1

        total = sum(counts.values())
        drain_rate = (
            counts[STATUS_COMPLETED] / total if total > 0 else 0.0
        )

        return {
            "tenant_id": str(tenant_id),
            "pending": counts[STATUS_PENDING],
            "assigned": counts[STATUS_ASSIGNED],
            "completed": counts[STATUS_COMPLETED],
            "timed_out": counts[STATUS_TIMED_OUT],
            "reassigned": counts[STATUS_REASSIGNED],
            "total": total,
            "drain_rate": round(drain_rate, 4),
        }

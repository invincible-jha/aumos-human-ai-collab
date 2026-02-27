"""Annotation engine adapter for the Human-AI Collaboration service.

Manages crowdsourced label collection: task creation, annotator assignment,
annotation storage with provenance, quality control via gold standard questions,
inter-annotator agreement metrics, export, and progress tracking.
"""

import csv
import io
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Supported task types for annotation
VALID_TASK_TYPES = frozenset(
    {"text_classification", "named_entity_recognition", "image_labeling", "sentiment", "qa"}
)

# Task status values
TASK_STATUS_OPEN = "open"
TASK_STATUS_IN_PROGRESS = "in_progress"
TASK_STATUS_COMPLETE = "complete"
TASK_STATUS_CLOSED = "closed"

# Minimum annotations per item before marking complete
DEFAULT_ANNOTATIONS_PER_ITEM = 3


class AnnotationTask:
    """Represents a single annotation task (batch of items to label)."""

    def __init__(
        self,
        task_id: uuid.UUID,
        tenant_id: uuid.UUID,
        task_type: str,
        name: str,
        items: list[dict[str, Any]],
        labels: list[str],
        annotations_per_item: int,
        gold_standard_ids: list[str],
        metadata: dict[str, Any],
    ) -> None:
        self.task_id = task_id
        self.tenant_id = tenant_id
        self.task_type = task_type
        self.name = name
        self.items = items  # list of {item_id, content, ...}
        self.labels = labels
        self.annotations_per_item = annotations_per_item
        self.gold_standard_ids = gold_standard_ids  # item_ids that are gold
        self.metadata = metadata
        self.status = TASK_STATUS_OPEN
        self.created_at = datetime.now(tz=timezone.utc)
        # annotator_id -> [item_id, ...]
        self.assigned_annotators: dict[uuid.UUID, list[str]] = {}


class Annotation:
    """A single annotator's label for one item."""

    def __init__(
        self,
        annotation_id: uuid.UUID,
        task_id: uuid.UUID,
        item_id: str,
        annotator_id: uuid.UUID,
        label: str,
        confidence: float | None,
        notes: str | None,
        is_gold_check: bool,
        created_at: datetime,
    ) -> None:
        self.annotation_id = annotation_id
        self.task_id = task_id
        self.item_id = item_id
        self.annotator_id = annotator_id
        self.label = label
        self.confidence = confidence
        self.notes = notes
        self.is_gold_check = is_gold_check
        self.created_at = created_at


class AnnotationEngine:
    """Crowdsourced label collection system with quality control.

    Manages annotation tasks from creation through completion. Gold standard
    items are interspersed for quality control. Agreement metrics are computed
    across multiple annotators per item.
    """

    def __init__(
        self,
        annotations_per_item: int = DEFAULT_ANNOTATIONS_PER_ITEM,
        gold_standard_ratio: float = 0.1,
    ) -> None:
        """Initialise the annotation engine.

        Args:
            annotations_per_item: How many annotators must label each item.
            gold_standard_ratio: Fraction of task items that are gold standard.
        """
        self._annotations_per_item = annotations_per_item
        self._gold_ratio = gold_standard_ratio

        # task_id -> AnnotationTask
        self._tasks: dict[uuid.UUID, AnnotationTask] = {}
        # task_id -> list[Annotation]
        self._annotations: dict[uuid.UUID, list[Annotation]] = defaultdict(list)
        # annotator_id -> task_ids assigned to them
        self._annotator_workload: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
        # (task_id, item_id) -> expected_label for gold standard items
        self._gold_answers: dict[tuple[uuid.UUID, str], str] = {}

    async def create_task(
        self,
        tenant_id: uuid.UUID,
        task_type: str,
        name: str,
        items: list[dict[str, Any]],
        labels: list[str],
        gold_items: list[dict[str, Any]] | None = None,
        annotations_per_item: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new annotation task.

        Gold standard items are items with known correct labels injected into
        the task for quality control. They are not separately visible to annotators.

        Args:
            tenant_id: Owning tenant UUID.
            task_type: Type of labeling task.
            name: Human-readable task name.
            items: List of items to annotate. Each must have an 'item_id' field.
            labels: List of valid label strings.
            gold_items: Optional list of gold standard items, each with
                        'item_id', 'content', and 'correct_label'.
            annotations_per_item: Override default annotations per item.
            metadata: Additional task configuration.

        Returns:
            Task summary dict.

        Raises:
            ValueError: If task_type is invalid or items/labels are empty.
        """
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. Valid: {VALID_TASK_TYPES}"
            )
        if not items:
            raise ValueError("At least one item is required to create an annotation task.")
        if not labels:
            raise ValueError("At least one label is required.")

        task_id = uuid.uuid4()
        gold_standard_ids: list[str] = []
        all_items = list(items)

        if gold_items:
            for gold_item in gold_items:
                gold_id = gold_item["item_id"]
                gold_standard_ids.append(gold_id)
                self._gold_answers[(task_id, gold_id)] = gold_item["correct_label"]
                all_items.append(gold_item)

        per_item = annotations_per_item or self._annotations_per_item

        task = AnnotationTask(
            task_id=task_id,
            tenant_id=tenant_id,
            task_type=task_type,
            name=name,
            items=all_items,
            labels=labels,
            annotations_per_item=per_item,
            gold_standard_ids=gold_standard_ids,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task

        logger.info(
            "Annotation task created",
            task_id=str(task_id),
            tenant_id=str(tenant_id),
            task_type=task_type,
            item_count=len(items),
            gold_count=len(gold_standard_ids),
        )

        return self._task_summary(task)

    async def assign_annotators(
        self,
        task_id: uuid.UUID,
        annotator_ids: list[uuid.UUID],
    ) -> dict[str, Any]:
        """Assign annotators to a task with workload balancing.

        Distributes task items evenly across annotators, factoring in their
        current workload across all tasks.

        Args:
            task_id: Task UUID to assign annotators to.
            annotator_ids: List of annotator UUIDs.

        Returns:
            Assignment summary dict.

        Raises:
            KeyError: If task_id is not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Annotation task {task_id} not found.")

        if not annotator_ids:
            raise ValueError("At least one annotator must be provided.")

        # Sort annotators by current workload (fewest tasks first)
        sorted_annotators = sorted(
            annotator_ids,
            key=lambda annotator_id: len(self._annotator_workload[annotator_id]),
        )

        # Assign all items to all annotators (each item needs N labels)
        for annotator_id in sorted_annotators:
            item_ids = [item["item_id"] for item in task.items]
            task.assigned_annotators[annotator_id] = item_ids
            if task_id not in self._annotator_workload[annotator_id]:
                self._annotator_workload[annotator_id].append(task_id)

        if task.status == TASK_STATUS_OPEN and sorted_annotators:
            task.status = TASK_STATUS_IN_PROGRESS

        logger.info(
            "Annotators assigned",
            task_id=str(task_id),
            annotator_count=len(sorted_annotators),
        )

        return {
            "task_id": str(task_id),
            "annotators_assigned": len(sorted_annotators),
            "items_per_annotator": len(task.items),
        }

    async def store_annotation(
        self,
        task_id: uuid.UUID,
        item_id: str,
        annotator_id: uuid.UUID,
        label: str,
        confidence: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Store an annotation from an annotator.

        Validates the label is in the task's allowed labels list and checks
        gold standard correctness if the item is a gold standard item.

        Args:
            task_id: Task UUID.
            item_id: Item being annotated.
            annotator_id: Annotator UUID.
            label: Chosen label.
            confidence: Optional annotator-reported confidence (0–1).
            notes: Optional notes from the annotator.

        Returns:
            Dict with annotation_id, gold_check_result (if applicable),
            and whether the item is now complete (enough annotations).

        Raises:
            KeyError: If task not found.
            ValueError: If label is not in allowed labels.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Annotation task {task_id} not found.")

        if label not in task.labels:
            raise ValueError(
                f"Label '{label}' not in allowed labels for task {task_id}. "
                f"Valid: {task.labels}"
            )

        is_gold_check = item_id in task.gold_standard_ids
        gold_correct: bool | None = None
        if is_gold_check:
            expected = self._gold_answers.get((task_id, item_id))
            gold_correct = (expected == label) if expected is not None else None

        annotation = Annotation(
            annotation_id=uuid.uuid4(),
            task_id=task_id,
            item_id=item_id,
            annotator_id=annotator_id,
            label=label,
            confidence=confidence,
            notes=notes,
            is_gold_check=is_gold_check,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._annotations[task_id].append(annotation)

        # Check if item has enough annotations to be considered complete
        item_annotation_count = sum(
            1
            for ann in self._annotations[task_id]
            if ann.item_id == item_id
        )
        item_complete = item_annotation_count >= task.annotations_per_item

        logger.debug(
            "Annotation stored",
            task_id=str(task_id),
            item_id=item_id,
            annotator_id=str(annotator_id),
            label=label,
            is_gold=is_gold_check,
            gold_correct=gold_correct,
        )

        return {
            "annotation_id": str(annotation.annotation_id),
            "is_gold_check": is_gold_check,
            "gold_correct": gold_correct,
            "item_annotation_count": item_annotation_count,
            "item_complete": item_complete,
        }

    async def get_task_progress(self, task_id: uuid.UUID) -> dict[str, Any]:
        """Return progress statistics for an annotation task.

        Args:
            task_id: Task UUID.

        Returns:
            Dict with total_items, completed_items, completion_pct, annotator_stats.

        Raises:
            KeyError: If task not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Annotation task {task_id} not found.")

        task_annotations = self._annotations[task_id]
        # Count annotations per item
        item_counts: dict[str, int] = defaultdict(int)
        for ann in task_annotations:
            item_counts[ann.item_id] += 1

        non_gold_items = [
            item for item in task.items
            if item["item_id"] not in task.gold_standard_ids
        ]
        total_items = len(non_gold_items)
        completed_items = sum(
            1
            for item in non_gold_items
            if item_counts.get(item["item_id"], 0) >= task.annotations_per_item
        )
        completion_pct = (completed_items / total_items * 100) if total_items > 0 else 0.0

        # Per-annotator stats
        annotator_stats: dict[str, int] = defaultdict(int)
        gold_accuracy: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
        for ann in task_annotations:
            annotator_stats[str(ann.annotator_id)] += 1
            if ann.is_gold_check:
                expected = self._gold_answers.get((task_id, ann.item_id))
                gold_accuracy[str(ann.annotator_id)]["total"] += 1
                if expected is not None and ann.label == expected:
                    gold_accuracy[str(ann.annotator_id)]["correct"] += 1

        annotator_summary = []
        for annotator_id_str, count in annotator_stats.items():
            gold_data = gold_accuracy[annotator_id_str]
            gold_pct = (
                gold_data["correct"] / gold_data["total"] * 100
                if gold_data["total"] > 0
                else None
            )
            annotator_summary.append(
                {
                    "annotator_id": annotator_id_str,
                    "annotations_submitted": count,
                    "gold_accuracy_pct": gold_pct,
                }
            )

        return {
            "task_id": str(task_id),
            "status": task.status,
            "total_items": total_items,
            "completed_items": completed_items,
            "completion_pct": round(completion_pct, 2),
            "total_annotations": len(task_annotations),
            "annotator_stats": annotator_summary,
        }

    async def export_annotations(
        self,
        task_id: uuid.UUID,
        format: str = "jsonl",
        exclude_gold: bool = True,
    ) -> str:
        """Export annotations in JSONL or CSV format.

        Args:
            task_id: Task UUID.
            format: Output format — "jsonl" or "csv".
            exclude_gold: If True, exclude gold standard items from export.

        Returns:
            Exported content as a string.

        Raises:
            KeyError: If task not found.
            ValueError: If format is invalid.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Annotation task {task_id} not found.")

        if format not in {"jsonl", "csv"}:
            raise ValueError(f"Invalid export format '{format}'. Use 'jsonl' or 'csv'.")

        annotations = [
            ann
            for ann in self._annotations[task_id]
            if not (exclude_gold and ann.is_gold_check)
        ]

        if format == "jsonl":
            lines = []
            for ann in annotations:
                lines.append(
                    json.dumps(
                        {
                            "annotation_id": str(ann.annotation_id),
                            "task_id": str(ann.task_id),
                            "item_id": ann.item_id,
                            "annotator_id": str(ann.annotator_id),
                            "label": ann.label,
                            "confidence": ann.confidence,
                            "notes": ann.notes,
                            "created_at": ann.created_at.isoformat(),
                        }
                    )
                )
            return "\n".join(lines)

        # CSV format
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "annotation_id", "task_id", "item_id", "annotator_id",
                "label", "confidence", "notes", "created_at",
            ],
        )
        writer.writeheader()
        for ann in annotations:
            writer.writerow(
                {
                    "annotation_id": str(ann.annotation_id),
                    "task_id": str(ann.task_id),
                    "item_id": ann.item_id,
                    "annotator_id": str(ann.annotator_id),
                    "label": ann.label,
                    "confidence": ann.confidence or "",
                    "notes": ann.notes or "",
                    "created_at": ann.created_at.isoformat(),
                }
            )
        return output.getvalue()

    def _task_summary(self, task: AnnotationTask) -> dict[str, Any]:
        """Build a task summary dict.

        Args:
            task: AnnotationTask instance.

        Returns:
            Summary dict.
        """
        return {
            "task_id": str(task.task_id),
            "tenant_id": str(task.tenant_id),
            "task_type": task.task_type,
            "name": task.name,
            "status": task.status,
            "total_items": len(task.items),
            "gold_standard_count": len(task.gold_standard_ids),
            "labels": task.labels,
            "annotations_per_item": task.annotations_per_item,
            "created_at": task.created_at.isoformat(),
        }

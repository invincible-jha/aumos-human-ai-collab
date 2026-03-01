"""Label Studio integration adapter for aumos-human-ai-collab.

Forwards HITL review tasks to registered Label Studio projects and
processes completed annotation payloads received via Label Studio webhooks.
Uses httpx for async HTTP communication with Label Studio's REST API.

Gap Coverage: GAP-259 (Label Studio Integration)
"""

import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Label Studio API path constants
_LS_TASKS_PATH = "/api/projects/{project_id}/tasks/"
_LS_TASK_PATH = "/api/tasks/{task_id}/"
_LS_ANNOTATIONS_PATH = "/api/tasks/{task_id}/annotations/"


@dataclass
class LabelStudioTaskPayload:
    """Payload for creating a task in a Label Studio project.

    Attributes:
        hitl_review_id: UUID of the source HITLReview record.
        task_data: Dict of fields shown to annotators in the Label Studio UI.
        meta: Metadata attached to the task for filtering/routing.
    """

    hitl_review_id: uuid.UUID
    task_data: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class LabelStudioExportResult:
    """Result of exporting a HITL review task to Label Studio.

    Attributes:
        success: Whether the task was successfully created.
        label_studio_task_id: Numeric task ID assigned by Label Studio.
        error_detail: Error message on failure.
    """

    success: bool
    label_studio_task_id: int | None
    error_detail: str | None = None


@dataclass
class ParsedAnnotation:
    """Parsed annotation returned from a Label Studio webhook callback.

    Attributes:
        hitl_review_id: UUID extracted from task meta.
        label_studio_task_id: Numeric Label Studio task ID.
        label_studio_annotation_id: Numeric annotation ID.
        annotation_result: Raw annotation result list from Label Studio.
        annotator_email: Email of the annotator who submitted.
        completed_at: ISO timestamp when annotation was submitted.
    """

    hitl_review_id: uuid.UUID
    label_studio_task_id: int
    label_studio_annotation_id: int
    annotation_result: list[dict[str, Any]]
    annotator_email: str | None
    completed_at: str | None


class LabelStudioAdapter:
    """Adapter for bidirectional integration with a Label Studio instance.

    Exports HITL review tasks to Label Studio via its REST API and parses
    incoming webhook payloads when annotations are completed.

    Args:
        base_url: Base URL of the Label Studio instance (no trailing slash).
        api_token: Label Studio API token for authentication.
        http_timeout_seconds: HTTP request timeout.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        http_timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize the adapter.

        Args:
            base_url: Label Studio base URL.
            api_token: Label Studio API token.
            http_timeout_seconds: Per-request HTTP timeout.
        """
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._http_timeout = http_timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Create the shared HTTP client with auth headers.

        Must be called before any export/fetch methods are used.
        Typically called during application lifespan startup.
        """
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Token {self._api_token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._http_timeout),
            follow_redirects=False,
            verify=True,
        )
        logger.info(
            "LabelStudioAdapter initialized",
            base_url=self._base_url,
        )

    async def close(self) -> None:
        """Close the HTTP client on application shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def export_task(
        self,
        project_id: int,
        payload: LabelStudioTaskPayload,
    ) -> LabelStudioExportResult:
        """Export a HITL review as a task to a Label Studio project.

        Creates a task in the specified Label Studio project. The task data
        must match the project's label configuration schema.

        Args:
            project_id: Label Studio numeric project ID.
            payload: Task payload containing review data and metadata.

        Returns:
            LabelStudioExportResult with success flag and task ID.
        """
        if self._client is None:
            raise RuntimeError("LabelStudioAdapter.initialize() was not called")

        task_body = {
            "data": {
                **payload.task_data,
                "hitl_review_id": str(payload.hitl_review_id),
            },
            "meta": payload.meta,
        }

        path = _LS_TASKS_PATH.format(project_id=project_id)

        try:
            response = await self._client.post(path, json=task_body)

            if response.status_code == 201:
                response_data = response.json()
                task_id = response_data.get("id")
                logger.info(
                    "Task exported to Label Studio",
                    project_id=project_id,
                    label_studio_task_id=task_id,
                    hitl_review_id=str(payload.hitl_review_id),
                )
                return LabelStudioExportResult(
                    success=True,
                    label_studio_task_id=task_id,
                )

            error_body = response.text[:500]
            logger.warning(
                "Label Studio task creation failed",
                project_id=project_id,
                http_status=response.status_code,
                error_body=error_body,
                hitl_review_id=str(payload.hitl_review_id),
            )
            return LabelStudioExportResult(
                success=False,
                label_studio_task_id=None,
                error_detail=f"HTTP {response.status_code}: {error_body}",
            )

        except httpx.TimeoutException as exc:
            logger.error(
                "Label Studio export timed out",
                project_id=project_id,
                hitl_review_id=str(payload.hitl_review_id),
                error=str(exc),
            )
            return LabelStudioExportResult(
                success=False,
                label_studio_task_id=None,
                error_detail=f"Timeout: {exc}",
            )
        except Exception as exc:
            logger.error(
                "Unexpected error exporting to Label Studio",
                project_id=project_id,
                hitl_review_id=str(payload.hitl_review_id),
                error=str(exc),
            )
            return LabelStudioExportResult(
                success=False,
                label_studio_task_id=None,
                error_detail=str(exc),
            )

    def parse_webhook_payload(
        self,
        raw_payload: dict[str, Any],
    ) -> ParsedAnnotation | None:
        """Parse an incoming Label Studio webhook annotation event.

        Label Studio sends a webhook POST when an annotation is submitted.
        This method extracts the hitl_review_id from the task's data/meta
        and returns a structured ParsedAnnotation.

        Args:
            raw_payload: The raw JSON body from the Label Studio webhook.

        Returns:
            ParsedAnnotation if the payload is valid, None if it cannot
            be parsed or is missing the hitl_review_id field.
        """
        try:
            annotation = raw_payload.get("annotation", {})
            task = raw_payload.get("task", {})

            task_id = task.get("id")
            annotation_id = annotation.get("id")

            # hitl_review_id is stored in task data during export
            task_data = task.get("data", {})
            hitl_review_id_str = task_data.get("hitl_review_id") or task.get("meta", {}).get(
                "hitl_review_id"
            )

            if not hitl_review_id_str:
                logger.warning(
                    "Label Studio webhook payload missing hitl_review_id",
                    task_id=task_id,
                    annotation_id=annotation_id,
                )
                return None

            hitl_review_id = uuid.UUID(hitl_review_id_str)
            annotation_result = annotation.get("result", [])

            completed_by = annotation.get("completed_by", {})
            annotator_email = (
                completed_by.get("email") if isinstance(completed_by, dict) else None
            )

            logger.info(
                "Parsed Label Studio webhook annotation",
                hitl_review_id=str(hitl_review_id),
                task_id=task_id,
                annotation_id=annotation_id,
                n_results=len(annotation_result),
            )

            return ParsedAnnotation(
                hitl_review_id=hitl_review_id,
                label_studio_task_id=task_id,
                label_studio_annotation_id=annotation_id,
                annotation_result=annotation_result,
                annotator_email=annotator_email,
                completed_at=annotation.get("created_at"),
            )

        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "Failed to parse Label Studio webhook payload",
                error=str(exc),
            )
            return None

    async def get_task_annotations(
        self,
        task_id: int,
    ) -> list[dict[str, Any]]:
        """Retrieve all annotations for a Label Studio task.

        Used for polling-based sync when webhooks are unavailable.

        Args:
            task_id: Label Studio numeric task ID.

        Returns:
            List of annotation dicts from Label Studio.
        """
        if self._client is None:
            raise RuntimeError("LabelStudioAdapter.initialize() was not called")

        path = _LS_ANNOTATIONS_PATH.format(task_id=task_id)
        try:
            response = await self._client.get(path)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(
                "Failed to fetch Label Studio annotations",
                task_id=task_id,
                error=str(exc),
            )
            return []


__all__ = [
    "LabelStudioTaskPayload",
    "LabelStudioExportResult",
    "ParsedAnnotation",
    "LabelStudioAdapter",
]

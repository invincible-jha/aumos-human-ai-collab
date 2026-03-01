"""Server-rendered reviewer interface routes for GAP-256.

Provides a Jinja2 HTML review interface for HITL reviewers to receive,
review, and submit decisions on AI-routed tasks. This is the primary
human-facing interface for the human-AI collaboration system.

Routes:
    GET  /ui/review-queue        — Queue of pending reviews for the reviewer
    GET  /ui/review/{review_id}  — Single review detail and decision form
    POST /ui/review/{review_id}/decide — Form submission for review decision
    GET  /ui/dashboard           — Reviewer performance dashboard

The UI can be disabled via AUMOS_HUMAN_AI_UI_ENABLED=false.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Templates directory is relative to the installed package
import pathlib

_PACKAGE_DIR = pathlib.Path(__file__).parent.parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

ui_router = APIRouter(prefix="/ui", tags=["reviewer-ui"])


def _get_hitl_service(request: Request):  # type: ignore[return]
    """FastAPI dependency to retrieve HITLReviewService from app state."""
    return request.app.state.hitl_service


def _get_workforce_service(request: Request):  # type: ignore[return]
    """FastAPI dependency to retrieve WorkforceService from app state."""
    return getattr(request.app.state, "workforce_service", None)


@ui_router.get("/review-queue", response_class=HTMLResponse)
async def review_queue_page(
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    reviewer_id: Annotated[str, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    """Render the HITL review queue for the current reviewer.

    Args:
        request: FastAPI request (needed for template rendering).
        tenant_id: Current tenant UUID from auth middleware.
        reviewer_id: Current user UUID from auth middleware.
        db: Async database session.

    Returns:
        HTMLResponse with rendered review_queue.html template.
    """
    hitl_service = _get_hitl_service(request)
    pending_reviews = await hitl_service.list_pending_for_reviewer(
        reviewer_id=uuid.UUID(reviewer_id),
        tenant_id=uuid.UUID(tenant_id),
        db=db,
    )
    return templates.TemplateResponse(
        "review_queue.html",
        {
            "request": request,
            "reviews": pending_reviews,
            "reviewer_id": reviewer_id,
        },
    )


@ui_router.get("/review/{review_id}", response_class=HTMLResponse)
async def review_detail_page(
    request: Request,
    review_id: uuid.UUID,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    reviewer_id: Annotated[str, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    """Render the review interface for a specific HITL task.

    Args:
        request: FastAPI request.
        review_id: UUID of the HITL review to display.
        tenant_id: Current tenant UUID.
        reviewer_id: Current reviewer user UUID.
        db: Async database session.

    Returns:
        HTMLResponse with rendered review_detail.html template.
    """
    hitl_service = _get_hitl_service(request)
    review = await hitl_service.get_review(
        review_id=review_id,
        tenant_id=uuid.UUID(tenant_id),
        db=db,
    )

    ai_output = getattr(review, "ai_output", {})
    return templates.TemplateResponse(
        "review_detail.html",
        {
            "request": request,
            "review": review,
            "ai_output": ai_output,
            "compliance_gate": None,
        },
    )


@ui_router.post("/review/{review_id}/decide", response_class=HTMLResponse)
async def submit_review_decision_ui(
    request: Request,
    review_id: uuid.UUID,
    decision: str = Form(...),
    justification: str = Form(...),
    modifications: str | None = Form(default=None),
    tenant_id: Annotated[str, Depends(get_current_tenant)] = "",
    reviewer_id: Annotated[str, Depends(get_current_user)] = "",
    db: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """Process form submission for a review decision and redirect to queue.

    The decision is immutable once submitted. Duplicate POSTs (browser back,
    network retry) will return 409 Conflict from the underlying service.

    Args:
        request: FastAPI request.
        review_id: UUID of the HITL review being decided.
        decision: approved | rejected | modified.
        justification: Reviewer's written justification.
        modifications: Optional JSON string of modified AI output.
        tenant_id: Current tenant UUID.
        reviewer_id: Current reviewer user UUID.
        db: Async database session.

    Returns:
        303 redirect to /ui/review-queue on success.
    """
    hitl_service = _get_hitl_service(request)

    reviewer_output: dict = {}
    if modifications:
        import json

        try:
            reviewer_output = json.loads(modifications)
        except json.JSONDecodeError:
            reviewer_output = {"raw_modification": modifications}

    await hitl_service.submit_decision(
        review_id=review_id,
        tenant_id=uuid.UUID(tenant_id),
        reviewer_id=uuid.UUID(reviewer_id),
        decision=decision,
        reviewer_notes=justification,
        reviewer_output=reviewer_output,
        db=db,
    )

    logger.info(
        "Review decision submitted via UI",
        review_id=str(review_id),
        decision=decision,
        reviewer_id=reviewer_id,
    )
    return RedirectResponse(url="/ui/review-queue", status_code=303)


@ui_router.get("/dashboard", response_class=HTMLResponse)
async def reviewer_dashboard(
    request: Request,
    tenant_id: Annotated[str, Depends(get_current_tenant)],
    reviewer_id: Annotated[str, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    """Render the reviewer performance dashboard.

    Args:
        request: FastAPI request.
        tenant_id: Current tenant UUID.
        reviewer_id: Current reviewer user UUID.
        db: Async database session.

    Returns:
        HTMLResponse with rendered reviewer_dashboard.html template.
    """
    workforce_service = _get_workforce_service(request)
    profile = None
    if workforce_service is not None:
        from sqlalchemy import and_, select

        from aumos_human_ai_collab.core.gap_models import HacReviewerProfile

        result = await db.execute(
            select(HacReviewerProfile).where(
                and_(
                    HacReviewerProfile.tenant_id == uuid.UUID(tenant_id),
                    HacReviewerProfile.user_id == uuid.UUID(reviewer_id),
                )
            )
        )
        profile = result.scalar_one_or_none()

    return templates.TemplateResponse(
        "reviewer_dashboard.html",
        {
            "request": request,
            "profile": profile,
            "reviewer_id": reviewer_id,
        },
    )

"""SQLAlchemy repository implementations for the Human-AI Collaboration service.

Concrete repository classes implementing the interface protocols defined in
core/interfaces.py. All queries enforce tenant isolation via RLS (SET app.current_tenant).
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import get_db_session
from aumos_common.observability import get_logger

from aumos_human_ai_collab.core.models import (
    AttributionRecord,
    ComplianceGate,
    FeedbackCorrection,
    HITLReview,
    RoutingDecision,
)

logger = get_logger(__name__)


class RoutingDecisionRepository:
    """SQLAlchemy repository for RoutingDecision persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        task_id: uuid.UUID,
        task_type: str,
        confidence_score: float,
        routing_outcome: str,
        ai_threshold_applied: float,
        model_id: str | None,
        compliance_gate_triggered: bool,
        compliance_gate_id: uuid.UUID | None,
        routing_metadata: dict[str, Any],
    ) -> RoutingDecision:
        """Create and persist a routing decision."""
        async with get_db_session(tenant_id) as session:
            decision = RoutingDecision(
                tenant_id=tenant_id,
                task_id=task_id,
                task_type=task_type,
                confidence_score=confidence_score,
                routing_outcome=routing_outcome,
                ai_threshold_applied=ai_threshold_applied,
                model_id=model_id,
                compliance_gate_triggered=compliance_gate_triggered,
                compliance_gate_id=compliance_gate_id,
                routing_metadata=routing_metadata,
            )
            session.add(decision)
            await session.commit()
            await session.refresh(decision)
            return decision

    async def get_by_id(
        self, decision_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> RoutingDecision | None:
        """Retrieve a routing decision by UUID within a tenant."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(RoutingDecision).where(
                    and_(
                        RoutingDecision.id == decision_id,
                        RoutingDecision.tenant_id == tenant_id,
                    )
                )
            )
            return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        task_type: str | None,
        routing_outcome: str | None,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> tuple[list[RoutingDecision], int]:
        """List routing decisions for a tenant with optional filters."""
        async with get_db_session(tenant_id) as session:
            conditions = [RoutingDecision.tenant_id == tenant_id]

            if task_type is not None:
                conditions.append(RoutingDecision.task_type == task_type)
            if routing_outcome is not None:
                conditions.append(RoutingDecision.routing_outcome == routing_outcome)
            if date_from is not None:
                conditions.append(RoutingDecision.created_at >= date_from)
            if date_to is not None:
                conditions.append(RoutingDecision.created_at <= date_to)

            count_result = await session.execute(
                select(func.count()).select_from(RoutingDecision).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                select(RoutingDecision)
                .where(and_(*conditions))
                .order_by(RoutingDecision.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            decisions = list(result.scalars().all())
            return decisions, total

    async def mark_resolved(
        self,
        decision_id: uuid.UUID,
        resolved_by: str,
        resolved_at: datetime,
    ) -> RoutingDecision:
        """Mark a routing decision as resolved."""
        async with get_db_session(None) as session:
            await session.execute(
                update(RoutingDecision)
                .where(RoutingDecision.id == decision_id)
                .values(resolved_by=resolved_by, resolved_at=resolved_at)
            )
            await session.commit()

            result = await session.execute(
                select(RoutingDecision).where(RoutingDecision.id == decision_id)
            )
            return result.scalar_one()


class ComplianceGateRepository:
    """SQLAlchemy repository for ComplianceGate persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        gate_name: str,
        regulation: str,
        task_types: list[str],
        description: str | None,
        reviewer_role: str | None,
        metadata: dict[str, Any],
    ) -> ComplianceGate:
        """Create a compliance gate."""
        async with get_db_session(tenant_id) as session:
            gate = ComplianceGate(
                tenant_id=tenant_id,
                gate_name=gate_name,
                regulation=regulation,
                task_types=task_types,
                description=description,
                reviewer_role=reviewer_role,
                metadata=metadata,
            )
            session.add(gate)
            await session.commit()
            await session.refresh(gate)
            return gate

    async def get_by_id(
        self, gate_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> ComplianceGate | None:
        """Retrieve a compliance gate by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(ComplianceGate).where(
                    and_(
                        ComplianceGate.id == gate_id,
                        ComplianceGate.tenant_id == tenant_id,
                    )
                )
            )
            return result.scalar_one_or_none()

    async def find_matching_gates(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> list[ComplianceGate]:
        """Find all active compliance gates that apply to a given task type.

        Uses PostgreSQL JSONB containment operator to check if task_type
        appears in the task_types JSONB array.
        """
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(ComplianceGate).where(
                    and_(
                        ComplianceGate.tenant_id == tenant_id,
                        ComplianceGate.is_active.is_(True),
                        ComplianceGate.task_types.contains([task_type]),  # type: ignore[attr-defined]
                    )
                )
            )
            return list(result.scalars().all())

    async def list_by_tenant(
        self, tenant_id: uuid.UUID, active_only: bool
    ) -> list[ComplianceGate]:
        """List compliance gates for a tenant."""
        async with get_db_session(tenant_id) as session:
            conditions = [ComplianceGate.tenant_id == tenant_id]
            if active_only:
                conditions.append(ComplianceGate.is_active.is_(True))

            result = await session.execute(
                select(ComplianceGate)
                .where(and_(*conditions))
                .order_by(ComplianceGate.gate_name)
            )
            return list(result.scalars().all())

    async def update(
        self, gate_id: uuid.UUID, tenant_id: uuid.UUID, updates: dict[str, Any]
    ) -> ComplianceGate:
        """Apply partial updates to a compliance gate."""
        async with get_db_session(tenant_id) as session:
            await session.execute(
                update(ComplianceGate)
                .where(
                    and_(
                        ComplianceGate.id == gate_id,
                        ComplianceGate.tenant_id == tenant_id,
                    )
                )
                .values(**updates)
            )
            await session.commit()

            result = await session.execute(
                select(ComplianceGate).where(ComplianceGate.id == gate_id)
            )
            return result.scalar_one()


class HITLReviewRepository:
    """SQLAlchemy repository for HITLReview persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        routing_decision_id: uuid.UUID,
        task_type: str,
        ai_output: dict[str, Any],
        reviewer_id: uuid.UUID | None,
        due_at: datetime | None,
        priority: int,
    ) -> HITLReview:
        """Create a HITL review task."""
        async with get_db_session(tenant_id) as session:
            review = HITLReview(
                tenant_id=tenant_id,
                routing_decision_id=routing_decision_id,
                task_type=task_type,
                ai_output=ai_output,
                reviewer_id=reviewer_id,
                due_at=due_at,
                priority=priority,
            )
            session.add(review)
            await session.commit()
            await session.refresh(review)
            return review

    async def get_by_id(
        self, review_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> HITLReview | None:
        """Retrieve a HITL review by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(HITLReview).where(
                    and_(
                        HITLReview.id == review_id,
                        HITLReview.tenant_id == tenant_id,
                    )
                )
            )
            return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        status: str | None,
        reviewer_id: uuid.UUID | None,
        task_type: str | None,
    ) -> tuple[list[HITLReview], int]:
        """List HITL reviews for a tenant with optional filters."""
        async with get_db_session(tenant_id) as session:
            conditions = [HITLReview.tenant_id == tenant_id]

            if status is not None:
                conditions.append(HITLReview.status == status)
            if reviewer_id is not None:
                conditions.append(HITLReview.reviewer_id == reviewer_id)
            if task_type is not None:
                conditions.append(HITLReview.task_type == task_type)

            count_result = await session.execute(
                select(func.count()).select_from(HITLReview).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                select(HITLReview)
                .where(and_(*conditions))
                .order_by(HITLReview.priority.desc(), HITLReview.created_at.asc())
                .offset(offset)
                .limit(page_size)
            )
            reviews = list(result.scalars().all())
            return reviews, total

    async def submit_decision(
        self,
        review_id: uuid.UUID,
        decision: str,
        reviewer_output: dict[str, Any],
        reviewer_notes: str | None,
        review_completed_at: datetime,
    ) -> HITLReview:
        """Record the reviewer's decision and output."""
        async with get_db_session(None) as session:
            await session.execute(
                update(HITLReview)
                .where(HITLReview.id == review_id)
                .values(
                    status=decision,
                    decision=decision,
                    reviewer_output=reviewer_output,
                    reviewer_notes=reviewer_notes,
                    review_completed_at=review_completed_at,
                )
            )
            await session.commit()

            result = await session.execute(
                select(HITLReview).where(HITLReview.id == review_id)
            )
            return result.scalar_one()

    async def update_status(
        self,
        review_id: uuid.UUID,
        status: str,
        reviewer_id: uuid.UUID | None,
        review_started_at: datetime | None,
    ) -> HITLReview:
        """Update the status of a HITL review."""
        async with get_db_session(None) as session:
            updates: dict[str, Any] = {"status": status}
            if reviewer_id is not None:
                updates["reviewer_id"] = reviewer_id
            if review_started_at is not None:
                updates["review_started_at"] = review_started_at

            await session.execute(
                update(HITLReview).where(HITLReview.id == review_id).values(**updates)
            )
            await session.commit()

            result = await session.execute(
                select(HITLReview).where(HITLReview.id == review_id)
            )
            return result.scalar_one()


class AttributionRepository:
    """SQLAlchemy repository for AttributionRecord persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        task_id: uuid.UUID,
        task_type: str,
        routing_decision_id: uuid.UUID | None,
        hitl_review_id: uuid.UUID | None,
        ai_contribution_score: float,
        human_contribution_score: float,
        time_saved_seconds: int | None,
        attribution_method: str,
        resolution: str,
        attribution_metadata: dict[str, Any],
    ) -> AttributionRecord:
        """Create an attribution record."""
        async with get_db_session(tenant_id) as session:
            record = AttributionRecord(
                tenant_id=tenant_id,
                task_id=task_id,
                task_type=task_type,
                routing_decision_id=routing_decision_id,
                hitl_review_id=hitl_review_id,
                ai_contribution_score=ai_contribution_score,
                human_contribution_score=human_contribution_score,
                time_saved_seconds=time_saved_seconds,
                attribution_method=attribution_method,
                resolution=resolution,
                attribution_metadata=attribution_metadata,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_report(
        self,
        tenant_id: uuid.UUID,
        days: int,
        task_type: str | None,
    ) -> dict[str, Any]:
        """Generate an attribution analytics report for a tenant."""
        from datetime import timedelta

        async with get_db_session(tenant_id) as session:
            since = datetime.now(tz=timezone.utc) - timedelta(days=days)
            conditions = [
                AttributionRecord.tenant_id == tenant_id,
                AttributionRecord.created_at >= since,
            ]
            if task_type is not None:
                conditions.append(AttributionRecord.task_type == task_type)

            # Aggregate totals
            count_result = await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(
                        func.cast(AttributionRecord.resolution == "ai", type_=None)
                    ).label("ai_handled"),
                    func.sum(
                        func.cast(AttributionRecord.resolution == "human", type_=None)
                    ).label("human_handled"),
                    func.sum(
                        func.cast(AttributionRecord.resolution == "hybrid", type_=None)
                    ).label("hybrid_handled"),
                    func.avg(AttributionRecord.ai_contribution_score).label("avg_ai"),
                    func.avg(AttributionRecord.human_contribution_score).label("avg_human"),
                    func.coalesce(
                        func.sum(AttributionRecord.time_saved_seconds), 0
                    ).label("time_saved"),
                )
                .where(and_(*conditions))
            )
            row = count_result.one()

            total = int(row.total or 0)
            ai_pct = float(row.avg_ai or 0.0) * 100
            human_pct = float(row.avg_human or 0.0) * 100

            # Per-task-type breakdown
            type_result = await session.execute(
                select(
                    AttributionRecord.task_type,
                    func.count().label("count"),
                    func.avg(AttributionRecord.ai_contribution_score).label("avg_ai"),
                    func.avg(AttributionRecord.human_contribution_score).label("avg_human"),
                )
                .where(and_(*conditions))
                .group_by(AttributionRecord.task_type)
                .order_by(func.count().desc())
            )
            by_type = [
                {
                    "task_type": r.task_type,
                    "count": int(r.count),
                    "avg_ai_pct": float(r.avg_ai or 0.0) * 100,
                    "avg_human_pct": float(r.avg_human or 0.0) * 100,
                }
                for r in type_result.all()
            ]

            return {
                "total_tasks": total,
                "ai_handled": int(row.ai_handled or 0),
                "human_handled": int(row.human_handled or 0),
                "hybrid_handled": int(row.hybrid_handled or 0),
                "ai_contribution_pct": ai_pct,
                "human_contribution_pct": human_pct,
                "total_time_saved_seconds": int(row.time_saved or 0),
                "by_task_type": by_type,
            }


class FeedbackRepository:
    """SQLAlchemy repository for FeedbackCorrection persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        routing_decision_id: uuid.UUID,
        hitl_review_id: uuid.UUID | None,
        task_type: str,
        original_confidence: float,
        original_routing: str,
        corrected_routing: str,
        correction_reason: str | None,
        correction_type: str,
        submitted_by: uuid.UUID | None,
    ) -> FeedbackCorrection:
        """Record a human feedback correction."""
        async with get_db_session(tenant_id) as session:
            correction = FeedbackCorrection(
                tenant_id=tenant_id,
                routing_decision_id=routing_decision_id,
                hitl_review_id=hitl_review_id,
                task_type=task_type,
                original_confidence=original_confidence,
                original_routing=original_routing,
                corrected_routing=corrected_routing,
                correction_reason=correction_reason,
                correction_type=correction_type,
                submitted_by=submitted_by,
            )
            session.add(correction)
            await session.commit()
            await session.refresh(correction)
            return correction

    async def list_uncalibrated(
        self, tenant_id: uuid.UUID, task_type: str, limit: int
    ) -> list[FeedbackCorrection]:
        """List feedback corrections not yet applied to calibration."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(FeedbackCorrection)
                .where(
                    and_(
                        FeedbackCorrection.tenant_id == tenant_id,
                        FeedbackCorrection.task_type == task_type,
                        FeedbackCorrection.calibration_applied.is_(False),
                    )
                )
                .order_by(FeedbackCorrection.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def mark_calibrated(
        self,
        correction_ids: list[uuid.UUID],
        calibration_applied_at: datetime,
    ) -> int:
        """Mark feedback corrections as applied to calibration."""
        async with get_db_session(None) as session:
            result = await session.execute(
                update(FeedbackCorrection)
                .where(FeedbackCorrection.id.in_(correction_ids))
                .values(
                    calibration_applied=True,
                    calibration_applied_at=calibration_applied_at,
                )
            )
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

    async def get_calibration_summary(
        self, tenant_id: uuid.UUID, task_type: str
    ) -> dict[str, Any]:
        """Summarise feedback corrections for recalibration input."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(
                        func.cast(FeedbackCorrection.calibration_applied.is_(False), type_=None)
                    ).label("uncalibrated"),
                    func.avg(
                        func.abs(
                            FeedbackCorrection.original_confidence - 0.5
                        )
                    ).label("mean_delta"),
                    func.max(FeedbackCorrection.calibration_applied_at).label("last_calibration"),
                )
                .where(
                    and_(
                        FeedbackCorrection.tenant_id == tenant_id,
                        FeedbackCorrection.task_type == task_type,
                    )
                )
            )
            row = result.one()

            total = int(row.total or 0)
            error_rate = float(row.uncalibrated or 0) / max(total, 1)

            # Correction type breakdown
            type_result = await session.execute(
                select(
                    FeedbackCorrection.correction_type,
                    func.count().label("count"),
                )
                .where(
                    and_(
                        FeedbackCorrection.tenant_id == tenant_id,
                        FeedbackCorrection.task_type == task_type,
                    )
                )
                .group_by(FeedbackCorrection.correction_type)
            )
            type_breakdown = {r.correction_type: int(r.count) for r in type_result.all()}

            return {
                "total_corrections": total,
                "uncalibrated_count": int(row.uncalibrated or 0),
                "error_rate": error_rate,
                "mean_confidence_delta": float(row.mean_delta or 0.0),
                "correction_type_breakdown": type_breakdown,
                "last_calibration_at": row.last_calibration,
            }

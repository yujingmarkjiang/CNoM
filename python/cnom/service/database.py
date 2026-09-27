from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint, create_engine, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    object_key: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class JobStore:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False, "timeout": 30} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.sqlite = self.engine.dialect.name == "sqlite"
        if self.sqlite:
            self._migrate_sqlite()

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _migrate_sqlite(self) -> None:
        with self.engine.begin() as connection:
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(jobs)").fetchall()}
            if "idempotency_key" not in columns:
                connection.exec_driver_sql("ALTER TABLE jobs ADD COLUMN idempotency_key VARCHAR(128)")
            connection.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_idempotency_key_idx ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL")

    def create(self, object_key: str, metadata: dict[str, Any], idempotency_key: str | None = None) -> tuple[Job, bool]:
        job = Job(id=str(uuid.uuid4()), idempotency_key=idempotency_key, status="queued", object_key=object_key, metadata_json=metadata, attempts=0)
        try:
            with self.sessions.begin() as session:
                session.add(job)
            return job, True
        except IntegrityError:
            if not idempotency_key:
                raise
            with self.sessions() as session:
                existing = session.execute(select(Job).where(Job.idempotency_key == idempotency_key)).scalar_one()
                session.expunge(existing)
                return existing, False

    def get(self, job_id: str) -> Job | None:
        with self.sessions() as session:
            job = session.get(Job, job_id)
            if job is not None:
                session.expunge(job)
            return job

    def lease(self, lease_seconds: int, max_attempts: int) -> Job | None:
        jobs = self.lease_batch(1, lease_seconds, max_attempts)
        return jobs[0] if jobs else None

    def lease_batch(self, batch_size: int, lease_seconds: int, max_attempts: int) -> list[Job]:
        if batch_size <= 0 or lease_seconds <= 0 or max_attempts <= 0:
            raise ValueError("batch_size, lease_seconds, and max_attempts must be positive")
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            if self.sqlite:
                session.execute(text("BEGIN IMMEDIATE"))
            session.execute(
                update(Job).where(Job.attempts >= max_attempts).where(
                    (Job.status == "queued") | ((Job.status == "processing") & (Job.lease_until < now))
                ).values(status="failed", error="Maximum processing attempts exceeded", lease_until=None, updated_at=now)
            )
            query = select(Job).where(Job.attempts < max_attempts).where(
                (Job.status == "queued") | ((Job.status == "processing") & (Job.lease_until < now))
            ).order_by(Job.created_at).limit(batch_size)
            if not self.sqlite:
                query = query.with_for_update(skip_locked=True)
            jobs = list(session.execute(query).scalars())
            for job in jobs:
                job.status = "processing"
                job.attempts += 1
                job.lease_until = now + timedelta(seconds=lease_seconds)
                job.updated_at = now
            session.flush()
            for job in jobs:
                session.expunge(job)
            return jobs

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        self._update(job_id, status="completed", result_json=result, error=None, lease_until=None)

    def fail(self, job_id: str, error: str, terminal: bool = False) -> None:
        self._update(job_id, status="failed" if terminal else "queued", error=error[:8000], lease_until=None)

    def counts(self) -> dict[str, int]:
        from sqlalchemy import func
        with self.sessions() as session:
            rows = session.execute(select(Job.status, func.count(Job.id)).group_by(Job.status)).all()
        return {str(status): int(count) for status, count in rows}

    def _update(self, job_id: str, **values: Any) -> None:
        values["updated_at"] = datetime.now(UTC)
        with self.sessions.begin() as session:
            result = session.execute(update(Job).where(Job.id == job_id).values(**values))
            if result.rowcount != 1:
                raise KeyError(f"Unknown job: {job_id}")

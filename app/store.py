from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, Optional

from app.models import JobStatus, RuntimePlan, ValidationJobRecord, ValidationJobRequest, ValidationMetrics


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, ValidationJobRecord] = {}
        self._lock = Lock()

    def create(self, request: ValidationJobRequest) -> ValidationJobRecord:
        with self._lock:
            if request.job_id in self._jobs:
                raise ValueError(f"job_id already exists: {request.job_id}")
            record = ValidationJobRecord(
                job_id=request.job_id,
                status=JobStatus.QUEUED,
                request=request,
            )
            self._jobs[request.job_id] = record
            return record

    def get(self, job_id: str) -> Optional[ValidationJobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = status
            record.updated_at = datetime.utcnow()

    def attach_runtime_plan(self, job_id: str, runtime_plan: RuntimePlan) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.runtime_plan = runtime_plan
            record.updated_at = datetime.utcnow()

    def complete(self, job_id: str, result: ValidationMetrics) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.COMPLETED
            record.result = result
            record.updated_at = datetime.utcnow()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.FAILED
            record.error = error
            record.updated_at = datetime.utcnow()


job_store = InMemoryJobStore()


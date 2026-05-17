from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.adapters.registry import get_adapter
from app.models import JobStatus
from app.runtime.docker_executor import DockerValidationExecutor
from app.runtime.materializer import materialize_path, read_text_path
from app.runtime.orchestrator import RuntimeOrchestrationAgent
from app.runtime.resolver import resolve_runtime_plan
from app.store import InMemoryJobStore


def run_validation_job(job_id: str, store: InMemoryJobStore) -> None:
    record = store.get(job_id)
    if record is None:
        return

    try:
        store.update_status(job_id, JobStatus.RESOLVING_ENVIRONMENT)
        requirements_text = read_text_path(record.request.requirements_gcs_path)
        runtime_plan = resolve_runtime_plan(record.request, requirements_text)
        runtime_plan.execution_mode = get_execution_mode()
        store.attach_runtime_plan(job_id, runtime_plan)

        store.update_status(job_id, JobStatus.RUNNING_VALIDATION)
        if runtime_plan.execution_mode == "docker":
            if is_runtime_agent_enabled():
                result, runtime_plan = RuntimeOrchestrationAgent().run(record.request, runtime_plan, requirements_text)
                store.attach_runtime_plan(job_id, runtime_plan)
            else:
                result = DockerValidationExecutor().run(record.request, runtime_plan, requirements_text)
        else:
            result = run_local_validation(record, runtime_plan)

        store.complete(job_id, result)
    except Exception as exc:
        store.fail(job_id, str(exc))


def run_local_validation(record, runtime_plan):
    with tempfile.TemporaryDirectory(prefix=f"validation-{record.job_id}-") as tmp:
        root = Path(tmp)
        model = materialize_path(record.request.model_gcs_path, root / "model")
        dataset = materialize_path(record.request.dataset_gcs_path, root / "dataset")
        artifacts_dir = root / "artifacts"

        adapter = get_adapter(runtime_plan.adapter)
        return adapter.validate(record.request, model.local_path, dataset.local_path, artifacts_dir)


def get_execution_mode() -> str:
    mode = os.environ.get("BYOM_EXECUTOR", "local").strip().lower()
    if mode not in {"local", "docker"}:
        raise ValueError("BYOM_EXECUTOR must be either 'local' or 'docker'")
    return mode


def is_runtime_agent_enabled() -> bool:
    value = os.environ.get("BYOM_RUNTIME_AGENT", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}

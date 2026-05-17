from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import RuntimePlan, ValidationJobRequest, ValidationMetrics
from app.runtime.docker_executor import DockerValidationExecutor, PROJECT_ROOT
from app.runtime.resolver import build_dependency_hash


class RuntimeOrchestrationAgent:
    """Small deterministic runtime agent for retrying obvious BYOM failures.

    This is intentionally not an LLM. It is a rules-based loop that can:
    - run a Docker validation attempt,
    - classify a failure,
    - apply safe runtime changes such as Python version correction,
    - persist what happened.
    """

    def __init__(self, project_root: Optional[Path] = None, max_attempts: int = 3) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.max_attempts = max_attempts
        self.executor = DockerValidationExecutor(self.project_root)
        self.history_dir = self.project_root / ".runtime_cache" / "orchestration_history"
        self.knowledge_path = self.project_root / ".runtime_cache" / "runtime_knowledge.json"

    def run(
        self,
        request: ValidationJobRequest,
        runtime_plan: RuntimePlan,
        requirements_text: str,
    ) -> tuple[ValidationMetrics, RuntimePlan]:
        attempts: List[Dict[str, Any]] = []
        current_plan = deepcopy(runtime_plan)

        for attempt_number in range(1, self.max_attempts + 1):
            try:
                result = self.executor.run(request, current_plan, requirements_text)
                attempts.append(self._attempt_record(attempt_number, current_plan, "completed"))
                self._store_success(request, current_plan, attempts)
                return result, current_plan
            except Exception as exc:
                error = str(exc)
                diagnosis = self._diagnose(error)
                attempts.append(
                    self._attempt_record(
                        attempt_number,
                        current_plan,
                        "failed",
                        diagnosis=diagnosis,
                        error=error,
                    )
                )

                next_plan = self._next_plan(current_plan, requirements_text, diagnosis)
                if next_plan is None:
                    self._store_failure(request, current_plan, attempts)
                    raise
                current_plan = next_plan

        self._store_failure(request, current_plan, attempts)
        raise RuntimeError("runtime orchestration exhausted all attempts")

    def _diagnose(self, error: str) -> Dict[str, Any]:
        requires_python = re.search(r"Requires-Python\s*>=\s*(\d+\.\d+)", error)
        if requires_python:
            return {
                "type": "python_version_too_low",
                "recommended_python": requires_python.group(1),
            }

        no_matching_dist = re.search(r"No matching distribution found for ([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+-]+)", error)
        if no_matching_dist:
            return {
                "type": "dependency_resolution_failed",
                "package": no_matching_dist.group(1),
                "version": no_matching_dist.group(2),
            }

        missing_pickle_class = re.search(r"Can't get attribute '([^']+)'", error)
        if missing_pickle_class:
            return {
                "type": "missing_pickle_supporting_code",
                "missing_class": missing_pickle_class.group(1),
            }

        missing_module = re.search(r"No module named '([^']+)'", error)
        if missing_module:
            return {
                "type": "missing_python_module",
                "module": missing_module.group(1),
            }

        return {"type": "unknown"}

    def _next_plan(
        self,
        current_plan: RuntimePlan,
        requirements_text: str,
        diagnosis: Dict[str, Any],
    ) -> Optional[RuntimePlan]:
        if diagnosis.get("type") == "python_version_too_low":
            recommended_python = diagnosis.get("recommended_python")
            if isinstance(recommended_python, str) and recommended_python != current_plan.python_version:
                next_plan = deepcopy(current_plan)
                next_plan.python_version = recommended_python
                next_plan.base_image = f"python:{recommended_python}-slim"
                next_plan.dependency_hash = build_dependency_hash(recommended_python, requirements_text)
                next_plan.image_tag = f"byom-runtime:{next_plan.dependency_hash}"
                next_plan.warnings.append(
                    f"runtime agent changed Python from {current_plan.python_version} to {recommended_python}"
                )
                return next_plan

        return None

    def _attempt_record(
        self,
        attempt_number: int,
        plan: RuntimePlan,
        status: str,
        diagnosis: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "attempt": attempt_number,
            "status": status,
            "diagnosis": diagnosis or {},
            "python_version": plan.python_version,
            "framework": plan.framework,
            "adapter": plan.adapter,
            "dependency_hash": plan.dependency_hash,
            "image_tag": plan.image_tag,
            "error_tail": error[-2000:] if error else None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _store_success(self, request: ValidationJobRequest, plan: RuntimePlan, attempts: List[Dict[str, Any]]) -> None:
        self._write_history(request.job_id, "completed", plan, attempts)
        knowledge = self._read_knowledge()
        knowledge[plan.dependency_hash] = {
            "status": "working",
            "framework": plan.framework,
            "adapter": plan.adapter,
            "python_version": plan.python_version,
            "image_tag": plan.image_tag,
            "last_success_job_id": request.job_id,
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._write_knowledge(knowledge)

    def _store_failure(self, request: ValidationJobRequest, plan: RuntimePlan, attempts: List[Dict[str, Any]]) -> None:
        self._write_history(request.job_id, "failed", plan, attempts)

    def _write_history(
        self,
        job_id: str,
        status: str,
        plan: RuntimePlan,
        attempts: List[Dict[str, Any]],
    ) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": job_id,
            "status": status,
            "final_plan": plan.dict() if hasattr(plan, "dict") else plan.model_dump(),
            "attempts": attempts,
            "updated_at": datetime.utcnow().isoformat(),
        }
        (self.history_dir / f"{job_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_knowledge(self) -> Dict[str, Any]:
        if not self.knowledge_path.exists():
            return {}
        return json.loads(self.knowledge_path.read_text(encoding="utf-8"))

    def _write_knowledge(self, knowledge: Dict[str, Any]) -> None:
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")

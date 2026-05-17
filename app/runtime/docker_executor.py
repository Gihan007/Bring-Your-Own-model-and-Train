from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.models import RuntimePlan, ValidationJobRequest, ValidationMetrics
from app.runtime.materializer import materialize_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CACHE_DIR = PROJECT_ROOT / ".runtime_cache"


class DockerValidationExecutor:
    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.runtime_cache_dir = self.project_root / ".runtime_cache"

    def run(
        self,
        request: ValidationJobRequest,
        runtime_plan: RuntimePlan,
        requirements_text: str,
    ) -> ValidationMetrics:
        if not runtime_plan.image_tag:
            raise RuntimeError("runtime_plan.image_tag is required for Docker execution")

        self._ensure_docker_available()
        self._build_image_if_needed(runtime_plan, requirements_text)
        run_dir = self._prepare_run_dir(request, requirements_text)
        self._run_container(runtime_plan, run_dir)
        return self._read_result(run_dir / "artifacts" / "validation_result.json")

    def _ensure_docker_available(self) -> None:
        self._run_command(["docker", "version", "--format", "{{.Server.Version}}"])

    def _build_image_if_needed(self, runtime_plan: RuntimePlan, requirements_text: str) -> None:
        assert runtime_plan.image_tag is not None
        inspect = subprocess.run(
            ["docker", "image", "inspect", runtime_plan.image_tag],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if inspect.returncode == 0:
            return

        build_context = self._prepare_build_context(runtime_plan, requirements_text)
        self._run_command(
            [
                "docker",
                "build",
                "--build-arg",
                f"PYTHON_VERSION={runtime_plan.python_version}",
                "-t",
                runtime_plan.image_tag,
                str(build_context),
            ],
            cwd=self.project_root,
        )

    def _prepare_build_context(self, runtime_plan: RuntimePlan, requirements_text: str) -> Path:
        build_context = self.runtime_cache_dir / "build_contexts" / runtime_plan.dependency_hash
        if build_context.exists():
            shutil.rmtree(build_context)
        build_context.mkdir(parents=True, exist_ok=True)

        shutil.copytree(self.project_root / "app", build_context / "app")
        base_requirements = self.project_root / "docker" / "base-runtime-requirements.txt"
        shutil.copy2(base_requirements, build_context / "base-runtime-requirements.txt")
        (build_context / "requirements.txt").write_text(requirements_text, encoding="utf-8")
        (build_context / "Dockerfile").write_text(
            self._dockerfile_text(),
            encoding="utf-8",
        )
        return build_context

    def _prepare_run_dir(self, request: ValidationJobRequest, requirements_text: str) -> Path:
        run_dir = self.runtime_cache_dir / "runs" / request.job_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        input_dir = run_dir / "input"
        artifacts_dir = run_dir / "artifacts"
        input_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model = materialize_path(request.model_gcs_path, input_dir)
        dataset = materialize_path(request.dataset_gcs_path, input_dir)
        requirements_path = input_dir / "requirements.txt"
        requirements_path.write_text(requirements_text, encoding="utf-8")

        container_request = request.copy(update={
            "model_gcs_path": f"/workspace/input/{model.local_path.name}",
            "dataset_gcs_path": f"/workspace/input/{dataset.local_path.name}",
            "requirements_gcs_path": "/workspace/input/requirements.txt",
        })
        payload = container_request.dict() if hasattr(container_request, "dict") else container_request.model_dump()
        (input_dir / "job_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_dir

    def _run_container(self, runtime_plan: RuntimePlan, run_dir: Path) -> None:
        assert runtime_plan.image_tag is not None
        input_dir = run_dir / "input"
        artifacts_dir = run_dir / "artifacts"
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            os.environ.get("BYOM_DOCKER_CPUS", "2"),
            "--memory",
            os.environ.get("BYOM_DOCKER_MEMORY", "4g"),
            "-v",
            f"{input_dir.resolve()}:/workspace/input:ro",
            "-v",
            f"{artifacts_dir.resolve()}:/workspace/artifacts",
            "-e",
            "JOB_PAYLOAD_PATH=/workspace/input/job_payload.json",
            "-e",
            "REQUIREMENTS_PATH=/workspace/input/requirements.txt",
            "-e",
            "ARTIFACTS_DIR=/workspace/artifacts",
            runtime_plan.image_tag,
        ]
        self._run_command(command, cwd=self.project_root)

    def _read_result(self, result_path: Path) -> ValidationMetrics:
        if not result_path.exists():
            raise RuntimeError(f"Docker validation did not produce result file: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return ValidationMetrics(**payload)

    def _run_command(self, command: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            raise RuntimeError(
                "command failed: "
                + " ".join(command)
                + "\nSTDOUT:\n"
                + stdout[-4000:]
                + "\nSTDERR:\n"
                + stderr[-4000:]
            )
        return result

    def _dockerfile_text(self) -> str:
        return """\
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY base-runtime-requirements.txt /workspace/base-runtime-requirements.txt
RUN pip install --no-cache-dir -r /workspace/base-runtime-requirements.txt

COPY requirements.txt /workspace/requirements.txt
RUN if [ -s /workspace/requirements.txt ]; then pip install --no-cache-dir -r /workspace/requirements.txt; fi

COPY app /workspace/app

ENTRYPOINT ["python", "-m", "app.runtime_entrypoint"]
"""

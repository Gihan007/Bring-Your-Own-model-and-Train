from __future__ import annotations

import json
import os
from pathlib import Path

from app.models import ValidationJobRequest
from app.runtime.materializer import materialize_path
from app.runtime.resolver import resolve_runtime_plan
from app.adapters.registry import get_adapter


def main() -> None:
    payload_path = os.environ.get("JOB_PAYLOAD_PATH", "/workspace/job_payload.json")
    requirements_path = os.environ.get("REQUIREMENTS_PATH", "/workspace/requirements.txt")
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/workspace/artifacts"))

    request = ValidationJobRequest(**json.loads(Path(payload_path).read_text(encoding="utf-8")))
    requirements_text = Path(requirements_path).read_text(encoding="utf-8")
    runtime_plan = resolve_runtime_plan(request, requirements_text)

    model = materialize_path(request.model_gcs_path, Path("/workspace/model"))
    dataset = materialize_path(request.dataset_gcs_path, Path("/workspace/dataset"))
    adapter = get_adapter(runtime_plan.adapter)
    result = adapter.validate(request, model.local_path, dataset.local_path, artifacts_dir)

    if hasattr(result, "model_dump_json"):
        print(result.model_dump_json())
    else:
        print(result.json())


if __name__ == "__main__":
    main()

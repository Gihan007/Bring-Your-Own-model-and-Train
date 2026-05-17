from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from app.models import ValidationJobRecord, ValidationJobRequest
from app.store import job_store
from app.worker import run_validation_job

UPLOAD_ROOT = Path("uploads")

app = FastAPI(
    title="BYOM Model Validation Prototype",
    version="0.1.0",
    description="Prototype for containerized bring-your-own-model validation jobs.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/validation-jobs", response_model=ValidationJobRecord, status_code=202)
def create_validation_job(
    request: ValidationJobRequest,
    background_tasks: BackgroundTasks,
) -> ValidationJobRecord:
    try:
        record = job_store.create(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(run_validation_job, request.job_id, job_store)
    return record


@app.post("/validation-jobs/upload", response_model=ValidationJobRecord, status_code=202)
def upload_validation_job(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    requirements_file: Optional[UploadFile] = File(None),
    job_id: str = Form(...),
    tenant_id: str = Form(...),
    model_id: str = Form(...),
    framework: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    model_type: Optional[str] = Form(None),
    numeric_features: str = Form("[]"),
    categorical_features: str = Form("[]"),
    model_params: str = Form("{}"),
    dimensions: str = Form("{}"),
) -> ValidationJobRecord:
    if job_store.get(job_id) is not None:
        raise HTTPException(status_code=409, detail=f"job_id already exists: {job_id}")

    upload_dir = UPLOAD_ROOT / _safe_path_part(tenant_id) / _safe_path_part(job_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    model_path = _save_upload(model_file, upload_dir, "model")
    dataset_path = _save_upload(dataset_file, upload_dir, "dataset")
    requirements_path = None
    if requirements_file is not None and requirements_file.filename:
        requirements_path = _save_upload(requirements_file, upload_dir, "requirements")

    request = ValidationJobRequest(
        job_id=job_id,
        tenant_id=tenant_id,
        model_id=model_id,
        model_gcs_path=str(model_path),
        dataset_gcs_path=str(dataset_path),
        requirements_gcs_path=str(requirements_path) if requirements_path else None,
        framework=framework,
        task_type=task_type,
        target_column=target_column,
        model_type=model_type,
        numeric_features=_parse_list_field(numeric_features, "numeric_features"),
        categorical_features=_parse_list_field(categorical_features, "categorical_features"),
        model_params=_parse_dict_field(model_params, "model_params"),
        dimensions=_parse_dict_field(dimensions, "dimensions"),
    )

    try:
        record = job_store.create(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(run_validation_job, request.job_id, job_store)
    return record


@app.get("/validation-jobs/{job_id}", response_model=ValidationJobRecord)
def get_validation_job(job_id: str) -> ValidationJobRecord:
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="validation job not found")
    return record


def _save_upload(upload: UploadFile, upload_dir: Path, fallback_name: str) -> Path:
    filename = Path(upload.filename or fallback_name).name
    destination = upload_dir / filename
    with destination.open("wb") as file_handle:
        shutil.copyfileobj(upload.file, file_handle)
    return destination


def _parse_list_field(value: str, field_name: str) -> List[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if not stripped.startswith("["):
        return [item.strip() for item in stripped.split(",") if item.strip()]

    parsed = _parse_json(value, field_name)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(status_code=422, detail=f"{field_name} must be a JSON array of strings")
    return parsed


def _parse_dict_field(value: str, field_name: str) -> Dict[str, Any]:
    parsed = _parse_json(value, field_name)
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail=f"{field_name} must be a JSON object")
    return parsed


def _parse_json(value: str, field_name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} must be valid JSON") from exc


def _safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)

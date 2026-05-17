from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RESOLVING_ENVIRONMENT = "resolving_environment"
    RUNNING_VALIDATION = "running_validation"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationJobRequest(BaseModel):
    job_id: str
    tenant_id: str
    model_id: str
    model_gcs_path: str
    dataset_gcs_path: str
    requirements_gcs_path: Optional[str] = None
    framework: Optional[str] = None
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    model_type: Optional[str] = None
    numeric_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    model_params: Dict[str, Any] = Field(default_factory=dict)
    dimensions: Dict[str, Any] = Field(default_factory=dict)


class RuntimePlan(BaseModel):
    adapter: str
    framework: str
    python_version: str
    base_image: str
    dependency_hash: str
    image_tag: Optional[str] = None
    execution_mode: str = "local"
    requirements_source: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ValidationMetrics(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    predictions_preview: List[Any] = Field(default_factory=list)
    classes: List[Any] = Field(default_factory=list)
    row_count: Optional[int] = None
    feature_count: Optional[int] = None


class ValidationJobRecord(BaseModel):
    job_id: str
    status: JobStatus
    request: ValidationJobRequest
    runtime_plan: Optional[RuntimePlan] = None
    result: Optional[ValidationMetrics] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

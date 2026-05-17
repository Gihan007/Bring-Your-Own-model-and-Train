import pytest

from app.models import JobStatus, ValidationJobRequest
from app.store import InMemoryJobStore


def test_job_store_rejects_duplicate_job_id():
    store = InMemoryJobStore()
    request = ValidationJobRequest(
        job_id="duplicate",
        tenant_id="tenant",
        model_id="model",
        model_gcs_path="model.joblib",
        dataset_gcs_path="validation.csv",
    )

    first = store.create(request)

    assert first.status == JobStatus.QUEUED
    with pytest.raises(ValueError):
        store.create(request)


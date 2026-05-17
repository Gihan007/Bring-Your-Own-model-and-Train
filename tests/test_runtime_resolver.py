from app.models import ValidationJobRequest
from app.runtime.resolver import parse_requirements, resolve_runtime_plan


def test_parse_requirements_extracts_package_versions():
    packages = parse_requirements(
        """
        pandas==2.2.2
        scikit-learn==1.5.1
        # comment
        joblib>=1.4
        """
    )

    assert packages["pandas"] == "2.2.2"
    assert packages["scikit-learn"] == "1.5.1"
    assert packages["joblib"] == "1.4"


def test_resolve_runtime_plan_for_sklearn_payload():
    request = ValidationJobRequest(
        job_id="job-1",
        tenant_id="tenant-1",
        model_id="model-1",
        model_gcs_path="gs://bucket/model.joblib",
        dataset_gcs_path="gs://bucket/validation.csv",
        requirements_gcs_path="gs://bucket/requirements.txt",
        framework="sklearn",
        task_type="classification",
        target_column="Risk",
    )

    plan = resolve_runtime_plan(
        request,
        "pandas==2.2.2\njoblib==1.4.2\nscikit-learn==1.5.1\n",
    )

    assert plan.adapter == "sklearn_joblib"
    assert plan.framework == "sklearn"
    assert plan.python_version == "3.10"
    assert plan.base_image == "python:3.10-slim"
    assert len(plan.dependency_hash) == 16
    assert plan.image_tag == f"byom-runtime:{plan.dependency_hash}"


def test_resolve_runtime_plan_infers_sklearn_from_joblib_extension():
    request = ValidationJobRequest(
        job_id="job-2",
        tenant_id="tenant-1",
        model_id="model-1",
        model_gcs_path="model.joblib",
        dataset_gcs_path="validation.csv",
        target_column="Risk",
    )

    plan = resolve_runtime_plan(request, "")

    assert plan.framework == "sklearn"
    assert plan.adapter == "sklearn_joblib"


def test_resolve_runtime_plan_supports_requested_frameworks():
    cases = [
        ("xgboost", "model.xgb", "xgboost", "xgboost==2.1.1\n", "3.10"),
        ("onnx", "model.onnx", "onnx", "onnxruntime==1.18.1\n", "3.10"),
        ("tensorflow", "model.keras", "tensorflow", "tensorflow==2.16.1\n", "3.10"),
        ("pytorch", "model.pt", "pytorch_torchscript", "torch==2.3.1\n", "3.10"),
        ("huggingface", "model.zip", "huggingface_transformers", "transformers==4.44.2\n", "3.10"),
    ]

    for framework, model_path, adapter, requirements, python_version in cases:
        request = ValidationJobRequest(
            job_id=f"job-{framework}",
            tenant_id="tenant-1",
            model_id="model-1",
            model_gcs_path=model_path,
            dataset_gcs_path="validation.csv",
            framework=framework,
            target_column="target",
        )

        plan = resolve_runtime_plan(request, requirements)

        assert plan.adapter == adapter
        assert plan.python_version == python_version


def test_resolve_runtime_plan_infers_common_model_extensions():
    cases = [
        ("model.xgb", "xgboost"),
        ("model.onnx", "onnx"),
        ("model.keras", "tensorflow"),
        ("model.pt", "pytorch_torchscript"),
    ]

    for model_path, adapter in cases:
        request = ValidationJobRequest(
            job_id=f"job-{adapter}",
            tenant_id="tenant-1",
            model_id="model-1",
            model_gcs_path=model_path,
            dataset_gcs_path="validation.csv",
            target_column="target",
        )

        plan = resolve_runtime_plan(request, "")

        assert plan.adapter == adapter

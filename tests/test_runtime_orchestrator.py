from app.models import RuntimePlan
from app.runtime.orchestrator import RuntimeOrchestrationAgent


def test_orchestrator_recommends_new_python_for_requires_python_error():
    agent = RuntimeOrchestrationAgent()
    plan = RuntimePlan(
        adapter="sklearn_joblib",
        framework="sklearn",
        python_version="3.10",
        base_image="python:3.10-slim",
        dependency_hash="oldhash",
        image_tag="byom-runtime:oldhash",
        execution_mode="docker",
    )
    error = "ERROR: Ignored versions: scikit-learn==1.8.0 Requires-Python >=3.11"

    diagnosis = agent._diagnose(error)
    next_plan = agent._next_plan(plan, "scikit-learn==1.8.0\n", diagnosis)

    assert diagnosis["type"] == "python_version_too_low"
    assert next_plan is not None
    assert next_plan.python_version == "3.11"
    assert next_plan.base_image == "python:3.11-slim"
    assert next_plan.image_tag == f"byom-runtime:{next_plan.dependency_hash}"


def test_orchestrator_diagnoses_missing_pickle_supporting_code():
    agent = RuntimeOrchestrationAgent()

    diagnosis = agent._diagnose("AttributeError: Can't get attribute 'ModelGroup' on <module 'x'>")

    assert diagnosis == {
        "type": "missing_pickle_supporting_code",
        "missing_class": "ModelGroup",
    }


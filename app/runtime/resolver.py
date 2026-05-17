from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.models import RuntimePlan, ValidationJobRequest


DEFAULT_PYTHON_VERSION = "3.10"

FRAMEWORK_ADAPTERS: Dict[str, str] = {
    "sklearn": "sklearn_joblib",
    "scikit-learn": "sklearn_joblib",
    "xgboost": "xgboost",
    "onnx": "onnx",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "pytorch": "pytorch_torchscript",
    "torch": "pytorch_torchscript",
    "huggingface": "huggingface_transformers",
    "transformers": "huggingface_transformers",
}

PACKAGE_FRAMEWORK_HINTS: Dict[str, str] = {
    "scikit-learn": "sklearn",
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "tensorflow": "tensorflow",
    "torch": "pytorch",
    "onnxruntime": "onnx",
    "onnx": "onnx",
    "transformers": "huggingface",
}

PYTHON_COMPATIBILITY_RULES: List[Tuple[str, str, str]] = [
    ("tensorflow", r"^2\.[0-4](\.|$)", "3.8"),
    ("tensorflow", r"^2\.(5|6|7|8|9|10|11|12)(\.|$)", "3.9"),
    ("tensorflow", r"^2\.(13|14|15|16)(\.|$)", "3.10"),
    ("scikit-learn", r"^0\.(20|21|22|23|24)(\.|$)", "3.8"),
    ("scikit-learn", r"^1\.(0|1|2)(\.|$)", "3.9"),
    ("scikit-learn", r"^1\.(3|4|5|6)(\.|$)", "3.10"),
    ("scikit-learn", r"^1\.(7|8)(\.|$)", "3.11"),
    ("torch", r"^1\.(8|9|10|11|12|13)(\.|$)", "3.9"),
    ("torch", r"^2\.", "3.10"),
    ("xgboost", r"^1\.", "3.9"),
    ("xgboost", r"^2\.", "3.10"),
    ("onnxruntime", r"^1\.", "3.10"),
    ("transformers", r"^4\.", "3.10"),
]


def resolve_runtime_plan(request: ValidationJobRequest, requirements_text: str = "") -> RuntimePlan:
    packages = parse_requirements(requirements_text)
    framework = normalize_framework(request.framework) or infer_framework(packages, request.model_gcs_path)
    adapter = FRAMEWORK_ADAPTERS.get(framework or "")

    warnings: List[str] = []
    if not framework:
        raise ValueError("framework could not be inferred from payload, model path, or requirements.txt")
    if not adapter:
        raise ValueError(f"unsupported framework for prototype: {framework}")

    python_version = infer_python_version(packages)
    if python_version == DEFAULT_PYTHON_VERSION and requirements_text.strip():
        warnings.append("python_version was not provided; selected default/inferred Python 3.10")
    if not requirements_text.strip():
        warnings.append("requirements.txt was not provided; runtime plan uses adapter defaults only")

    dependency_hash = build_dependency_hash(python_version, requirements_text)
    base_image = f"python:{python_version}-slim"
    image_tag = f"byom-runtime:{dependency_hash}"

    return RuntimePlan(
        adapter=adapter,
        framework=framework,
        python_version=python_version,
        base_image=base_image,
        dependency_hash=dependency_hash,
        image_tag=image_tag,
        requirements_source=request.requirements_gcs_path,
        warnings=warnings,
    )


def normalize_framework(framework: Optional[str]) -> Optional[str]:
    if not framework:
        return None
    value = framework.strip().lower()
    if value in ("sklearn", "scikit_learn", "scikit-learn"):
        return "sklearn"
    if value in ("torch", "pytorch", "torchscript"):
        return "pytorch"
    if value in ("tensorflow", "tf", "keras"):
        return "tensorflow"
    if value in ("huggingface", "hf", "transformers"):
        return "huggingface"
    return value


def infer_framework(packages: Dict[str, Optional[str]], model_path: str) -> Optional[str]:
    for package_name in packages:
        hint = PACKAGE_FRAMEWORK_HINTS.get(package_name)
        if hint:
            return hint
    suffix = Path(model_path).suffix.lower()
    if suffix in (".joblib", ".pkl", ".pickle"):
        return "sklearn"
    if suffix in (".xgb", ".ubj"):
        return "xgboost"
    if suffix == ".onnx":
        return "onnx"
    if suffix in (".keras", ".h5"):
        return "tensorflow"
    if suffix in (".pt", ".pth", ".torchscript"):
        return "pytorch"
    if suffix == ".zip":
        return None
    return None


def infer_python_version(packages: Dict[str, Optional[str]]) -> str:
    for package_name, version in packages.items():
        if not version:
            continue
        normalized_name = "scikit-learn" if package_name == "sklearn" else package_name
        for rule_package, version_pattern, python_version in PYTHON_COMPATIBILITY_RULES:
            if normalized_name == rule_package and re.match(version_pattern, version):
                return python_version
    return DEFAULT_PYTHON_VERSION


def parse_requirements(requirements_text: str) -> Dict[str, Optional[str]]:
    packages: Dict[str, Optional[str]] = {}
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?:==|~=|>=|<=|>|<)?\s*([A-Za-z0-9_.!*+-]+)?", line)
        if not match:
            continue
        name = match.group(1).lower().replace("_", "-")
        version = match.group(2)
        packages[name] = version
    return packages


def build_dependency_hash(python_version: str, requirements_text: str) -> str:
    normalized_lines = sorted(normalize_requirement_lines(requirements_text))
    payload = "\n".join([python_version, *normalized_lines])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalize_requirement_lines(requirements_text: str) -> Iterable[str]:
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            yield line

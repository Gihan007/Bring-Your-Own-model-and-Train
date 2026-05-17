from __future__ import annotations

from app.adapters.base import ValidationAdapter
from app.adapters.huggingface_adapter import HuggingFaceAdapter
from app.adapters.onnx_adapter import OnnxAdapter
from app.adapters.pytorch_adapter import PyTorchAdapter
from app.adapters.sklearn_joblib import SklearnJoblibAdapter
from app.adapters.tensorflow_adapter import TensorFlowAdapter
from app.adapters.xgboost_adapter import XGBoostAdapter


def get_adapter(name: str) -> ValidationAdapter:
    if name == "sklearn_joblib":
        return SklearnJoblibAdapter()
    if name == "xgboost":
        return XGBoostAdapter()
    if name == "onnx":
        return OnnxAdapter()
    if name == "tensorflow":
        return TensorFlowAdapter()
    if name == "pytorch_torchscript":
        return PyTorchAdapter()
    if name == "huggingface_transformers":
        return HuggingFaceAdapter()
    raise ValueError(f"adapter is not registered: {name}")

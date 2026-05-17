from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.utils import (
    build_metrics,
    decode_predictions,
    import_numpy,
    load_dataset,
    prepare_tabular_features,
    select_feature_columns,
    write_result,
)
from app.models import ValidationJobRequest, ValidationMetrics


class OnnxAdapter:
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        ort = self._lazy_import()
        np = import_numpy()

        dataset = load_dataset(request, dataset_path)
        feature_columns = select_feature_columns(request, dataset)
        x_values = prepare_tabular_features(request, dataset, feature_columns)
        y_true = dataset[request.target_column]

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = request.dimensions.get("onnx_input_name") or session.get_inputs()[0].name
        output_name = request.dimensions.get("onnx_output_name")
        output_names = [output_name] if output_name else None

        raw_outputs = session.run(output_names, {input_name: x_values.to_numpy(dtype=np.float32)})
        raw_predictions = self._select_prediction_output(raw_outputs)
        predictions = decode_predictions(request, raw_predictions)
        result = build_metrics(
            request,
            y_true,
            predictions,
            row_count=int(len(dataset)),
            feature_count=int(x_values.shape[1]),
            extra_metrics={"onnx_outputs": len(raw_outputs)},
        )
        write_result(artifacts_dir, result)
        return result

    def _select_prediction_output(self, raw_outputs: Any) -> Any:
        first = raw_outputs[0]
        if isinstance(first, list) and first and isinstance(first[0], dict):
            return [max(row, key=row.get) for row in first]
        return first

    def _lazy_import(self) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnx adapter dependency is missing: onnxruntime") from exc
        return ort


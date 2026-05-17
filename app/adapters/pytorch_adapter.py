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


class PyTorchAdapter:
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        torch = self._lazy_import()
        np = import_numpy()

        dataset = load_dataset(request, dataset_path)
        feature_columns = select_feature_columns(request, dataset)
        x_values = prepare_tabular_features(request, dataset, feature_columns)
        y_true = dataset[request.target_column]

        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        tensor = torch.tensor(x_values.to_numpy(dtype=np.float32))
        with torch.no_grad():
            raw_predictions = model(tensor).detach().cpu().numpy()

        predictions = decode_predictions(request, raw_predictions)
        result = build_metrics(
            request,
            y_true,
            predictions,
            row_count=int(len(dataset)),
            feature_count=int(x_values.shape[1]),
        )
        write_result(artifacts_dir, result)
        return result

    def _lazy_import(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("pytorch adapter dependency is missing: torch") from exc
        return torch


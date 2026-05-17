from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    import kagglehub

    path = Path(kagglehub.model_download("samarthpujari/logistic-regression/scikitLearn/pkl_file"))
    project_root = Path(__file__).resolve().parents[1]
    case_dir = project_root / "samples" / "cases" / "kaggle_logistic_regression"
    case_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(path / "logistic_regression_model.pkl", case_dir / "logistic_regression_model.pkl")
    shutil.copy2(path / "logistic_regression.py", case_dir / "source_logistic_regression.py")
    print(f"downloaded Kaggle model to {case_dir}")


if __name__ == "__main__":
    main()


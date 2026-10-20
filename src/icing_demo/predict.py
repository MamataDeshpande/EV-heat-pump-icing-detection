"""Run a saved model on measurements with matching input columns."""

import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from .schema import FEATURES, validate_dataframe


def predict_file(csv_file: Path, model_dir: Path, destination: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_file)
    validate_dataframe(data, labeled=False)
    feature_names = tuple(json.loads((model_dir / "feature_names.json").read_text(encoding="utf-8")))
    if feature_names != FEATURES:
        raise ValueError("Unexpected model input schema")
    model = XGBClassifier()
    model.load_model(str(model_dir / "icing_xgboost.json"))
    result = data.copy()
    result["predicted_icing"] = model.predict(data.loc[:, list(FEATURES)])
    result["icing_probability"] = model.predict_proba(data.loc[:, list(FEATURES)])[:, 1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict icing from generic numerical inputs")
    parser.add_argument("--data", default="data/synthetic_recordings.csv")
    parser.add_argument("--model", default="outputs/model")
    parser.add_argument("--out", default="outputs/predictions.csv")
    args = parser.parse_args()
    result = predict_file(Path(args.data), Path(args.model), Path(args.out))
    print(f"Saved {len(result)} predictions to {args.out}")


if __name__ == "__main__":
    main()

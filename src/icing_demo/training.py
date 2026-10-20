"""Train an XGBoost demonstration classifier using independent recording holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .schema import FEATURES, GROUP, LABEL, validate_dataframe


def split_by_recording(df: pd.DataFrame, *, test_size: float = 0.25, seed: int = 42):
    """No rows from the same recording may appear in both sets."""
    validate_dataframe(df)
    rec_labels = df.groupby(GROUP, sort=True)[LABEL].first()
    if rec_labels.value_counts().min() < 2:
        raise ValueError("Need at least two independent recordings per class")
    train_ids, test_ids = train_test_split(
        rec_labels.index.to_numpy(), test_size=test_size,
        random_state=seed, stratify=rec_labels.to_numpy(),
    )
    train = df.loc[df[GROUP].isin(train_ids)].copy()
    test = df.loc[df[GROUP].isin(test_ids)].copy()
    if set(train[GROUP]) & set(test[GROUP]):
        raise AssertionError("Recording leakage")
    return train, test


def train_and_evaluate(df: pd.DataFrame, out_dir: Path, *, seed: int = 42) -> dict:
    train, test = split_by_recording(df, seed=seed)
    model = XGBClassifier(
        n_estimators=90, max_depth=3, learning_rate=0.07,
        subsample=0.9, colsample_bytree=1.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=seed, tree_method="hist", n_jobs=2,
    )
    # Gradient-boosted trees do not generally need StandardScaler.
    model.fit(train.loc[:, list(FEATURES)], train[LABEL])
    predictions = model.predict(test.loc[:, list(FEATURES)])
    probabilities = model.predict_proba(test.loc[:, list(FEATURES)])[:, 1]
    test = test.assign(predicted_icing=predictions, icing_probability=probabilities)

    metrics = {
        "dataset": "SYNTHETIC DEMONSTRATION — NOT THESIS RESULTS",
        "validation": "independent synthetic recordings",
        "training_recordings": int(train[GROUP].nunique()),
        "test_recordings": int(test[GROUP].nunique()),
        "training_samples": int(len(train)),
        "test_samples": int(len(test)),
        "accuracy": float(accuracy_score(test[LABEL], predictions)),
        "precision": float(precision_score(test[LABEL], predictions, zero_division=0)),
        "recall": float(recall_score(test[LABEL], predictions, zero_division=0)),
        "f1": float(f1_score(test[LABEL], predictions, zero_division=0)),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
            test[LABEL], predictions, labels=[0, 1]
        ).tolist(),
    }
    if "restart_transient_context_only" in test.columns:
        transient = test.loc[
            (test["restart_transient_context_only"].astype(str).str.lower() == "true")
            & (test[LABEL] == 0)
        ]
        metrics["non_iced_restart_samples_in_test"] = int(len(transient))
        metrics["non_iced_restart_false_alarms_in_test"] = int(
            transient["predicted_icing"].sum()
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "icing_xgboost.json"))
    (out_dir / "feature_names.json").write_text(
        json.dumps(list(FEATURES), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    test.to_csv(out_dir / "held_out_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate synthetic XGBoost demo")
    parser.add_argument("--data", default="data/synthetic_recordings.csv")
    parser.add_argument("--out", default="outputs/model")
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(pd.read_csv(args.data), Path(args.out)), indent=2))


if __name__ == "__main__":
    main()

"""Fit Combined LR and select C using the reconstructed Week 9 validation set."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "artifacts/week09_rebuilt"
HISTORY = ["last_correct", "recent_accuracy", "prefix_accuracy"]
TIMING = ["time_latest", "time_recent", "time_prefix"]
FEATURES = HISTORY + TIMING


def main():
    split_path = OUT / "participant_split.csv"
    original_path = PROJECT / "artifacts/splits/participant_split.csv"
    windows_path = PROJECT / "data/processed/development_windows_with_timing.csv"
    base_path = PROJECT / "data/processed/development_windows.csv"

    split = pd.read_csv(split_path)
    original = pd.read_csv(original_path)
    for frame in [split, original]:
        if frame["ID"].isna().any() or frame["ID"].duplicated().any():
            raise ValueError("Split IDs must be present and unique.")

    if original["split"].value_counts().to_dict() != {"development": 995, "test": 249}:
        raise ValueError("Expected 995 development IDs and 249 test IDs.")
    if split["split"].value_counts().to_dict() != {"train": 746, "validation": 249, "test": 249}:
        raise ValueError("Run scripts/week09/make_split.py first.")

    test_ids = set(split.loc[split["split"] == "test", "ID"])
    original_test_ids = set(original.loc[original["split"] == "test", "ID"])
    if set(split["ID"]) != set(original["ID"]) or test_ids != original_test_ids:
        raise ValueError("Participant IDs or test membership changed.")

    windows = pd.read_csv(windows_path)
    base = pd.read_csv(base_path)
    pd.testing.assert_frame_equal(windows[base.columns], base)
    if not np.isfinite(windows[FEATURES].to_numpy()).all():
        raise ValueError("All numeric features must be finite.")
    if not windows["label"].isin([0, 1]).all():
        raise ValueError("Window labels must be 0 or 1.")

    dev = split.loc[split["split"].isin(["train", "validation"])]
    if set(windows["participant_id"]) != set(dev["ID"]):
        raise ValueError("Windows must contain exactly the development participants.")

    keys = ["participant_id", "block", "anchor"]
    if (
        windows[keys].isna().any().any()
        or windows.duplicated(keys).any()
        or not windows["block"].isin([1, 2]).all()
        or not windows["anchor"].isin([3, 4, 5, 6, 7]).all()
        or not windows.groupby("participant_id").size().eq(10).all()
    ):
        raise ValueError("Expected ten unique block/anchor windows per participant.")

    frame = windows.merge(
        dev, left_on="participant_id", right_on="ID", validate="many_to_one"
    )
    if not (frame["group"] == frame["Group"]).all():
        raise ValueError("Instruction groups disagree across input files.")

    frame["instruction"] = "speed"
    frame.loc[frame["group"] == frame["block"], "instruction"] = "accuracy"
    frame["context"] = (
        frame["block"].astype(str) + "_"
        + frame["anchor"].astype(str) + "_" + frame["instruction"]
    )
    train = frame.loc[frame["split"] == "train"]
    valid = frame.loc[frame["split"] == "validation"]
    print(f"Training: {len(train)} windows, {int(train['label'].sum())} positive")
    print(f"Validation: {len(valid)} windows, {int(valid['label'].sum())} positive")

    metrics = []
    predictions = valid[keys + ["label"]].copy()

    for c in [0.1, 1.0]:
        # Fit the encoder, scaler, and LR using training participants only.
        preprocess = ColumnTransformer([
            ("context", OneHotEncoder(handle_unknown="ignore"), ["context"]),
            ("numeric", StandardScaler(), FEATURES),
        ])
        model = make_pipeline(
            preprocess, LogisticRegression(C=c, max_iter=1000)
        )
        model.fit(train[["context"] + FEATURES], train["label"])
        probability = model.predict_proba(valid[["context"] + FEATURES])[:, 1]

        predictions[f"prob_C_{c}"] = probability
        metrics.append({
            "model": "Combined LR",
            "C": c,
            "validation_log_loss": log_loss(valid["label"], probability),
            "validation_brier": brier_score_loss(valid["label"], probability),
            "validation_auc": roc_auc_score(valid["label"], probability),
        })

    table = pd.DataFrame(metrics)
    best_index = table["validation_log_loss"].idxmin()
    table["selected"] = table.index == best_index
    table.to_csv(OUT / "combined_validation_metrics.csv", index=False)
    predictions.to_csv(OUT / "combined_validation_predictions.csv", index=False)

    # Record the exact input and script bytes for later report checks.
    checksums = {}
    for path in [split_path, original_path, windows_path, base_path, Path(__file__).resolve()]:
        relative_path = path.relative_to(PROJECT).as_posix()
        checksums[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "script": "scripts/week09/train_combined.py",
        "function": "main",
        "command": "python scripts/week09/train_combined.py",
        "split_status": "reconstructed",
        "training_participants": int(train["participant_id"].nunique()),
        "validation_participants": int(valid["participant_id"].nunique()),
        "categorical_feature": "block_anchor_instruction",
        "numeric_features": FEATURES,
        "selection": "minimum validation log loss within Combined LR only",
        "selected_C": float(table.loc[best_index, "C"]),
        "test_evaluated": False,
        "model_saved": False,
        "sha256": checksums,
    }
    (OUT / "combined_run.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    print(table.to_string(index=False))
    print("Saved: artifacts/week09_rebuilt/combined_validation_metrics.csv")
    print("Saved: artifacts/week09_rebuilt/combined_validation_predictions.csv")
    print("Saved: artifacts/week09_rebuilt/combined_run.json")


if __name__ == "__main__":
    main()


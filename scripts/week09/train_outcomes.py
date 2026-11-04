"""Fit Outcomes LR on 746 participants and select C on 249 validation participants."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "artifacts" / "week09_rebuilt"
HISTORY = ["last_correct", "recent_accuracy", "prefix_accuracy"]


def main():
    split = pd.read_csv(OUT / "participant_split.csv")
    original = pd.read_csv(PROJECT / "artifacts/splits/participant_split.csv")
    for frame in [split, original]:
        if frame["ID"].isna().any() or frame["ID"].duplicated().any():
            raise ValueError("Split IDs must be present and unique.")
    if split["split"].value_counts().to_dict() != {"train": 746, "validation": 249, "test": 249}:
        raise ValueError("Run make_split.py first.")
    if set(split["ID"]) != set(original["ID"]) or set(split.loc[split["split"] == "test","ID"]) != set(original.loc[original["split"] == "test","ID"]):
        raise ValueError("Participant IDs or test membership do not match the source split.")

    windows = pd.read_csv(PROJECT / "data/processed/development_windows.csv")
    dev = split.loc[split["split"].isin(["train", "validation"])]
    if set(windows["participant_id"]) != set(dev["ID"]):
        raise ValueError("The window file must contain exactly the development participants.")
    keys = ["participant_id", "block", "anchor"]
    if (windows.duplicated(keys).any() or not windows["block"].isin([1, 2]).all()
            or not windows["anchor"].isin(range(3, 8)).all()
            or not windows.groupby("participant_id").size().eq(10).all()):
        raise ValueError("Expected ten unique block/anchor windows per participant.")

    frame = windows.merge(dev, left_on="participant_id", right_on="ID", validate="many_to_one")
    if not (frame["group"] == frame["Group"]).all():
        raise ValueError("Instruction groups disagree across input files.")
    frame["instruction"] = "speed"
    frame.loc[frame["group"] == frame["block"], "instruction"] = "accuracy"
    frame["context"] = (frame["block"].astype(str) + "_" + frame["anchor"].astype(str)
                        + "_" + frame["instruction"])
    train = frame.loc[frame["split"] == "train"]
    valid = frame.loc[frame["split"] == "validation"]
    print(f"Training: {len(train)} windows, {int(train['label'].sum())} positive")
    print(f"Validation: {len(valid)} windows, {int(valid['label'].sum())} positive")
    metrics = []
    predictions = valid[keys + ["label"]].copy()
    for c in [0.1, 1.0]:
        preprocess = ColumnTransformer([
            ("context", OneHotEncoder(handle_unknown="ignore"), ["context"]),
            ("history", StandardScaler(), HISTORY),
        ])
        model = make_pipeline(preprocess, LogisticRegression(C=c, max_iter=1000))
        model.fit(train[["context"] + HISTORY], train["label"])
        probability = model.predict_proba(valid[["context"] + HISTORY])[:, 1]
        predictions[f"prob_C_{c}"] = probability
        metrics.append({"model": "Outcomes LR", "C": c,
                        "validation_log_loss": log_loss(valid["label"], probability),
                        "validation_brier": brier_score_loss(valid["label"], probability),
                        "validation_auc": roc_auc_score(valid["label"], probability)})
    table = pd.DataFrame(metrics)
    best_index = table["validation_log_loss"].idxmin()
    table["selected"] = table.index == best_index
    table.to_csv(OUT / "outcomes_validation_metrics.csv", index=False)
    predictions.to_csv(OUT / "outcomes_validation_predictions.csv", index=False)
    record = {"run_utc": datetime.now(timezone.utc).isoformat(), "pandas": pd.__version__,
              "scikit_learn": sklearn.__version__, "script": "scripts/week09/train_outcomes.py",
              "function": "main", "test_evaluated": False, "split_status": "reconstructed",
              "selection": "minimum validation log loss within Outcomes LR only",
              "selected_C": float(table.loc[best_index, "C"])}
    (OUT / "outcomes_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

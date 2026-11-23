"""Shared settings for the reconstructed participant-level experiment."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
RESULTS = ROOT / "Results"
SPLITS = RESULTS / "splits"
FEATURES = RESULTS / "features"
FIT = RESULTS / "fit"
TABLES = RESULTS / "tables"
MODEL_SEED = 42
BOOT_SEED = 2026
DRAWS = 2000
LR_GRID = (0.1, 1.0)
DEPTHS = (3, 5, 8)
LEAVES = (20, 50)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_table(path):
    return pd.read_csv(path, float_precision="round_trip")


def save_table(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


def record(stage, inputs, outputs, **details):
    folder = RESULTS / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    def checksums(paths):
        return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}
    dependencies = {
        "01_prepare_data": [],
        "02_build_features": ["response_data"],
        "03_train_models": ["score_models", "tree_models"],
        "04_evaluate_predictions": ["history_tables", "comparison_tables", "score_models", "tree_models"],
        "05_export_results": ["report_content"],
    }
    code = [ROOT / "Code" / (module + ".py") for module in [stage, "config", *dependencies[stage]]]
    payload = dict(stage=stage, completed_utc=datetime.now(timezone.utc).isoformat(),
                   inputs=checksums([*inputs, *code]), outputs=checksums(outputs), **details)
    path = folder / (stage + ".json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {stage}.")

"""Reconstruct each window from the released responses and ordered actions."""
import json

from config import DATA, FEATURES, ROOT, SPLITS, digest, record, save_table
from response_data import audit_items, build_windows, load_sources, prepare_splits, reuse_processed

FROZEN = {
    "participant_split.csv": "56404c3a6b905a90f407cf45f92a98a2c42615e6b7347f1c5870c29a3745ee01",
    "outer_folds.csv": "305e48222ac705741639e3b20e663a3f948774044a5c1d67aeb8ae22a04ed536",
}


def main():
    for name, checksum in FROZEN.items():
        if digest(SPLITS / name) != checksum:
            raise ValueError(f"Frozen participant assignments differ in {name}.")
    raw, logs, keys = load_sources(ROOT)
    _, dev = prepare_splits(raw, ROOT)
    items, audit = audit_items(raw, logs, keys)
    items = items.loc[items.participant_id.isin(dev.ID)].copy()
    windows, checks = reuse_processed(build_windows(items, dev), ROOT)
    save_table(items, FEATURES / "completed_items.csv")
    save_table(windows, FEATURES / "development_windows.csv")
    save_table(dev, FEATURES / "development_folds.csv")
    save_table(checks, FEATURES / "feature_checks.csv")
    audit.update(development_participants=len(dev), windows=len(windows), events=int(windows.label.sum()))
    (FEATURES / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    inputs = [*SPLITS.glob("*.csv"), *(DATA / "source").glob("*.csv"),
              *(DATA / "reference").glob("development_windows*.csv")]
    record("02_build_features", inputs, list(FEATURES.glob("*")))


if __name__ == "__main__":
    main()

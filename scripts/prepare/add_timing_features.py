"""Add completed-item timing features to the development windows."""
from math import isfinite, log1p
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
TIMING_COLUMNS = ["time_latest", "time_recent", "time_prefix"]


def get_timing_features(person, block, anchor):
    """Use only completed items within the current block."""
    log_times = []

    for position in range(1, anchor + 1):
        item = (block - 1) * 10 + position
        column = f"T_MR{item:02d}_{block:02d}"
        seconds = float(person[column]) / 1000

        if not isfinite(seconds) or seconds <= 0:
            raise ValueError(
                f"Invalid response time for ID {person.name}, column {column}."
            )

        log_times.append(log1p(seconds))

    latest = log_times[-1]
    recent = sum(log_times[-3:]) / 3
    prefix = sum(log_times) / len(log_times)
    return latest, recent, prefix


def main():
    raw_path = PROJECT / "data/raw/1_main_dt/tte_data.csv"
    windows_path = PROJECT / "data/processed/development_windows.csv"
    split_path = PROJECT / "artifacts/splits/participant_split.csv"
    output_path = PROJECT / "data/processed/development_windows_with_timing.csv"

    raw = pd.read_csv(raw_path, index_col="ID")
    windows = pd.read_csv(windows_path)
    split = pd.read_csv(split_path)

    if raw.index.hasnans or not raw.index.is_unique:
        raise ValueError("Raw data must contain one row per participant ID.")
    if split["ID"].isna().any() or split["ID"].duplicated().any():
        raise ValueError("Split IDs must be present and unique.")
    if split["split"].value_counts().to_dict() != {"development": 995, "test": 249}:
        raise ValueError("Expected 995 development participants and 249 test participants.")

    development_ids = set(split.loc[split["split"] == "development", "ID"])
    if set(windows["participant_id"]) != development_ids:
        raise ValueError("Windows must contain exactly the development participants.")
    if not development_ids.issubset(set(raw.index)):
        raise ValueError("Some development IDs are missing from the raw data.")

    keys = ["participant_id", "block", "anchor"]
    if windows[keys].isna().any().any() or windows.duplicated(keys).any():
        raise ValueError("Window identifiers must be complete and unique.")
    if not windows["block"].isin([1, 2]).all():
        raise ValueError("Expected block 1 or 2.")
    if not windows["anchor"].isin([3, 4, 5, 6, 7]).all():
        raise ValueError("Expected anchors 3 through 7.")
    if not windows.groupby("participant_id").size().eq(10).all():
        raise ValueError("Expected ten windows per development participant.")

    raw = raw.loc[raw.index.isin(development_ids)]
    latest_times = []
    recent_times = []
    prefix_times = []

    for _, window in windows.iterrows():
        participant_id = int(window["participant_id"])
        block = int(window["block"])
        anchor = int(window["anchor"])
        person = raw.loc[participant_id]

        latest, recent, prefix = get_timing_features(person, block, anchor)
        latest_times.append(latest)
        recent_times.append(recent)
        prefix_times.append(prefix)

    result = windows.copy()
    result["time_latest"] = latest_times
    result["time_recent"] = recent_times
    result["time_prefix"] = prefix_times
    result.to_csv(output_path, index=False)

    print(f"Development participants: {result['participant_id'].nunique()}")
    print(f"Total windows: {len(result)}")
    print(f"Positive windows: {int(result['label'].sum())}")
    print("Added timing features: " + ", ".join(TIMING_COLUMNS))
    print("First three windows:")
    print(result[keys + TIMING_COLUMNS].head(3).to_string(index=False))
    print("Saved: data/processed/development_windows_with_timing.csv")


if __name__ == "__main__":
    main()


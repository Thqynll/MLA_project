"""Create a documented Week 9 reconstruction from the existing participant split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "artifacts" / "week09_rebuilt"
SEED = 42


def main():
    source = pd.read_csv(PROJECT / "artifacts" / "splits" / "participant_split.csv",
                         usecols=["ID", "Group", "split"])
    if source.isna().any().any() or source["ID"].duplicated().any():
        raise ValueError("Expected one complete record per participant.")
    if source["split"].value_counts().to_dict() != {"development": 995, "test": 249}:
        raise ValueError("Expected 995 development IDs and 249 test IDs.")

    dev = source.loc[source["split"] == "development"]
    train_ids, valid_ids = train_test_split(
        dev["ID"], test_size=249, stratify=dev["Group"].astype(str), random_state=SEED
    )
    result = source.copy()
    result.loc[result["ID"].isin(train_ids), "split"] = "train"
    result.loc[result["ID"].isin(valid_ids), "split"] = "validation"

    counts = result["split"].value_counts().to_dict()
    if counts != {"train": 746, "validation": 249, "test": 249}:
        raise ValueError("Unexpected reconstructed split sizes.")
    test = result.loc[result["split"] == "test", ["ID"]]
    if set(test["ID"]) != set(source.loc[source["split"] == "test", "ID"]):
        raise ValueError("Test membership changed.")

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "participant_split.csv", index=False)
    test.to_csv(OUT / "test_ids.csv", index=False)
    print("Week 9 reconstructed split")
    for name in ["train", "validation", "test"]:
        print(f"{name}: {counts[name]} participants")
    print("Test IDs match the input split.")
    print("Saved to artifacts/week09_rebuilt/")


if __name__ == "__main__":
    main()

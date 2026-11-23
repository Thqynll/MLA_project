"""Verify the published release and retain the fields used in the analysis."""
import argparse
from io import BytesIO
from pathlib import Path
import hashlib
from urllib.request import urlopen

import pandas as pd

from config import DATA, digest, record

SOURCES = {
    "tte_data.csv": ("p25gr", "main_responses.csv",
                     "288515597704dc9279a3202fd16ad81ac8a5a5eab12c7c8863ae93e61505e1a3"),
    "mr_logdata.csv": ("pw87s", "action_log.csv",
                       "0f27c344f41a7804239461167e99ecf1df24da7b709c853f3ecf4a8f9ca2e4b4"),
    "mr_itemsolutions.csv": ("yvx7k", "scoring_keys.csv",
                             "e5fb1b59fcafd1cd9e8513c012733e23c38328d124982f9fedf4ab9991b678d8"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--release", type=Path, help="Directory containing the three original OSF files.")
    group.add_argument("--download", action="store_true", help="Read the original files from OSF.")
    args = parser.parse_args()
    folder = DATA / "source"
    folder.mkdir(exist_ok=True, parents=True)
    columns = {
        "tte_data.csv": ["ID", "Group"] + [
            f"{field}_MR{i:02d}_{1 if i <= 10 else 2:02d}"
            for field in ("X", "Y", "T") for i in range(1, 21)],
        "mr_logdata.csv": ["ID", "item", "elementOrder", "action", "elementCounter", "cumulTime"],
        "mr_itemsolutions.csv": ["item", "solutionCode"],
    }
    outputs = []
    for name, (osf_id, target, checksum) in SOURCES.items():
        path = folder / target
        if args.release or args.download:
            if args.release:
                matches = list(args.release.rglob(name))
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one release file named {name}.")
                content = matches[0].read_bytes()
            else:
                with urlopen(f"https://osf.io/download/{osf_id}/", timeout=90) as response:
                    content = response.read()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f"Release checksum mismatch for {name}.")
            frame = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)
            selected = frame.loc[:, columns[name]]
            if "item" in selected:
                selected = selected.loc[selected.item.isin([f"MR{i:02d}" for i in range(1, 21)])]
            selected.to_csv(path, index=False, lineterminator="\n")
        if list(pd.read_csv(path, nrows=0).columns) != columns[name]:
            raise ValueError(f"Unexpected fields in {target}.")
        outputs.append(path)
    manifest = DATA / "source_checksums.csv"
    inventory = pd.read_csv(manifest)
    if sorted(inventory.file) != sorted(path.name for path in outputs):
        raise ValueError("The source manifest must identify all three input files.")
    for row in inventory.itertuples(index=False):
        if digest(folder / row.file) != row.sha256:
            raise ValueError(f"Prepared source checksum mismatch for {row.file}.")
    record("01_prepare_data", [manifest], outputs,
           release_sha256={name: value[2] for name, value in SOURCES.items()})


if __name__ == "__main__":
    main()

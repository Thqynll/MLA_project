"""Reconstruct the basic development windows while preserving matching saved bytes."""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'scripts/week10'))
from w10_data import audit_items, build_windows, load_sources, prepare_splits


def main():
    main_data, logs, keys = load_sources(PROJECT)
    _, dev = prepare_splits(main_data, PROJECT)
    items, _ = audit_items(main_data, logs, keys)
    windows = build_windows(items.loc[items['participant_id'].isin(dev['ID'])], dev)
    columns = ['participant_id', 'group', 'block', 'anchor', 'recent_correct',
               'last_correct', 'recent_accuracy', 'prefix_accuracy', 'label']
    basic = windows[columns]
    path = PROJECT / 'data/processed/development_windows.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), basic)
        print('Existing windows match reconstruction; saved file preserved.')
    else:
        basic.to_csv(path, index=False)
        print('Saved: data/processed/development_windows.csv')
    print(f'Development participants: {len(dev)}')
    print(f'Total windows: {len(basic)}')
    print(f"Positive windows: {int(basic['label'].sum())}")


if __name__ == '__main__':
    main()

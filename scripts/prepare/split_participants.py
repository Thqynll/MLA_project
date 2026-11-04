"""Create missing split/fold files, or validate existing frozen assignments."""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'scripts/week10'))
from w10_data import prepare_splits


def main():
    people = pd.read_csv(PROJECT / 'data/raw/1_main_dt/tte_data.csv', usecols=['ID', 'Group'])
    split, dev = prepare_splits(people, PROJECT)
    print(f'Development participants: {len(dev)}')
    print(f"Held-out test participants: {int((split['split'] == 'test').sum())}")
    print('Frozen files: artifacts/splits/participant_split.csv and outer_folds.csv')


if __name__ == '__main__':
    main()

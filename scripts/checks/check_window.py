"""Describe the published participant 1535 example without fitting or scoring models."""
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def main():
    raw = pd.read_csv(PROJECT / 'data/raw/1_main_dt/tte_data.csv', index_col='ID', keep_default_na=False)
    person = raw.loc[1535]
    rows = []
    for block in [1, 2]:
        scores = []
        seconds = []
        for position in range(1, 11):
            item = (block - 1) * 10 + position
            value = str(person[f'Y_MR{item:02d}_{block:02d}']).strip()
            scores.append(int(value or '0'))
            seconds.append(float(person[f'T_MR{item:02d}_{block:02d}']) / 1000)
        for anchor in range(3, 8):
            history = scores[anchor - 3:anchor]
            future = scores[anchor:anchor + 3]
            rows.append({'participant_id': 1535, 'block': block, 'anchor': anchor,
                         'history': str(history), 'recent_seconds': str(seconds[anchor - 3:anchor]),
                         'future': str(future), 'label': int(sum(future) < 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print('Descriptive example only; no model evaluation of this frozen-test participant.')


if __name__ == '__main__':
    main()

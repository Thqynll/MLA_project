"""Inspect one development OOF case and double its latest completed response time."""
import argparse
import json
from math import log1p
from config import FEATURES, FIT, TABLES

import joblib
import pandas as pd

from response_data import make_window
from score_models import add_relative




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', type=int, default=None, help='Development participant ID; default first selected case.')
    parser.add_argument('--block', type=int, default=None)
    parser.add_argument('--anchor', type=int, default=None)
    args = parser.parse_args()
    cases = pd.read_csv(TABLES / 'case_review_10.csv')
    first = cases.head(1).to_dict('records')[0]
    participant_id = args.id if args.id is not None else int(first['participant_id'])
    block = args.block if args.block is not None else int(first['block'])
    anchor = args.anchor if args.anchor is not None else int(first['anchor'])
    if block not in [1, 2] or anchor not in [3, 4, 5, 6, 7]:
        raise ValueError('Use block 1 or 2 and anchor 3 through 7.')
    items = pd.read_csv(FEATURES / 'completed_items.csv', float_precision='round_trip')
    rows = items.loc[(items['participant_id'] == participant_id) & (items['block'] == block)]
    if len(rows) != 10:
        raise ValueError('Choose a development ID; this tool does not evaluate frozen test participants.')
    records = rows.sort_values('position').to_dict('records')
    original = pd.DataFrame([make_window(records, anchor)])
    records[anchor - 1]['seconds'] *= 2
    records[anchor - 1]['log_time'] = log1p(records[anchor - 1]['seconds'])
    changed = pd.DataFrame([make_window(records, anchor)])
    folds = pd.read_csv(FEATURES / 'development_folds.csv')
    fold = int(folds.loc[folds['ID'] == participant_id, 'outer_fold'].item())
    result = {'participant_id': participant_id, 'block': block, 'anchor': anchor,
              'outer_fold': fold, 'label': int(original.loc[0, 'label']),
              'recent_correct': int(original.loc[0, 'recent_correct']),
              'latest_seconds_original': records[anchor - 1]['seconds'] / 2,
              'latest_seconds_doubled': records[anchor - 1]['seconds']}
    for slug in ['outcomes', 'combined']:
        bundle = joblib.load(FIT / 'models' / f'fold_{fold}_{slug}.joblib')
        features = ['context'] + bundle['columns']
        before = bundle['model'].predict_proba(add_relative(original, bundle['item_medians'])[features])[0, 1]
        after = bundle['model'].predict_proba(add_relative(changed, bundle['item_medians'])[features])[0, 1]
        result[slug] = {'original_probability': before, 'perturbed_probability': after, 'difference': after - before}
    print(json.dumps(result, indent=2))
    print('A fixed-model feature sensitivity check; this is not a causal intervention estimate.')


if __name__ == '__main__':
    main()

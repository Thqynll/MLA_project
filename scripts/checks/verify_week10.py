"""Independently check completed Week 10 outputs and saved fold models."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'scripts/week10'))
from w10_models import FEATURES, PREDICTIONS, add_relative, item_medians


def main():
    out = PROJECT / 'artifacts/week10_rebuilt'
    manifest = json.loads((out / 'run_manifest.json').read_text())
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() == digest, path
    for path, digest in manifest['result_sha256'].items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == digest, path
    oof = pd.read_csv(out / 'oof_predictions.csv', float_precision='round_trip')
    items = pd.read_csv(out / 'development_items.csv', float_precision='round_trip')
    split = pd.read_csv(PROJECT / 'artifacts/splits/participant_split.csv')
    dev = pd.read_csv(out / 'outer_folds_used.csv')
    test_ids = set(split.loc[split['split'] == 'test', 'ID'])
    assert not set(oof['participant_id']) & test_ids
    assert len(oof) == 9950 and oof['participant_id'].nunique() == 995
    assert int(oof['label'].sum()) == 6587
    assert oof['recent_correct'].value_counts().sort_index().tolist() == [4165, 1456, 1803, 2526]
    assert not oof.duplicated(['participant_id', 'block', 'anchor']).any()
    assert (oof['outer_fold'] == oof['participant_id'].map(dev.set_index('ID')['outer_fold'])).all()
    inner = pd.read_csv(out / 'inner_folds.csv')
    tuning = pd.read_csv(out / 'tuning_results.csv')
    saved_medians = pd.read_csv(out / 'training_item_medians.csv', float_precision='round_trip')
    for fold in range(1, 6):
        train_ids = set(dev.loc[dev['outer_fold'] != fold, 'ID'])
        valid_ids = set(dev.loc[dev['outer_fold'] == fold, 'ID'])
        assignments = inner.loc[inner['outer_fold'] == fold]
        assert len(assignments) == 796 and not assignments['ID'].duplicated().any()
        assert set(assignments['ID']) == train_ids
        assert not train_ids & valid_ids and not train_ids & test_ids
        for inner_fold in [0, 1, 2, 3]:
            ids = train_ids if inner_fold == 0 else set(assignments.loc[assignments['inner_fold'] != inner_fold, 'ID'])
            expected = item_medians(items, ids).sort_index()
            actual = saved_medians.loc[(saved_medians['outer_fold'] == fold) & (saved_medians['inner_fold'] == inner_fold)]
            np.testing.assert_allclose(actual.sort_values('item')['median_log_time'], expected, rtol=1e-12)
        valid = oof.loc[oof['outer_fold'] == fold]
        for name, columns in FEATURES.items():
            selection = tuning.loc[(tuning['outer_fold'] == fold) & (tuning['model'] == name)]
            assert selection['selected'].sum() == 1
            assert selection.loc[selection['selected'], 'inner_log_loss'].item() == selection['inner_log_loss'].min()
            bundle = joblib.load(out / 'models' / f"fold_{fold}_{PREDICTIONS[name][2:]}.joblib")
            assert set(bundle['training_ids']) == train_ids
            assert set(bundle['validation_ids']) == valid_ids
            assert bundle['model'].named_steps['standardscaler'].n_samples_seen_ == 7960
            frame = add_relative(valid, bundle['item_medians'])
            probability = bundle['model'].predict_proba(frame[['context'] + columns])[:, 1]
            np.testing.assert_allclose(probability, valid[PREDICTIONS[name]], rtol=1e-10, atol=1e-12)
    overall = pd.read_csv(out / 'tables/table04_overall_metrics.csv')
    for row in overall.itertuples(index=False):
        probability = oof[PREDICTIONS[row.model]]
        np.testing.assert_allclose([row.log_loss, row.brier, row.auc],
                                   [log_loss(oof['label'], probability), brier_score_loss(oof['label'], probability),
                                    roc_auc_score(oof['label'], probability)], rtol=1e-12)
    paired = pd.read_csv(out / 'tables/table05_overall_differences.csv')
    indexed = overall.set_index('model')
    for row in paired.itertuples(index=False):
        expected = indexed.loc[row.first, 'log_loss'] - indexed.loc[row.second, 'log_loss']
        np.testing.assert_allclose(row.log_loss_difference, expected, atol=1e-12)
    cases = pd.read_csv(out / 'tables/case_review_10.csv')
    assert len(cases) == 10 and cases['selection'].value_counts().tolist() == [5, 5]
    eligible = oof.loc[oof['recent_correct'] == 1].copy()
    eligible['score'] = (2 * eligible['label'] - 1) * (eligible['p_combined'] - eligible['p_outcomes'])
    positive = eligible.loc[eligible['score'] > 0].sort_values(['score', 'participant_id', 'block', 'anchor'], ascending=[False, True, True, True]).head(5)
    negative = eligible.loc[eligible['score'] < 0].sort_values(['score', 'participant_id', 'block', 'anchor']).head(5)
    keys = ['participant_id', 'block', 'anchor']
    assert cases[keys].values.tolist() == pd.concat([positive, negative])[keys].values.tolist()
    budget = pd.read_csv(out / 'tables/table10_review_budgets.csv')
    for row in budget.itertuples(index=False):
        assert row.selected_windows == {0.05: 495, 0.10: 995, 0.20: 1990}[row.budget]
    record = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'checks': ['source and result hashes', '995 people and 9950 unique windows', 'test exclusion',
                         'outer and inner participant separation', 'all inner and outer training medians',
                         'all 25 saved models reproduce their OOF predictions', 'C selection',
                         'overall metrics and paired point differences', 'case selection', 'review budgets']}
    (out / 'verification.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('PASS: sources, folds, models, metrics, cases, and budgets verified.')


if __name__ == '__main__':
    main()

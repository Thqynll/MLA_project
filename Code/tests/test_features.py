"""Meaningful checks for time boundaries, fold statistics, and paired uncertainty."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_data import make_window
from history_tables import paired_differences
from score_models import ACTIONS, HISTORY, PREDICTIONS, RAW_TIME, fit_lr, item_medians


class Week10Tests(unittest.TestCase):
    def records(self):
        return [{'participant_id': 1, 'group': 1, 'block': 1, 'instruction': 'accuracy',
                 'correct': i % 2, 'seconds': 10.0 + i, 'log_time': np.log1p(10 + i),
                 'log_toggle_count': np.log1p(i + 1), 'deselection_fraction': 0.2,
                 'log_first_action_seconds': np.log1p(2 + i)} for i in range(10)]

    def test_future_items_never_change_current_features(self):
        columns = HISTORY + RAW_TIME + ACTIONS + ['person_relative_latest', 'person_relative_recent']
        for anchor in range(3, 8):
            original = self.records()
            changed = copy.deepcopy(original)
            for item in changed[anchor:]:
                item.update(correct=0, seconds=999999, log_time=99,
                            log_toggle_count=99, deselection_fraction=1, log_first_action_seconds=99)
            a, b = make_window(original, anchor), make_window(changed, anchor)
            for column in columns:
                self.assertEqual(a[column], b[column])
            self.assertEqual(b['label'], 1)
            expected = int(sum(item['correct'] for item in original[anchor:anchor + 3]) < 2)
            self.assertEqual(a['label'], expected)

    def test_held_out_times_do_not_change_training_medians(self):
        data = pd.DataFrame({'participant_id': [1, 2, 3], 'item': [1, 1, 1], 'log_time': [1., 3., 10.]})
        before = item_medians(data, {1, 2})
        data.loc[data['participant_id'] == 3, 'log_time'] = 1000000
        pd.testing.assert_series_equal(before, item_medians(data, {1, 2}))
        self.assertEqual(before.loc[1], 2.)

    def test_scaler_learns_only_training_features(self):
        train = pd.DataFrame({'context': ['a', 'b'] * 6, 'last_correct': [0, 1] * 6,
                              'recent_accuracy': [0.0, 0.5, 1.0] * 4,
                              'prefix_accuracy': [0.2, 0.4, 0.6, 0.8] * 3,
                              'label': [0, 0, 1, 0, 1, 1] * 2})
        model = fit_lr(train, 'Outcomes LR', 0.1)
        scaler = model.named_steps['standardscaler']
        np.testing.assert_allclose(scaler.mean_[-3:], train[HISTORY].mean())
        before = scaler.mean_.copy()
        held_out = train.copy()
        held_out[HISTORY] = 1000.0
        model.predict_proba(held_out[['context'] + HISTORY])
        np.testing.assert_array_equal(before, scaler.mean_)

    def test_identical_predictions_have_zero_paired_intervals(self):
        rows = []
        for participant_id in range(8):
            for recent in range(4):
                row = {'participant_id': participant_id, 'recent_correct': recent,
                       'label': (participant_id + recent) % 2}
                for column in PREDICTIONS.values():
                    row[column] = 0.15 + 0.1 * recent
                rows.append(row)
        result = paired_differences(pd.DataFrame(rows), list(range(8)), draws=100)
        columns = [c for c in result if c.startswith('log_loss_') or c.startswith('brier_')]
        np.testing.assert_allclose(result[columns], 0.0, atol=1e-15)


if __name__ == '__main__':
    unittest.main()

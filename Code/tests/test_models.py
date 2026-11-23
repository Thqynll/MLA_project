"""Check training-only tree encoding and paired uncertainty with known predictions."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tree_models import fit_cart, predict
from score_models import HISTORY
from comparison_tables import PREDICTIONS, paired


class Week11Tests(unittest.TestCase):
    def test_cart_training_encoder_and_feature_boundary(self):
        train = pd.DataFrame({'context': ['a', 'b'] * 10, 'last_correct': [0, 1] * 10,
                              'recent_accuracy': [0., .5] * 10, 'prefix_accuracy': [.3, .7] * 10,
                              'label': [0, 1] * 10})
        model = fit_cart(train, HISTORY, {'max_depth': 3, 'min_samples_leaf': 2})
        encoder = model.named_steps['columntransformer'].named_transformers_['context']
        held_out = train.copy()
        held_out['context'] = 'unseen'
        before = predict(model, held_out, HISTORY)
        held_out['future_correct_1'] = 1
        held_out['label'] = 1-held_out.label
        np.testing.assert_array_equal(before, predict(model, held_out, HISTORY))
        self.assertEqual(encoder.categories_[0].tolist(), ['a', 'b'])
        self.assertTrue(np.isfinite(before).all())

    def test_paired_identical_probabilities_zero_difference(self):
        rows = []
        for participant in range(8):
            for recent in range(4):
                row = {'participant_id': participant, 'recent_correct': recent,
                       'label': (participant+recent) % 2}
                row.update({column: .1+.2*recent for column in PREDICTIONS.values()})
                rows.append(row)
        result = paired(pd.DataFrame(rows), list(range(8)))
        columns = [c for c in result if c.startswith(('log_loss_', 'brier_'))]
        np.testing.assert_allclose(result[columns], 0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()

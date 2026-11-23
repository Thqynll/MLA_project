"""Rebuild LR and Decision Tree comparisons; never evaluate the frozen test cohort."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from config import BOOT_SEED, DRAWS
from score_models import PREDICTIONS as LR_PREDICTIONS
from history_tables import scope_rows
from tree_models import PREDICTIONS as CART_PREDICTIONS

PREDICTIONS = {**LR_PREDICTIONS, **CART_PREDICTIONS}
SCOPES = ['all', '0/3', '1/3', '2/3', '3/3', '1/3+2/3']
PAIRS = [('Outcomes LR', 'History prior'), ('Combined LR', 'Outcomes LR'),
         ('Process LR', 'Combined LR'), ('Outcomes CART', 'Outcomes LR'),
         ('Combined CART', 'Combined LR'), ('Process CART', 'Process LR'),
         ('Combined CART', 'Outcomes CART'), ('Process CART', 'Combined CART')]


def metrics(oof):
    rows = []
    for scope in SCOPES:
        data = scope_rows(oof, scope)
        for name, column in PREDICTIONS.items():
            rows.append({'scope': scope, 'model': name, 'participants': data.participant_id.nunique(),
                         'windows': len(data), 'events': int(data.label.sum()),
                         'non_events': int((data.label == 0).sum()), 'event_rate': data.label.mean(),
                         'log_loss': log_loss(data.label, data[column], labels=[0, 1]),
                         'brier': brier_score_loss(data.label, data[column]),
                         'auc': roc_auc_score(data.label, data[column])})
    return pd.DataFrame(rows)


def paired(oof, ids):
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.array([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                        for _ in range(DRAWS)])
    rows = []
    for scope in SCOPES:
        data = scope_rows(oof, scope)
        counts = data.groupby('participant_id').size().reindex(ids, fill_value=0).to_numpy()
        denominator = weights @ counts
        usable = denominator > 0
        y = data.label.to_numpy()
        for first, second in PAIRS:
            p = np.clip(data[PREDICTIONS[first]].to_numpy(), 1e-15, 1-1e-15)
            q = np.clip(data[PREDICTIONS[second]].to_numpy(), 1e-15, 1-1e-15)
            record = {'scope': scope, 'first': first, 'second': second,
                      'windows': len(data), 'bootstrap_draws': DRAWS, 'valid_draws': int(usable.sum())}
            for metric, delta in [
                ('log_loss', -y*np.log(p)-(1-y)*np.log1p(-p)+y*np.log(q)+(1-y)*np.log1p(-q)),
                ('brier', (y-p)**2-(y-q)**2),
            ]:
                sums = pd.Series(delta, index=data.participant_id).groupby(level=0).sum().reindex(ids, fill_value=0)
                distribution = (weights @ sums.to_numpy())[usable] / denominator[usable]
                low, high = np.percentile(distribution, [2.5, 97.5])
                record.update({metric+'_difference': delta.mean(), metric+'_ci_low': low, metric+'_ci_high': high})
            rows.append(record)
    return pd.DataFrame(rows)


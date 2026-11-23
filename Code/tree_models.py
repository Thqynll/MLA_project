"""CART on the exact saved participant folds used by Logistic Regression."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_SEED, DEPTHS, LEAVES
from score_models import HISTORY, RAW_TIME, ACTIONS

FEATURES = {
    'Outcomes CART': HISTORY,
    'Combined CART': HISTORY + RAW_TIME,
    'Process CART': HISTORY + RAW_TIME + ACTIONS,
}
PREDICTIONS = {name: 'p_' + name.lower().replace(' ', '_') for name in FEATURES}
GRID = [{'max_depth': depth, 'min_samples_leaf': leaf}
        for depth in DEPTHS for leaf in LEAVES]


def fit_cart(train, columns, parameters):
    encoder = ColumnTransformer([
        ('context', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['context']),
        ('numeric', 'passthrough', columns),
    ])
    model = make_pipeline(encoder, DecisionTreeClassifier(
        criterion='gini', random_state=MODEL_SEED, **parameters))
    model.fit(train[['context'] + columns], train['label'])
    return model


def predict(model, frame, columns):
    # Same numerical clipping in tuning, saved probabilities, metrics and bootstrap.
    return np.clip(model.predict_proba(frame[['context'] + columns])[:, 1], 1e-15, 1 - 1e-15)


def run_cart(oof, dev, inner, out):
    oof = oof.copy()
    tuning = []
    for name in FEATURES:
        oof[PREDICTIONS[name]] = np.nan
    for fold in range(1, 6):
        train_ids = set(dev.loc[dev.outer_fold != fold, 'ID'])
        valid_ids = set(dev.loc[dev.outer_fold == fold, 'ID'])
        assignments = inner.loc[inner.outer_fold == fold]
        if (train_ids & valid_ids or set(assignments.ID) != train_ids
                or assignments.ID.duplicated().any()):
            raise ValueError('Invalid participant folds.')
        train = oof.loc[oof.participant_id.isin(train_ids)]
        valid = oof.loc[oof.participant_id.isin(valid_ids)]
        for name, columns in FEATURES.items():
            candidates = []
            for parameters in GRID:
                total_loss = 0.
                for inner_fold in range(1, 4):
                    fit_ids = set(assignments.loc[assignments.inner_fold != inner_fold, 'ID'])
                    check_ids = set(assignments.loc[assignments.inner_fold == inner_fold, 'ID'])
                    fit = train.loc[train.participant_id.isin(fit_ids)]
                    check = train.loc[train.participant_id.isin(check_ids)]
                    model = fit_cart(fit, columns, parameters)
                    total_loss += log_loss(check.label, predict(model, check, columns),
                                           labels=[0, 1], normalize=False)
                candidates.append({'outer_fold': fold, 'model': name, **parameters,
                                   'inner_log_loss': total_loss / len(train)})
            best = min(candidates, key=lambda r: (r['inner_log_loss'], r['max_depth'], -r['min_samples_leaf']))
            for candidate in candidates:
                candidate['selected'] = candidate is best
                tuning.append(candidate)
            parameters = {key: best[key] for key in GRID[0]}
            model = fit_cart(train, columns, parameters)
            oof.loc[valid.index, PREDICTIONS[name]] = predict(model, valid, columns)
            joblib.dump({'model': model, 'columns': columns, 'parameters': parameters,
                         'training_ids': sorted(train_ids), 'validation_ids': sorted(valid_ids)},
                        out / 'models' / f'fold_{fold}_{PREDICTIONS[name][2:]}.joblib', compress=3)
        print(f'CART: completed outer fold {fold}/5.', flush=True)
    if not np.isfinite(oof[list(PREDICTIONS.values())].to_numpy()).all():
        raise ValueError('Missing CART OOF predictions.')
    return oof, pd.DataFrame(tuning)

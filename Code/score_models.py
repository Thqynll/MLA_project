"""Fold-specific feature construction, priors, nested tuning, and saved LR fits."""
import warnings
from config import MODEL_SEED, LR_GRID

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HISTORY = ['last_correct', 'recent_accuracy', 'prefix_accuracy']
RAW_TIME = ['time_latest', 'time_recent', 'time_prefix']
RELATIVE = ['item_relative_latest', 'item_relative_recent', 'item_relative_prefix',
            'person_relative_latest', 'person_relative_recent']
ACTIONS = ['action_toggle', 'action_deselection', 'action_latency']
FEATURES = {
    'Outcomes LR': HISTORY,
    'Combined LR': HISTORY + RAW_TIME,
    'Relative LR': HISTORY + RELATIVE,
    'Process LR': HISTORY + RAW_TIME + ACTIONS,
    'Relative Process LR': HISTORY + RELATIVE + ACTIONS,
}
PREDICTIONS = {
    'Global prior': 'p_global_prior', 'Item prior': 'p_item_prior',
    'Window prior': 'p_window_prior', 'History prior': 'p_history_prior',
    'Outcomes LR': 'p_outcomes', 'Combined LR': 'p_combined',
    'Relative LR': 'p_relative', 'Process LR': 'p_process',
    'Relative Process LR': 'p_relative_process',
}


def item_medians(items, train_ids):
    return items.loc[items['participant_id'].isin(train_ids)].groupby('item')['log_time'].median()


def add_relative(windows, medians):
    frame = windows.copy()
    for name in RELATIVE[:3]:
        frame[name] = np.nan
    for block in [1, 2]:
        for anchor in range(3, 8):
            mask = (frame['block'] == block) & (frame['anchor'] == anchor)
            last_item = (block - 1) * 10 + anchor
            recent_items = list(range(last_item - 2, last_item + 1))
            prefix_items = list(range((block - 1) * 10 + 1, last_item + 1))
            frame.loc[mask, 'item_relative_latest'] = frame.loc[mask, 'time_latest'] - medians.loc[last_item]
            frame.loc[mask, 'item_relative_recent'] = frame.loc[mask, 'time_recent'] - medians.loc[recent_items].mean()
            frame.loc[mask, 'item_relative_prefix'] = frame.loc[mask, 'time_prefix'] - medians.loc[prefix_items].mean()
    return frame


def fit_lr(train, name, c):
    columns = FEATURES[name]
    encoder = ColumnTransformer([
        ('context', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['context']),
        ('numeric', 'passthrough', columns),
    ])
    # Match the old Week 10 script's scaling of the entire encoded design matrix.
    # Each encoder and scaler is fitted inside the current training fold.
    pipeline = make_pipeline(encoder, StandardScaler(),
                             LogisticRegression(C=c, max_iter=2000, random_state=MODEL_SEED))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline.fit(train[['context'] + columns], train['label'])
    return pipeline


def prior_predictions(train, valid, training_items):
    output = {'Global prior': np.repeat(train['label'].mean(), len(valid))}
    for name, key in [('Window prior', 'context'), ('History prior', 'recent_correct')]:
        counts = train.groupby(key)['label'].agg(['sum', 'count'])
        rates = (counts['sum'] + 1) / (counts['count'] + 2)
        fallback = (train['label'].sum() + 1) / (len(train) + 2)
        output[name] = valid[key].map(rates).fillna(fallback).to_numpy()
    counts = training_items.groupby(['item', 'instruction'])['correct'].agg(['sum', 'count'])
    error_rates = ((counts['count'] - counts['sum'] + 1) / (counts['count'] + 2)).to_dict()
    probabilities = []
    for row in valid.itertuples(index=False):
        last_item = (row.block - 1) * 10 + row.anchor
        errors = [error_rates[(item, row.instruction)] for item in range(last_item + 1, last_item + 4)]
        a, b, c = errors
        probabilities.append(a * b + a * c + b * c - 2 * a * b * c)
    output['Item prior'] = np.array(probabilities)
    return output


def run_nested_cv(windows, items, dev, out):
    oof = windows.copy()
    for column in PREDICTIONS.values():
        oof[column] = np.nan
    tuning = []
    assignments = []
    median_rows = []
    models_folder = out / 'models'
    models_folder.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        outer_people = dev.loc[dev['outer_fold'] != fold].reset_index(drop=True)
        train_ids = set(outer_people['ID'])
        valid_ids = set(dev.loc[dev['outer_fold'] == fold, 'ID'])
        train = windows.loc[windows['participant_id'].isin(train_ids)].copy()
        valid = windows.loc[windows['participant_id'].isin(valid_ids)].copy()
        medians = item_medians(items, train_ids)
        train_relative = add_relative(train, medians)
        valid_relative = add_relative(valid, medians)
        for item, value in medians.items():
            median_rows.append({'outer_fold': fold, 'inner_fold': 0, 'item': item,
                                'median_log_time': value, 'training_participants': len(train_ids)})

        inner_data = []
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_SEED)
        for inner, (train_rows, valid_rows) in enumerate(inner_cv.split(outer_people['ID'], outer_people['Group']), 1):
            inner_train_ids = set(outer_people.loc[train_rows, 'ID'])
            inner_valid_ids = set(outer_people.loc[valid_rows, 'ID'])
            for participant_id in sorted(inner_valid_ids):
                assignments.append({'outer_fold': fold, 'ID': participant_id, 'inner_fold': inner})
            inner_medians = item_medians(items, inner_train_ids)
            for item, value in inner_medians.items():
                median_rows.append({'outer_fold': fold, 'inner_fold': inner, 'item': item,
                                    'median_log_time': value, 'training_participants': len(inner_train_ids)})
            inner_train = add_relative(train.loc[train['participant_id'].isin(inner_train_ids)], inner_medians)
            inner_valid = add_relative(train.loc[train['participant_id'].isin(inner_valid_ids)], inner_medians)
            inner_data.append((inner_train, inner_valid))

        for name, probability in prior_predictions(train, valid, items.loc[items['participant_id'].isin(train_ids)]).items():
            oof.loc[valid.index, PREDICTIONS[name]] = probability

        for name, columns in FEATURES.items():
            candidates = []
            for c in LR_GRID:
                total_loss = 0.0
                for inner_train, inner_valid in inner_data:
                    model = fit_lr(inner_train, name, c)
                    probability = model.predict_proba(inner_valid[['context'] + columns])[:, 1]
                    total_loss += log_loss(inner_valid['label'], probability, normalize=False)
                candidates.append({'outer_fold': fold, 'model': name, 'C': c,
                                   'inner_log_loss': total_loss / len(train)})
            best = min(candidates, key=lambda row: (row['inner_log_loss'], row['C']))
            for candidate in candidates:
                candidate['selected'] = candidate['C'] == best['C']
                tuning.append(candidate)
            model = fit_lr(train_relative, name, best['C'])
            probability = model.predict_proba(valid_relative[['context'] + columns])[:, 1]
            oof.loc[valid.index, PREDICTIONS[name]] = probability
            slug = PREDICTIONS[name][2:]
            joblib.dump({'model': model, 'model_name': name, 'columns': columns,
                         'item_medians': medians, 'C': best['C'], 'outer_fold': fold,
                         'training_ids': sorted(train_ids), 'validation_ids': sorted(valid_ids)},
                        models_folder / f'fold_{fold}_{slug}.joblib', compress=3)
        print(f'Completed outer fold {fold}/5: 796 training, 199 held-out participants.', flush=True)

    values = oof[list(PREDICTIONS.values())].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Every development window must have a valid OOF probability.')
    return oof, pd.DataFrame(tuning), pd.DataFrame(assignments), pd.DataFrame(median_rows)

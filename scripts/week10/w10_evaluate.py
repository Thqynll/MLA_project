"""Create report tables from fixed participant out-of-fold predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from w10_models import FEATURES, PREDICTIONS

PAIRS = [
    ('Combined LR', 'Outcomes LR'), ('Relative LR', 'Combined LR'),
    ('Process LR', 'Combined LR'), ('Relative Process LR', 'Process LR'),
    ('Relative LR', 'Outcomes LR'), ('Process LR', 'Outcomes LR'),
    ('Relative Process LR', 'Outcomes LR'),
]


def scope_rows(oof, scope):
    if scope == 'all':
        return oof
    if scope == '1/3+2/3':
        return oof.loc[oof['recent_correct'].isin([1, 2])]
    return oof.loc[oof['recent_correct'] == int(scope[0])]


def metric_table(oof, scopes, names):
    rows = []
    for scope in scopes:
        data = scope_rows(oof, scope)
        for name in names:
            probability = data[PREDICTIONS[name]]
            rows.append({
                'scope': scope, 'model': name, 'participants': data['participant_id'].nunique(),
                'windows': len(data), 'events': int(data['label'].sum()),
                'non_events': int((data['label'] == 0).sum()), 'event_rate': data['label'].mean(),
                'log_loss': log_loss(data['label'], probability, labels=[0, 1]),
                'brier': brier_score_loss(data['label'], probability),
                'auc': roc_auc_score(data['label'], probability) if data['label'].nunique() == 2 else np.nan,
            })
    return pd.DataFrame(rows)


def paired_differences(oof, participant_ids, draws=2000, seed=2026):
    """Resample all 995 participant clusters, paired across models and strata."""
    rng = np.random.default_rng(seed)
    n = len(participant_ids)
    weights = np.zeros((draws, n), dtype=np.int32)
    for draw in range(draws):
        sampled = rng.integers(0, n, size=n)
        weights[draw] = np.bincount(sampled, minlength=n)
    rows = []
    for scope in ['all', '0/3', '1/3', '2/3', '3/3', '1/3+2/3']:
        data = scope_rows(oof, scope)
        label = data['label'].to_numpy()
        counts = data.groupby('participant_id').size().reindex(participant_ids, fill_value=0).to_numpy()
        denominators = weights @ counts
        valid_draws = denominators > 0
        for first, second in PAIRS:
            p_first = np.clip(data[PREDICTIONS[first]].to_numpy(), 1e-15, 1 - 1e-15)
            p_second = np.clip(data[PREDICTIONS[second]].to_numpy(), 1e-15, 1 - 1e-15)
            loss_first = -(label * np.log(p_first) + (1 - label) * np.log1p(-p_first))
            loss_second = -(label * np.log(p_second) + (1 - label) * np.log1p(-p_second))
            record = {'scope': scope, 'first': first, 'second': second, 'windows': len(data),
                      'bootstrap_draws': draws, 'valid_draws': int(valid_draws.sum()), 'bootstrap_seed': seed}
            for metric, difference in [('log_loss', loss_first - loss_second),
                                       ('brier', (label - p_first) ** 2 - (label - p_second) ** 2)]:
                sums = pd.DataFrame({'participant_id': data['participant_id'].to_numpy(), 'difference': difference})
                sums = sums.groupby('participant_id')['difference'].sum().reindex(participant_ids, fill_value=0).to_numpy()
                sample_differences = (weights @ sums)[valid_draws] / denominators[valid_draws]
                lower, upper = np.percentile(sample_differences, [2.5, 97.5])
                record[metric + '_difference'] = difference.mean()
                record[metric + '_ci_low'] = lower
                record[metric + '_ci_high'] = upper
            rows.append(record)
    return pd.DataFrame(rows)


def review_budgets(oof):
    rows = []
    for name in FEATURES:
        probability = PREDICTIONS[name]
        for budget in [0.05, 0.10, 0.20]:
            selected = []
            for fold in range(1, 6):
                data = oof.loc[oof['outer_fold'] == fold]
                data = data.sort_values([probability, 'participant_id', 'block', 'anchor'],
                                        ascending=[False, True, True, True], kind='stable')
                selected.append(data.head(int(np.floor(budget * len(data)))))
            chosen = pd.concat(selected)
            hits = int(chosen['label'].sum())
            rows.append({'model': name, 'budget': budget, 'selected_windows': len(chosen),
                         'selected_fraction': len(chosen) / len(oof),
                         'people': chosen['participant_id'].nunique(),
                         'person_blocks': len(chosen[['participant_id', 'block']].drop_duplicates()),
                         'events_selected': hits, 'precision': hits / len(chosen),
                         'recall': hits / int(oof['label'].sum())})
    return pd.DataFrame(rows)


def select_cases(oof):
    data = oof.loc[oof['recent_correct'] == 1].copy()
    data['probability_change'] = data['p_combined'] - data['p_outcomes']
    data['direction_score'] = (2 * data['label'] - 1) * data['probability_change']
    tie = ['participant_id', 'block', 'anchor']
    helpful = data.loc[data['direction_score'] > 0].sort_values(
        ['direction_score'] + tie, ascending=[False, True, True, True], kind='stable').head(5).copy()
    harmful = data.loc[data['direction_score'] < 0].sort_values(
        ['direction_score'] + tie, ascending=True, kind='stable').head(5).copy()
    if len(helpful) != 5 or len(harmful) != 5:
        raise ValueError('The predefined case rule requires five cases in each direction.')
    helpful['selection'] = 'toward_observed_label'
    harmful['selection'] = 'away_from_observed_label'
    cases = pd.concat([helpful, harmful], ignore_index=True)
    cases['observation'] = [
        f"Combined {'raised' if row.probability_change > 0 else 'lowered'} event probability by "
        f"{abs(row.probability_change):.4f}; observed future-event label was {row.label}."
        for row in cases.itertuples(index=False)
    ]
    # These observations are generated descriptions; members still need to read each case.
    cases['reviewed_by_member'] = ''
    columns = ['selection', 'participant_id', 'block', 'anchor', 'outer_fold',
               'history_correct_1', 'history_seconds_1', 'history_correct_2', 'history_seconds_2',
               'history_correct_3', 'history_seconds_3', 'p_outcomes', 'p_combined', 'label',
               'probability_change', 'direction_score', 'observation', 'reviewed_by_member']
    return cases[columns]


def create_tables(oof, dev, draws):
    differences = paired_differences(oof, dev['ID'].tolist(), draws)
    by_history = metric_table(oof, ['0/3', '1/3', '2/3', '3/3'], list(FEATURES))
    tables = {
        'table04_overall_metrics.csv': metric_table(oof, ['all'], list(PREDICTIONS)),
        'table05_overall_differences.csv': differences.loc[differences['scope'] == 'all'],
        'table06_history_outcomes_combined.csv': by_history.loc[by_history['model'].isin(['Outcomes LR', 'Combined LR'])],
        'table07_history_all_lr_metrics.csv': by_history,
        'table07_history_differences.csv': differences.loc[differences['scope'].isin(['0/3', '1/3', '2/3', '3/3'])],
        'table08_relative_timing_differences.csv': differences.loc[(differences['first'] == 'Relative LR') & (differences['second'] == 'Combined LR')],
        'table09_action_differences.csv': differences.loc[((differences['first'] == 'Process LR') & (differences['second'] == 'Combined LR')) | ((differences['first'] == 'Relative Process LR') & (differences['second'] == 'Process LR'))],
        'table10_review_budgets.csv': review_budgets(oof),
        'mixed_history_metrics.csv': metric_table(oof, ['1/3+2/3'], list(FEATURES)),
        'mixed_history_differences.csv': differences.loc[differences['scope'] == '1/3+2/3'],
        'case_review_10.csv': select_cases(oof),
    }
    fold_rows = []
    for fold in range(1, 6):
        frame = metric_table(oof.loc[oof['outer_fold'] == fold], ['all'], list(PREDICTIONS))
        frame.insert(0, 'outer_fold', fold)
        fold_rows.append(frame)
    tables['fold_metrics.csv'] = pd.concat(fold_rows, ignore_index=True)
    return tables


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        values = [f'{value:.8f}' if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append('| ' + ' | '.join(value.replace('|', '/') for value in values) + ' |')
    return '\n'.join(lines)


def write_results(tables, out, run_utc):
    folder = out / 'tables'
    folder.mkdir(parents=True, exist_ok=True)
    sources = []
    functions = {
        'table04_overall_metrics.csv': 'metric_table', 'table05_overall_differences.csv': 'paired_differences',
        'table06_history_outcomes_combined.csv': 'metric_table', 'table07_history_all_lr_metrics.csv': 'metric_table',
        'table07_history_differences.csv': 'paired_differences',
        'table08_relative_timing_differences.csv': 'paired_differences',
        'table09_action_differences.csv': 'paired_differences', 'table10_review_budgets.csv': 'review_budgets',
        'mixed_history_metrics.csv': 'metric_table', 'mixed_history_differences.csv': 'paired_differences',
        'case_review_10.csv': 'select_cases', 'fold_metrics.csv': 'metric_table',
    }
    text = [
        '# Week 10 reconstructed results', '', f'Run started (UTC): {run_utc}', '',
        'These are new executed results from the packaged code. They do not replace old report numbers silently.', '',
        '995 development participants; 5 outer participant folds; 3 inner folds; C in [0.1, 1.0]. '
        'The original 249 test participants are excluded from all fits, tuning, predictions, and model evaluation.', '',
        'All intervals are paired participant-cluster percentile bootstrap intervals, conditional on the fitted OOF predictions. '
        'They are exploratory, unadjusted for multiple comparisons, and do not include refitting or model-selection uncertainty.', '',
        'The pooled 1/3+2/3 Combined-minus-Outcomes comparison is the reporting focus chosen for this reconstruction '
        'using prior feedback. It is not an original preregistered hypothesis or independent confirmation.', '',
        'Case rule fixed before this rerun: in the 1/3 group, select the five largest positive and five most negative '
        'values of (2*y - 1)*(p_Combined - p_Outcomes). Break ties by ID, block, anchor. '
        'Multiple windows from the same participant are permitted. Observations are automatic descriptions; human review is pending.', '',
        'Tables 08 and 09 are supporting tables added by this reconstruction; the older report skipped those numbers.', '',
    ]
    for filename, frame in tables.items():
        frame.to_csv(folder / filename, index=False, float_format='%.17g')
        sources.append({'output': 'tables/' + filename, 'source_file': 'scripts/week10/w10_evaluate.py',
                        'function': functions[filename], 'entrypoint': 'scripts/week10/run_week10.py',
                        'command': 'python scripts/week10/run_week10.py', 'run_utc': run_utc,
                        'prediction_source': 'oof_predictions.csv', 'sha256_source': 'run_manifest.json'})
        text.extend(['## ' + filename, '', 'Source: scripts/week10/w10_evaluate.py / ' + functions[filename]
                     + f'; run {run_utc}.', '', markdown_table(frame), ''])
    (out / 'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    pd.DataFrame(sources).to_csv(out / 'table_sources.csv', index=False)

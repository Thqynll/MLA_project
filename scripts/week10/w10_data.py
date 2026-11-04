"""Read, audit, and construct participant-level data for Week 10."""
from math import log1p

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_sources(project):
    main = pd.read_csv(project / 'data/raw/1_main_dt/tte_data.csv', dtype=str,
                       keep_default_na=False)
    main['ID'] = main['ID'].astype(int)
    main['Group'] = main['Group'].astype(int)
    if main['ID'].duplicated().any() or len(main) != 1244:
        raise ValueError('Expected 1,244 unique participants.')
    if not main['Group'].isin([1, 2]).all():
        raise ValueError('Unexpected instruction group.')
    logs = pd.read_csv(project / 'data/raw/2_supplement_dt/mr_logdata.csv')
    keys = pd.read_csv(project / 'data/raw/2_supplement_dt/mr_itemsolutions.csv',
                       dtype=str, keep_default_na=False)
    return main, logs, dict(zip(keys['item'], keys['solutionCode'].str.strip("'")))


def prepare_splits(main, project):
    """Reuse saved assignments; create them only when the files are absent."""
    folder = project / 'artifacts/splits'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'participant_split.csv'
    if not path.exists():
        _, test_ids = train_test_split(main['ID'], test_size=249, random_state=42,
                                       stratify=main['Group'].astype(str))
        split = main[['ID', 'Group']].copy()
        split['split'] = 'development'
        split.loc[split['ID'].isin(test_ids), 'split'] = 'test'
        split.to_csv(path, index=False)
    split = pd.read_csv(path)
    if split.isna().any().any() or split['ID'].duplicated().any():
        raise ValueError('Split records must be complete and unique.')
    if set(split['ID']) != set(main['ID']):
        raise ValueError('Saved split IDs do not match raw participants.')
    if split['split'].value_counts().to_dict() != {'development': 995, 'test': 249}:
        raise ValueError('Expected 995 development and 249 test participants.')
    raw_groups = main.set_index('ID')['Group']
    if not (split['Group'] == split['ID'].map(raw_groups)).all():
        raise ValueError('Instruction groups disagree with the saved split.')
    # Keep the saved row order for reproducible participant folds.
    dev = split.loc[split['split'] == 'development', ['ID', 'Group']].reset_index(drop=True)
    fold_path = folder / 'outer_folds.csv'
    if not fold_path.exists():
        folds = dev.copy()
        folds['outer_fold'] = 0
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_, valid_rows) in enumerate(cv.split(dev['ID'], dev['Group']), 1):
            folds.loc[valid_rows, 'outer_fold'] = fold
        folds.to_csv(fold_path, index=False)
    folds = pd.read_csv(fold_path)
    if folds['ID'].duplicated().any() or set(folds['ID']) != set(dev['ID']):
        raise ValueError('Outer folds must contain exactly the 995 development IDs.')
    if folds['outer_fold'].value_counts().to_dict() != dict.fromkeys(range(1, 6), 199):
        raise ValueError('Expected five outer folds of 199 participants each.')
    if 'Group' in folds and not (folds['Group'] == folds['ID'].map(raw_groups)).all():
        raise ValueError('Outer-fold groups disagree with raw data.')
    dev = dev.merge(folds[['ID', 'outer_fold']], on='ID', validate='one_to_one', sort=False)
    return split, dev


def audit_items(main, logs, answer_keys):
    """Verify main responses against action logs and published scoring keys."""
    people = main.set_index('ID').to_dict('index')
    main_names = {f'MR{i:02d}' for i in range(1, 21)}
    logs = logs.loc[logs['item'].isin(main_names)].copy()
    records = []
    repeated_submits = 0
    for (participant_id, item_name), actions in logs.groupby(['ID', 'item'], sort=False):
        actions = actions.sort_values('elementOrder', kind='stable')
        events = list(actions.itertuples(index=False))
        if actions['elementOrder'].tolist() != list(range(1, len(events) + 1)):
            raise ValueError(f'Non-sequential log order: {participant_id}, {item_name}.')
        submits = [event for event in events if event.action == 'submit']
        if not submits or submits[-1].elementOrder != events[-1].elementOrder:
            raise ValueError('Missing final submit or events after the final submit.')
        repeated_submits += int(len(submits) > 1)
        counters = {}
        response = ['0'] * 20
        deselections = 0
        for event in events:
            if event.action == 'submit':
                continue
            action = str(event.action)
            if not action.startswith('o') or not action[1:].isdigit():
                raise ValueError(f'Unexpected action: {action}.')
            element = int(action[1:])
            if not 0 <= element < 20:
                raise ValueError('Unexpected response element.')
            counters[action] = counters.get(action, 0) + 1
            if counters[action] != event.elementCounter:
                raise ValueError('Log toggle counters do not agree.')
            response[element] = str(counters[action] % 2)
            deselections += int(counters[action] % 2 == 0)
        item = int(item_name[2:])
        block = 1 if item <= 10 else 2
        suffix = f'{item_name}_{block:02d}'
        person = people[int(participant_id)]
        raw_response = person['X_' + suffix].strip()
        raw_score = person['Y_' + suffix].strip()
        if raw_score not in ['', '0', '1']:
            raise ValueError('Unexpected correctness value.')
        response_code = ''.join(response)
        if response_code != (raw_response or '0' * 20):
            raise ValueError('Reconstructed selections disagree with the main data.')
        if raw_score == '' and (raw_response or response_code != '0' * 20):
            raise ValueError('A blank score is not a verified omission.')
        correct = int(raw_score or '0')
        if int(response_code == answer_keys[item_name]) != correct:
            raise ValueError('Key-based score disagrees with the main data.')
        duration_ms = float(person['T_' + suffix])
        if not np.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Response times must be positive and finite.')
        if duration_ms != submits[-1].cumulTime:
            raise ValueError('Main duration differs from final submit time.')
        if not np.isfinite(actions['cumulTime']).all() or (actions['cumulTime'] < 0).any():
            raise ValueError('Invalid cumulative action times.')
        if (actions['cumulTime'].diff().dropna() < 0).any():
            raise ValueError('Cumulative action times must be nondecreasing.')
        toggles = sum(counters.values())
        first_seconds = float(events[0].cumulTime) / 1000
        seconds = duration_ms / 1000
        group = int(person['Group'])
        records.append({
            'participant_id': int(participant_id), 'group': group,
            'item': item, 'block': block, 'position': (item - 1) % 10 + 1,
            'instruction': 'accuracy' if group == block else 'speed',
            'correct': correct, 'omission': int(raw_score == ''),
            'seconds': seconds, 'log_time': log1p(seconds),
            'toggle_count': toggles, 'deselection_fraction': deselections / max(toggles, 1),
            'first_action_seconds': first_seconds,
            'log_toggle_count': log1p(toggles), 'log_first_action_seconds': log1p(first_seconds),
        })
    items = pd.DataFrame(records)
    if set(items['participant_id']) != set(main['ID']) or len(items) != 24880:
        raise ValueError('Expected all 24,880 main participant-item pairs.')
    if items.duplicated(['participant_id', 'item']).any():
        raise ValueError('Duplicate participant-item pairs.')
    if not items.groupby('participant_id').size().eq(20).all():
        raise ValueError('Expected twenty main items per participant.')
    audit = {
        'participants': len(main), 'main_item_responses': len(items),
        'correct': int(items['correct'].sum()), 'omissions': int(items['omission'].sum()),
        'attempted_incorrect': int(((items['correct'] == 0) & (items['omission'] == 0)).sum()),
        'participants_with_no_correct_main_item': int(items.groupby('participant_id')['correct'].sum().eq(0).sum()),
        'main_log_rows': len(logs), 'pairs_with_repeated_submits': repeated_submits,
        'response_key_and_timing_mismatches': 0,
        'audit_scope': 'All released main items; this is a data-integrity audit, not test model evaluation.',
    }
    return items, audit


def make_window(records, anchor):
    """Build features from completed items; future correctness supplies only the label."""
    completed = records[:anchor]
    recent = completed[-3:]
    future = records[anchor:anchor + 3]
    person = records[0]
    window = {
        'participant_id': person['participant_id'], 'group': person['group'],
        'block': person['block'], 'anchor': anchor, 'instruction': person['instruction'],
        'context': f"b{person['block']}_j{anchor}_{person['instruction']}",
        'recent_correct': sum(item['correct'] for item in recent),
        'last_correct': recent[-1]['correct'],
        'recent_accuracy': sum(item['correct'] for item in recent) / 3,
        'prefix_accuracy': sum(item['correct'] for item in completed) / anchor,
        'label': int(sum(item['correct'] for item in future) < 2),
        'time_latest': recent[-1]['log_time'],
        'time_recent': sum(item['log_time'] for item in recent) / 3,
        'time_prefix': sum(item['log_time'] for item in completed) / anchor,
        'action_toggle': sum(item['log_toggle_count'] for item in recent) / 3,
        'action_deselection': sum(item['deselection_fraction'] for item in recent) / 3,
        'action_latency': sum(item['log_first_action_seconds'] for item in recent) / 3,
    }
    window['person_relative_latest'] = window['time_latest'] - window['time_prefix']
    window['person_relative_recent'] = window['time_recent'] - window['time_prefix']
    for position, item in enumerate(recent, 1):
        window[f'history_correct_{position}'] = item['correct']
        window[f'history_seconds_{position}'] = item['seconds']
    for position, item in enumerate(future, 1):
        window[f'future_correct_{position}'] = item['correct']
    return window


def build_windows(items, dev):
    rows = []
    # Reindex the completed windows to the saved development participant order.
    for (_, _), block in items.groupby(['participant_id', 'block'], sort=False):
        records = block.sort_values('position').to_dict('records')
        for anchor in range(3, 8):
            rows.append(make_window(records, anchor))
    windows = pd.DataFrame(rows)
    order = dict(zip(dev['ID'], range(len(dev))))
    windows['participant_order'] = windows['participant_id'].map(order)
    windows = windows.sort_values(['participant_order', 'block', 'anchor']).drop(columns='participant_order')
    windows = windows.reset_index(drop=True)
    windows['outer_fold'] = windows['participant_id'].map(dev.set_index('ID')['outer_fold'])
    if len(windows) != 9950 or windows['outer_fold'].isna().any():
        raise ValueError('Expected 9,950 development windows with assigned outer folds.')
    return windows

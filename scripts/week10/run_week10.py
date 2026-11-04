"""Run the complete reconstructed Week 10 analysis from one entry point."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from w10_data import audit_items, build_windows, load_sources, prepare_splits
from w10_evaluate import create_tables, write_results
from w10_models import FEATURES, PREDICTIONS, run_nested_cv

PROJECT = Path(__file__).resolve().parents[2]


def save_csv(frame, path):
    """Finish and verify a complete temporary CSV before replacing the output."""
    temporary = path.with_suffix('.csv.tmp')
    frame.to_csv(temporary, index=False, float_format='%.17g')
    check = pd.read_csv(temporary, float_precision='round_trip')
    if len(check) != len(frame) or list(check.columns) != list(frame.columns):
        raise ValueError(f'CSV export is incomplete: {path.name}.')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=2000, help='Participant-cluster draws; default 2000.')
    parser.add_argument('--output-dir', default='artifacts/week10_rebuilt', help='Output directory relative to the project root.')
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap draws; use 2000 for the report.')
    out = PROJECT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run_utc = datetime.now(timezone.utc).isoformat()
    plan = {
        'written_before_model_fit_utc': run_utc,
        'status': 'Reconstruction informed by previous results and lecturer feedback; exploratory.',
        'reporting_focus': 'Combined LR minus Outcomes LR log loss in pooled recent-correct groups 1 and 2.',
        'originally_preregistered': False,
        'case_rule': 'Within recent_correct=1, five largest positive and five most negative (2*y-1)*(p_combined-p_outcomes).',
        'tie_break': ['participant_id', 'block', 'anchor'],
        'bootstrap_draws': args.bootstrap, 'bootstrap_seed': 2026,
        'multiplicity_adjustment': None, 'test_evaluation': False,
    }
    (out / 'analysis_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print('[1/4] Loading sources and validating frozen participant assignments.', flush=True)
    main_data, logs, keys = load_sources(PROJECT)
    split, dev = prepare_splits(main_data, PROJECT)
    print('[2/4] Auditing scores, response times, and action sequences.', flush=True)
    items, audit = audit_items(main_data, logs, keys)
    # Whole-release integrity audit is separate from development-only model analysis.
    items = items.loc[items['participant_id'].isin(dev['ID'])].copy()
    windows = build_windows(items, dev)
    audit['development_participants'] = len(dev)
    audit['development_windows'] = len(windows)
    audit['development_events'] = int(windows['label'].sum())
    audit['history_windows'] = {str(k): int(v) for k, v in windows['recent_correct'].value_counts().sort_index().items()}
    (out / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    save_csv(items, out / 'development_items.csv')
    save_csv(windows, out / 'development_windows.csv')
    save_csv(dev, out / 'outer_folds_used.csv')
    (out / 'model_specifications.json').write_text(json.dumps(FEATURES, indent=2), encoding='utf-8')
    print(f"Development: {len(dev)} participants, {len(windows)} windows, {int(windows['label'].sum())} events.", flush=True)
    print('[3/4] Running five outer folds and three inner folds for all five LR models.', flush=True)
    with threadpool_limits(limits=1):
        oof, tuning, inner, medians = run_nested_cv(windows, items, dev, out)
        save_csv(oof, out / 'oof_predictions.csv')
        save_csv(tuning, out / 'tuning_results.csv')
        save_csv(inner, out / 'inner_folds.csv')
        save_csv(medians, out / 'training_item_medians.csv')
        print('[4/4] Producing paired intervals, subgroup tables, budgets, and ten cases.', flush=True)
        tables = create_tables(oof, dev, args.bootstrap)
        write_results(tables, out, run_utc)
    tracked = [PROJECT / 'data/raw/1_main_dt/tte_data.csv',
               PROJECT / 'data/raw/2_supplement_dt/mr_logdata.csv',
               PROJECT / 'data/raw/2_supplement_dt/mr_itemsolutions.csv',
               PROJECT / 'artifacts/splits/participant_split.csv', PROJECT / 'artifacts/splits/outer_folds.csv']
    tracked.extend(sorted((PROJECT / 'scripts/week10').glob('*.py')))
    manifest = {
        'run_started_utc': run_utc, 'run_finished_utc': datetime.now(timezone.utc).isoformat(),
        'entrypoint': 'scripts/week10/run_week10.py',
        'command': f'python scripts/week10/run_week10.py --bootstrap {args.bootstrap} --output-dir {args.output_dir}',
        'status': 'completed', 'test_evaluated': False, 'bootstrap_draws': args.bootstrap,
        'random_seeds': {'participant_split_if_created': 42, 'outer_folds_if_created': 42, 'inner_folds': 42, 'lr': 42, 'bootstrap': 2026},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
        'models': list(PREDICTIONS), 'lr_fits_saved': 25, 'development_participants': len(dev),
        'source_sha256': {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
        'result_sha256': {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['run_manifest.json', 'verification.json']},
    }
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(tables['table04_overall_metrics.csv'][['model', 'log_loss', 'brier', 'auc']].to_string(index=False))
    print(f'Saved complete outputs to {out.relative_to(PROJECT).as_posix()}/', flush=True)


if __name__ == '__main__':
    main()

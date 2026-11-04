# MLA_project — Week 10 reconstruction

A runnable reconstruction of the Week 10 response-process analysis. The ZIP includes source data, code, saved participant assignments, fitted outer-fold models, and executed results. The new outputs are identified as reconstruction results; they are not presented as the original report's exact run.

## Open and run

1. Extract the ZIP to a new folder, for example `D:/MLA_project_week10/`.
2. In VS Code, open the folder containing this README, `data/`, `scripts/`, and `artifacts/`.
3. Select the Python interpreter. Install dependencies only if needed:

```powershell
python -m pip install -r requirements.txt
```

4. Open `scripts/week10/run_week10.py` and click **Run Python File**, or run from the project root:

```powershell
python scripts/week10/run_week10.py
```

5. Verify the completed outputs:

```powershell
python scripts/checks/verify_week10.py
python -m unittest discover -s tests -v
```

Already computed outputs are in `artifacts/week10_rebuilt/`. Open `RESULTS.md` with VS Code's Markdown preview (`Ctrl+Shift+V`) to inspect the tables before running anything. Runtime varies by computer. Re-running replaces the current Week 10 outputs, so retain the delivered ZIP as a checkpoint.

The scripts locate the root from their own file paths. Keep them directly under the documented folders. Do not create `scripts/week10/scripts/week10/` by extracting into the wrong directory.

## What is included

- An integrity audit of all 24,880 main responses: selections, public keys, omissions, durations, and action counters.
- A frozen 995-development / 249-test participant split and five saved outer folds.
- 9,950 development windows, anchors 3 through 7 within each of two blocks.
- Four training-only priors: global, item, window, and recent-history.
- Five LR models: Outcomes, Combined, Relative, Process, and Relative Process.
- Five outer participant folds and three inner participant folds, tuning C in [0.1, 1.0] separately for each LR model.
- Item-relative medians, context encoders, and scalers learned again within every training fold.
- Overall and four-history-group metrics, 2,000 paired participant-bootstrap draws, and 5%, 10%, 20% review budgets.
- Ten OOF cases selected by a documented rule, plus a fixed-model response-time sensitivity command.
- All 25 fitted outer-fold LR pipelines, selected C values, inner assignments, training medians, and input/code checksums.

The held-out 249 people are excluded from all new model fits, tuning, predictions, and evaluation. A whole-release integrity audit is kept separate from model evaluation. No participant is removed because they answered zero items correctly.

## Files to inspect first

| Path | Purpose |
| --- | --- |
| `scripts/week10/run_week10.py` | Single entry point for the full Week 10 run. |
| `scripts/week10/w10_data.py` | Raw-data audit, participant splits, and completed-history windows. |
| `scripts/week10/w10_models.py` | Priors, relative-time features, nested tuning, and model saving. |
| `scripts/week10/w10_evaluate.py` | Metrics, paired intervals, review budgets, and case selection. |
| `artifacts/week10_rebuilt/RESULTS.md` | All new tables with source functions and run date. |
| `artifacts/week10_rebuilt/table_sources.csv` | Table-to-code, command, date, and prediction-file mapping. |
| `artifacts/week10_rebuilt/run_manifest.json` | Versions, seeds, code/data/output hashes, and run completion. |
| `artifacts/week10_rebuilt/verification.json` | Independent verification outcome. |
| `docs/PROTOCOL.md` | Definitions and evaluation boundaries. |
| `docs/REPRODUCTION_NOTES.md` | Relationship to earlier report numbers and remaining work. |

CSV tables retain high precision. Round only for presentation; a displayed upper CI bound of `+0.0000` is not evidence that the full-precision bound is exactly zero.

## Existing Week 9 stage

The earlier `scripts/week09/make_split.py`, `train_outcomes.py`, and `train_combined.py` are included with their executed outputs in `artifacts/week09_rebuilt/`. They use the reconstructed 746/249 training/validation assignment and preserve the same 249 test IDs.

```powershell
python scripts/week09/make_split.py
python scripts/week09/train_outcomes.py
python scripts/prepare/add_timing_features.py
python scripts/week09/train_combined.py
```

Week 10 uses all 995 development participants through nested CV. It does not use the Week 9 train/validation labels for outer-fold assignment. Keep both split files; their roles differ.

## Ten cases and interview practice

Open `artifacts/week10_rebuilt/tables/case_review_10.csv`. The selection rule is in `analysis_plan.json` and above the tables in `RESULTS.md`. The `reviewed_by_member` field is intentionally blank: team members must actually inspect the cases before claiming that review is complete.

```powershell
python scripts/week10/inspect_case.py
python scripts/week10/inspect_case.py --id 2096 --block 1 --anchor 7
```

This uses the saved model that held the selected development person out. It doubles only the latest completed duration and recomputes the timing summaries without refitting. It is a feature sensitivity check, not a causal estimate. ID 1535 is part of the frozen test split, so this tool uses development cases instead. `scripts/checks/check_window.py` retains the descriptive ID 1535 example without fitting or scoring a model.

## Compatibility and source handling

The run's exact versions are in `run_manifest.json` and `environment_reference.txt`. CSV outputs are portable. Load saved joblib pipelines with the matching scikit-learn version, or rerun training in your environment first. The scripts use no `iloc` operations.

When integrating into an existing project, preserve its saved split and outer-fold CSVs. Compare them before copying; do not overwrite frozen assignments merely to obtain matching scores. The main runner reuses existing assignments and only creates them when absent.

Read `docs/DATA_SOURCES.md` for file usage and dataset attribution. Papers and codebooks are references, not training samples. AI assistance and the human-review checklist are recorded in `docs/AI_USE_AND_REVIEW.md`.

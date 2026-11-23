# Response Process Analysis for Assessment Review

Group 04 | Machine Learning Applications | 61FIT4MLA

After an adult completes a matrix-reasoning item, do recorded times and selection actions improve prediction of the next three responses beyond the correctness history? The final analysis is the reconstructed, matched LR and CART comparison. The October Week 10 probabilities are preserved as a separate historical reference.

## Data provenance and processing

| Source record | Published link | Member-confirmed official URL |
| --- | --- | --- |
| Official Data Source / Repository Link | [Much et al., OSF dataset](https://osf.io/9j6hm/overview) | **[Enter verified repository URL here]** |
| Data Paper / Reference Link | [Much et al. (2025), Journal of Open Psychology Data](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.124) | **[Enter verified paper URL here]** |

**License & Redistribution Status.** Much, Mutak, Pohl and Ranger released the dataset under CC BY 4.0. They replaced the Prolific identifiers with study-generated numeric IDs and report that the public release contains no identifying information. The six original data and codebook files in `Data/` are redistributed with attribution. The source register also identifies the local archived release, data paper, and supporting preregistration. [Data provenance and file inventory](Data/README.md) · [Data paper, sections 3.6–3.7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12270193/) · [License](https://creativecommons.org/licenses/by/4.0/)

The sibling archive `MLA_project_pre_restructure_20261121/` holds the local release in `data/raw/`, the data paper in `docs/references/dataset-paper.pdf`, and the supporting preregistration in `docs/references/Preregistration_Psychometric-Model-Validation.pdf`. [Study preregistration](https://osf.io/jm7ne/)

**Data processing selection.** Stage 01 verifies `Data/tte_data.csv`, `mr_logdata.csv` and `mr_itemsolutions.csv` and selects the 20 main items and required fields into `Data/source/`. Stage 02 audits scores and ordered actions, codes verified omissions as zero, rejects inconsistent measurements, and retains valid long durations without trimming. It converts milliseconds to seconds and builds correctness, timing and action histories using only completed items. It checks the saved participant assignments in `Results/splits/` and writes 9,950 windows from 995 development participants to `Results/features/development_windows.csv`. All windows for one person remain in one outer fold. Stage 03 fits preprocessing and tunes models on training participants; `Results/fit/inner_folds.csv` records its inner splits. The 249 historical test participants are excluded from model evaluation. [Preparation script](Code/01_prepare_data.py) · [Window construction](Code/02_build_features.py)

The source manifest records each original filename, file size, SHA-256 checksum and OSF download URL. The processing guide links each stage to its outputs. [Original file manifest](Data/release_checksums.csv) and [data processing guide](Data/README.md#data-processing-selection).

## Contributors

| Group member | Documented individual contribution |
| --- | --- |
| Le Hoang The Quyen | Member evidence pending |
| Tran Gia Thanh | Member evidence pending |
| Nguyen Phu Trong | Member evidence pending |

Each member records the source passage or code-to-result trace they verified in `AI_Log/AI_Use_Log.md`. The ten selected windows and member observations are kept separately so rerunning the analysis preserves manual work.

## Environment

The tested environment is 64-bit CPython 3.14.0 on Windows. All Python packages used for the analysis and PDF export are pinned in `requirements.txt`. Start in the directory containing this README. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

On a POSIX system use `python3 -m venv .venv`, `. .venv/bin/activate`, then `python -m pip install -r requirements.txt`. The paths in the analysis are relative to the project. The PDF renderer uses Matplotlib; TeX is optional.

## Reproduce the report

Run each line after the previous one succeeds. The first command rebuilds the projections from the original CSVs bundled directly in `Data/`. Running it without arguments validates the prepared copies. The remaining steps regenerate the features, fitted models, evaluation tables, figures and report using the frozen assignments and November numerical reference tables.

```text
python -B Code/01_prepare_data.py --release Data
python -B Code/02_build_features.py
python -B Code/03_train_models.py
python -B Code/04_evaluate_predictions.py
python -B Code/05_export_results.py
python -B -m unittest discover -s Code/tests -v
python -B Code/verify_results.py
```

Use one run at a time in a given working copy. Generated outputs replace their previous versions. The frozen participant assignments and member review register are inputs. Participant-level training, validation and test membership can be reconstructed from `Results/splits/`, `Results/features/development_folds.csv` and `Results/fit/inner_folds.csv`. Each outer fold trains on the other four folds. The 249 historical test participants contribute no probabilities to the final evaluation.

## Verify one reported result independently

From the project root, run:

```text
python -c "import pandas as pd; from sklearn.metrics import log_loss; x=pd.read_csv('Results/fit/oof_predictions.csv',float_precision='round_trip'); print(len(x), x.participant_id.nunique(), format(log_loss(x.label,x.p_combined),'.6f'))"
```

Expected output: `9950 995 0.304950`. In Table 1, this is the overall Combined LR log loss. The matching Outcomes LR value is 0.307811. Table 2 reports the paired change and its participant-bootstrap interval. Table 6 uses the original October probabilities in a separate column.

## Three-week development plan

| Week | Files and work | Acceptance evidence |
| --- | --- | --- |
| 1. Data and features | Steps 01 and 02, Data documentation, frozen assignments, response and action audit | Three release checksums, 24,880 audited item responses, 143 omissions, 995 development participants, 9,950 windows and an explicit anchor trace |
| 2. Models and evaluation | Steps 03 and 04, matched nested folds, LR and CART tuning, 2,000 paired participant draws, subgroup comparisons, budgets and ten selected cases | Replay of 40 fitted models, numerical comparison of 14 tables and member review of selected cases |
| 3. Report and delivery | Step 05, report provenance, environment setup, AI disclosure, English-language and release checks | Report PDF and LaTeX source, a fresh environment run, source and output checksums, documented personal contributions and signed case observations |

## Files to open

- `Report/Final_Report.pdf` and `Report/final_report.tex` contain the report.
- `Report/report_sources.csv` identifies the input and generating function for every displayed table and figure.
- `Results/table_sources.csv` maps the full evaluation exports to their source functions and held-out predictions. The stage logs contain the corresponding run times and hashes.
- `Results/figures/timing_by_history.png` shows paired timing effects for all four history groups.
- `Report/member_case_review.csv` retains public window identifiers and space for evidence, interpretation and a decision. Complete it after inspecting the data.
- `Data/README.md` and `AI_Log/AI_Use_Log.md` describe attribution, transformations, AI assistance and human review.

The final assessment package contains no video. The accompanying evidence concerns this protocol and does not establish the effectiveness of a deployment in classrooms.

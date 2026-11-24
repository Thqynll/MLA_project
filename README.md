# Response Process Analysis for Assessment Review

Group 04 | Machine Learning Applications | 61FIT4MLA

We tested whether response times and selection actions improve forecasts based on recent scores. After each eligible matrix-reasoning item, the model estimates the probability that a participant will answer at most one of the next three items correctly.

## Members and contributions

| Member | Contribution |
| --- | --- |
| Le Hoang The Quyen | Checked scores, response times, and outcome labels in three prediction windows. |
| Tran Gia Thanh | Compared model predictions with and without timing against the actual outcomes in three cases. |
| Nguyen Phu Trong | Documented four cases where timing worsened predictions and identified two overlapping windows. |

All ten manual checks, including each reviewer's observations and decisions, are in the [case review register](Results/member_case_review.csv). See the [AI use log](AI_Log/AI_Use_Log.md) for AI assistance and individual source checks.

## Setup

The project was tested on Windows with 64-bit Python 3.14.0. Open a terminal in the project folder containing this README and `Code/`.

### 1. Create and activate a Python environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```sh
python3 -m venv .venv
. .venv/bin/activate
```

### 2. Install the libraries before running the analysis

The project uses pandas and NumPy for data processing, scikit-learn and SciPy for modelling, joblib to save models, threadpoolctl to control training threads, and Matplotlib to draw tables and charts.

If you have the full submission, the package list is already in [requirements.txt](requirements.txt). If that file is missing, create a plain-text file named `requirements.txt` below this README and paste in these lines:

```text
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
fonttools==4.65.0
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.2
narwhals==2.26.0
numpy==2.5.3
packaging==25.0
pandas==3.0.5
pillow==12.0.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.9.1
scipy==1.18.1
six==1.17.0
threadpoolctl==3.6.0
tzdata==2026.3
```

With the environment active, install all the listed packages in one command:

```sh
python -m pip install -r requirements.txt
```

Then [run the analysis](#run-the-analysis) or [check the main result](#reproduce-one-main-result) using the saved predictions.

The dataset is bundled in `Data/`. The analysis uses `Data/tte_data.csv` for responses and durations, `Data/mr_logdata.csv` for actions, and `Data/mr_itemsolutions.csv` for scoring keys. The three accompanying codebooks are also included. File sizes and SHA-256 checksums are listed in `Data/release_checksums.csv`. See the [data guide](Data/DATA_README.md) for processing details.

Much, Mutak, Pohl, and Ranger released the dataset under CC BY 4.0. [OSF dataset](https://osf.io/9j6hm/overview) · [Data paper](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.124) · [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12270193/) · [Study preregistration](https://osf.io/jm7ne/) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

## Run the analysis

To rebuild the November results from the bundled data, run each command from the project root after the previous one succeeds. All five stages run offline once the dependencies are installed. To check Table 1 directly using the included predictions, skip to [Reproduce one main result](#reproduce-one-main-result).

```text
python -B Code/01_prepare_data.py --release Data
python -B Code/02_build_features.py
python -B Code/03_train_models.py
python -B Code/04_evaluate_predictions.py
python -B Code/05_export_results.py
```

Stages 01 and 02 verify the raw files, rebuild `Data/source/`, and create 9,950 prediction windows using the saved participant assignments in `Results/splits/`. Stage 03 trains logistic regression and decision trees on the same participant folds and writes [out-of-fold predictions](Results/fit/oof_predictions.csv). Stage 04 checks the calculated tables against `Data/benchmarks/` and writes them to `Results/tables/`. Stage 05 exports six table images and one chart to `Results/figures/`.

To test the analysis code and check saved predictions against the fitted models, run:

```text
python -B -m unittest discover -s Code/tests -v
python -B Code/verify_results.py
```

## Reproduce one main result

Table 1 reports a log loss of **0.304950** for logistic regression using score history, assessment context, and response times (Combined LR). The same model specification without timing (Outcomes LR) scored **0.307811**. Both scores use the same 9,950 out-of-fold windows from 995 development participants. Lower log loss indicates a better probability forecast.

From the project root, recompute both scores directly from the saved predictions:

```text
python -B -c "import pandas as pd; from sklearn.metrics import log_loss; x = pd.read_csv('Results/fit/oof_predictions.csv', float_precision='round_trip'); print(len(x), x.participant_id.nunique(), format(log_loss(x.label, x.p_outcomes), '.6f'), format(log_loss(x.label, x.p_combined), '.6f'))"
```

The output gives the window count, participant count, Outcomes LR log loss, and Combined LR log loss, in that order.

```text
9950 995 0.307811 0.304950
```

To cross-check [Table 1](Results/figures/table_01_performance.png), open [metrics.csv](Results/tables/metrics.csv), select `scope = all`, and read `log_loss` for `Outcomes LR` and `Combined LR`. Subtracting Outcomes LR from Combined LR gives **−0.002861**. The corresponding row in [paired_differences.csv](Results/tables/paired_differences.csv) gives a 95% participant-bootstrap interval of **[−0.005526, −0.000247]**.

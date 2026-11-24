# Data provenance and processing

Much, Mutak, Pohl and Ranger released *Modeling speed-ability trade-off and test-taking persistence: Parameter validation for two psychometric models* on 28 June 2024. They collected responses, durations and action sequences from 1,244 adults recruited through Prolific. Their 2025 data paper describes the protocol and lists the release files in Table 7. Group 04 uses the matrix-reasoning tasks to predict an observable score on the next three items.

## Official sources and member confirmation

| Record | Published reference supplied with the submission | Member-confirmed official URL (leave blank until checked) |
| --- | --- | --- |
| Official Data Source / Repository Link | [OSF project overview](https://osf.io/9j6hm/overview) and [dataset DOI](https://doi.org/10.17605/OSF.IO/9J6HM) | **[Enter verified repository URL here]** |
| Data Paper / Reference Link | [Much et al. (2025), Journal of Open Psychology Data, 13, Article 1](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.124) and [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12270193/) | **[Enter verified paper URL here]** |
| Supporting preregistration | [Original study registration](https://osf.io/jm7ne/) | **[Enter verified registration URL here]** |

**License & Redistribution Status.** The authors state in sections 3.6 and 3.7 of the data paper that the dataset is licensed under CC BY 4.0, contains no identifying information, and uses study-generated numeric identifiers in place of Prolific IDs. The public release files listed below may be redistributed with attribution. Group 04 keeps those files byte-for-byte and identifies its own transformations separately. [Data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12270193/) · [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)

For the local provenance trail, the sibling archive is named `MLA_project_pre_restructure_20261121`. Within that archive, the originals are in `data/raw/1_main_dt/` and `data/raw/2_supplement_dt/`. The reading copies are `docs/references/dataset-paper.pdf` and `docs/references/Preregistration_Psychometric-Model-Validation.pdf`. The registration describes the source study. Neither reading copy is required to run Group 04's analysis.

## Original files included directly in Data

| File | Contents | Role |
| --- | --- | --- |
| [tte_data.csv](tte_data.csv) | One row per participant, 1,244 rows and 164 published fields, including demographic descriptors | Source of ID, instruction group and the 20 main-item X, Y and T fields |
| [mr_logdata.csv](mr_logdata.csv) | Ordered item-level actions and cumulative times | Source for main-item response reconstruction and action features |
| [mr_itemsolutions.csv](mr_itemsolutions.csv) | Published scoring keys | Checks reconstructed correctness |
| [tte_codebook.csv](tte_codebook.csv) and [tte_codebook.pdf](tte_codebook.pdf) | Machine-readable and human-readable main variable dictionary | Documents variable definitions and units |
| [tte_suppl_codebook.pdf](tte_suppl_codebook.pdf) | Supplemental file and field definitions | Documents the action log and scoring keys |

The six originals match the SHA-256 values in the archived OSF file metadata. The release manifest gives each original filename, byte count, checksum and public download URL. The separate source manifest identifies the three filtered inputs used by Stage 02. [Original file manifest](release_checksums.csv) · [Prepared file manifest](source_checksums.csv)

## Data processing selection

The original main file includes the demographic descriptors released by the authors. Group 04's analysis projection has 62 fields: the public ID, instruction group and 60 item variables. A distribution check covered all 27 CSV files, this README and the extracted text of both codebooks. All ID values in the eight CSVs with participant identifiers belonged to the public release. The scan found no matches for the credential or direct-contact patterns tested.

Run the following commands from the project root to rebuild the analysis inputs and windows from the bundled release. Both commands work offline.

```text
python -B Code/01_prepare_data.py --release Data
python -B Code/02_build_features.py
```

1. **Select fields and items.** Stage 01 verifies the three computational input files against their pinned release hashes. It retains the public ID, randomized group, selections (X), correctness (Y) and duration (T) for items MR01 through MR20. It selects main-item actions and keys. Demographics, practice items and post-study disruptions are excluded from predictors. Codebooks document the fields. Stage 01 writes `Data/source/main_responses.csv`, `action_log.csv` and `scoring_keys.csv` and verifies their hashes. Running Stage 01 without arguments checks the existing projections. The `--download` option obtains the three computational inputs from OSF.
2. **Audit measurements and missingness.** Stage 02 calls `audit_items` in `Code/response_data.py`. For each person-item pair it replays toggles in `elementOrder`, checks the reconstructed selection against the scoring key and Y, and compares T with the last submit's `cumulTime`. A blank score contributes zero only for a verified omission; its flag is retained. Repeated submits use the final submit time. Positive finite durations are retained, including long durations, with no trimming or winsorization. Invalid durations, nonmonotone action times, duplicate person-item records and scoring mismatches raise errors. Across 24,880 records there were 143 verified omissions and 41 pairs with repeated submits. All 232 participants with zero correct main responses remained eligible.
3. **Enforce participant-disjoint evaluation.** Stage 02 verifies the hashes and membership of `Results/splits/participant_split.csv` and `outer_folds.csv`. It audits the full public release, then restricts feature construction to 995 development participants. All windows from one person stay together. Each of five outer folds holds out 199 people and trains on the other 796. The historical 249-person test cohort is excluded from the final model fits, tuning and evaluation.
4. **Construct predictors and labels.** After items 3 through 7 in each block, `make_window` uses only completed items in that block. It calculates recent and prefix correctness, log(1 + seconds) timing after converting milliseconds to seconds, and recent toggle, deselection and first-action summaries. The label is one if fewer than two of the next three items are correct. Future correctness supplies the label only. The two blocks yield ten windows per development participant. Relative timing uses completed person history; Stage 03 estimates item-specific reference medians using training participants within each fold.
5. **Trace outputs.** Stage 02 writes `Results/features/completed_items.csv`, `development_windows.csv`, `development_folds.csv`, `feature_checks.csv` and `data_audit.json`. The audited result is 9,950 development windows and 6,587 events. Stage 03 performs three inner participant folds for tuning within each outer training set and records those assignments in `Results/fit/inner_folds.csv`. The corresponding stage logs are in `Results/logs/`.

## Historical and regression references

The two development-window CSVs in `reference/` preserve the exported numeric precision of the November reconstruction. Stage 02 rebuilds all windows from the released data and checks the saved columns before reusing their values. `reference/historical_predictions.csv` contains selected columns from the original October Week 10 export. `reference/historical_week10.csv` contains the corresponding reference losses and the hash of the full October export. Stage 04 compares 14 newly generated tables with `benchmarks/` after evaluation. The historical probabilities and benchmark results are excluded from training.

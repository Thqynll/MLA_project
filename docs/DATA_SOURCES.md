# Data sources and file roles

The dataset is attributed to Much, Mutak, Pohl, and Ranger. The data paper reports a CC BY 4.0 data license. Preserve attribution when redistributing the data.

- Much, S., Mutak, A., Pohl, S., & Ranger, J. (2025). Data From a Validation Study of Two Psychometric Models on Test-taking Behavior. Journal of Open Psychology Data, 13, Article 1. https://doi.org/10.5334/jopd.124
- Dataset: Modeling speed-ability trade-off and test-taking persistence – Parameter validation for two psychometric models. https://osf.io/9j6hm/ ; https://doi.org/10.17605/OSF.IO/9J6HM
- Supporting preregistration: https://osf.io/jm7ne/ . Local reading copy: `docs/references/Preregistration_Psychometric-Model-Validation.pdf`.

These links and bibliographic details are taken from the attached source materials. This package does not claim to have verified the unrelated references [3]–[7] in the old report.

| File | Role in this project |
| --- | --- |
| `data/raw/1_main_dt/tte_data.csv` | Main correctness, response times, participant IDs, and instruction groups; supplies labels and completed-history predictors. |
| `data/raw/2_supplement_dt/mr_logdata.csv` | Supplies action histories for process predictors and response reconstruction. |
| `data/raw/2_supplement_dt/mr_itemsolutions.csv` | Verifies scoring; answer-key strings are not model predictors. |
| Main and supplemental codebooks | Define variables, units, and measurement interpretation; not training samples. |
| Item overview PDFs | Instrument references; not training samples. |
| `tte_data.Rds` | Included original alternative format; current scripts use CSV. |
| `ct_pagetimes.csv`, `tte_disruptions.csv` | Included supplemental sources; unused as predictive features in this analysis. |
| Data paper and preregistration PDF | Research and design references; not directly trained on. |

Practice, concentration, and persistence tasks are excluded from the prediction target. End-of-study disruption reports are unavailable at the prediction boundary and are not predictors. The source paper supplies experimental instruction groups; the ML train/validation/test split is the project's reconstruction.

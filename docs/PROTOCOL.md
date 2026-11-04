# Week 10 protocol

## Target and information boundary

After completed item j in {3,4,5,6,7}, predict whether the next three items contain fewer than two correct responses. History resets at each block boundary. Verified omissions count as zero. The event is a future-score outcome, not a psychological diagnosis. No feature uses future correctness, future durations, final scores, or end-of-study disruption reports.

## Features

| Family | Definition |
| --- | --- |
| Context | One-hot block × anchor × instruction. |
| Outcomes | Last correctness, mean last-three correctness, mean completed-prefix correctness. |
| Raw timing | log(1+seconds), summarized as latest, recent-three mean, and completed-prefix mean. |
| Item-relative timing | Subtract each item's median log time among current training participants, then take the same three summaries. |
| Person-relative timing | Latest minus completed-prefix mean, and recent-three mean minus completed-prefix mean. |
| Process | Recent-three mean log(1+toggle count), deselection fraction, and log(1+first-action seconds). |

Toggle count includes selections and deselections, not submit. Alternating even-numbered toggles deselect an element. Deselection fraction divides by max(toggle count,1). First-action time includes submit for submit-only omissions. Source milliseconds are converted to seconds before log transforms. Repeated submits are retained and audited against the final submit duration.

Relative LR replaces raw-time summaries with five relative-time summaries. Relative Process LR adds the three action summaries to Relative LR. Process LR adds the three action summaries to Combined LR.

## Training and priors

Keep the 249 test IDs frozen. Five outer folds each contain 199 development participants, with 796 available for training. All windows from a person stay together. Three inner participant folds select C=0.1 or 1.0 by pooled inner validation log loss, weighted by window counts. Every inner and outer training fold independently learns item medians, the context encoder, and the scaler. The instruction-order group supplies stratification; target labels do not determine participant assignments.

The full encoded design matrix is standardized in this Week 10 reconstruction, matching the older embedded Week 10 script. The earlier rebuilt Week 9 starters standardize numeric features only. Their validation metrics and Week 10 OOF metrics describe different protocols and should not be treated as the same experiment.

Global prior uses training event prevalence. Window and history priors use (events+1)/(windows+2) per context or recent-correct count. Item prior uses (errors+1)/(responses+2) for each item × instruction pair. For the three known upcoming items with error probabilities a,b,c, the independent majority-error prior is ab+ac+bc-2abc. Those probabilities are learned from training participants only.

## Evaluation and interpretation

Log loss is primary; Brier and ROC-AUC are secondary. Report all four recent-correct groups. Pooled OOF AUC for the global prior may differ from 0.5 because training prevalence varies across folds; its within-fold AUC is 0.5.

Use 2,000 paired participant-cluster bootstrap samples with seed 2026. Sample from all 995 development IDs and keep all selected-person windows together. Within a stratum, divide the resampled sum of loss differences by the resampled number of eligible windows. Intervals condition on fitted OOF predictions and do not include refitting or selection uncertainty. They are exploratory and not adjusted for multiple comparisons.

The reporting focus is Combined-minus-Outcomes log loss in pooled groups 1/3 and 2/3. It was chosen for this reconstruction using existing feedback, not preregistered before the original analysis. The remaining comparisons are exploratory too. This cohort has already informed earlier analysis choices; nested CV does not turn it into a new confirmatory dataset.

Review budgets rank probabilities separately in each outer fold and select floor(budget × fold windows). Ties break by public ID, block, anchor. Five percent therefore selects 495 windows; ten and twenty percent select 995 and 1,990. These are retrospective workload illustrations.

## Case selection fixed before this rerun

Within recent_correct=1, compute d=(2*y-1)*(p_Combined-p_Outcomes). Select the five largest positive d values and five most negative d values. Break ties by ID, block, anchor. Repeated participants are allowed. Each case records three observed scores and durations, OOF probabilities, label, fold, and an objective probability-change description. Human interpretation remains the responsibility of the group.

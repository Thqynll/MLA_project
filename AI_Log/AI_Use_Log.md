# AI use and verification

Group 04 used ChatGPT/Codex to:

- Clarify how the Week 9 experiments led to the final research question.
- Suggest ways to address the lecturer's feedback, such as comparing models on the same participants.
- Organize ideas from the research papers and identify which findings could be compared with the group's results.
- Refine the wording of those comparisons and the conclusions about what the results show.

## Verification

During the November reconstruction, Codex also helped adapt the evaluation code, run comparisons, generate tables, and draft the report. The table below lists the automated checks and the reviews recorded by group members.

| Check | Evidence | Checked |
| --- | --- | --- |
| Prediction boundary and training-fold preprocessing | Code/tests/ | Yes, by script |
| Agreement with saved numerical results | Data/benchmarks/ and Results/logs/04_evaluate_predictions.json | Yes, by script |
| Replay of held-out probabilities and historical labels | Code/verify_results.py | Yes, by script |
| Paper or code check by each member | [Two code checks per member](#code-checks) | Results checked by Codex. Member checks pending. |
| Review of ten cases | [Case review register](../Results/member_case_review.csv) | Yes, by group members |

## Member register

### Code checks

Codex checked these six results against the saved November data on 2026-11-24. Each member can enter their own date after following the check.

| Assigned member | Claim | How to check it | Result | Date checked by member |
| --- | --- | --- | --- | --- |
| Le Hoang The Quyen | A low score means zero or one of the next three answers was right. | Read `make_window` in [response_data.py](../Code/response_data.py). Add the three `future_correct` values in the [window data](../Results/features/development_windows.csv) and compare with `label`. | The rule matched all 9,950 labels. | Pending |
| Le Hoang The Quyen | Each person has ten prediction windows. | Find the loop over items 3 to 7 in [response_data.py](../Code/response_data.py). Count those items for each person and block in the window data. | All 995 people had five windows in each of two blocks. | Pending |
| Tran Gia Thanh | Each person stays in one test fold during cross-validation. | Read `run_nested_cv` in [score_models.py](../Code/score_models.py). In the [saved predictions](../Results/fit/oof_predictions.csv), group by person and count the different values of `outer_fold`. | Every person had one fold. Each of the five folds had 199 people. | Pending |
| Tran Gia Thanh | Adding time lowered the average log loss. | Use the [README command](../README.md#reproduce-one-main-result) to calculate log loss from the saved predictions. Compare both numbers with [Table 1](../Results/figures/table_01_performance.png). | Scores alone with test context gave 0.307811. Adding time gave 0.304950. | Pending |
| Nguyen Phu Trong | After three wrong answers, recovery was rare. | Follow the group filter in [history_tables.py](../Code/history_tables.py). Keep rows with `recent_correct = 0` in the saved predictions and count each label. | Of 4,165 windows, 4,127 had a low score next. Only 38 had two or three right. | Pending |
| Nguyen Phu Trong | The report picked cases where timing helped and cases where it hurt. | Read `select_cases` in [history_tables.py](../Code/history_tables.py). In the [ten selected cases](../Results/tables/case_review_10.csv), compare both probabilities with the actual label. | All ten had one of the last three answers right. Timing moved the forecast closer to the label in five cases and farther away in five. | Pending |

### Case reviews

These are two checks from each member's [case notes](../Results/member_case_review.csv). “Low score” means zero or one right answer on the next three items.

| Member | Date checked | Case (ID, block, item) | Claim | What the member checked |
| --- | --- | --- | --- | --- |
| Le Hoang The Quyen | 2026-11-23 | 2096, 1, 7 | The person did better on the next three items. | Only one of the last three answers was right. All three of the next answers were right. |
| Le Hoang The Quyen | 2026-11-23 | 2787, 1, 6 | One item took about 38 minutes. | The time record shows a long wait. It does not say why the item took so long. |
| Tran Gia Thanh | 2026-11-23 | 2112, 1, 7 | Adding time helped in this case. | The chance of a low score fell from 63.8% to 48.9%. The person got two of the next three right. |
| Tran Gia Thanh | 2026-11-23 | 1617, 1, 5 | The model with scores alone did better here. | It gave a low score a 75.5% chance. Adding time cut that to 54.3%, and all three answers were wrong. |
| Nguyen Phu Trong | 2026-11-23 | 1675, 2, 6 | Adding time made this forecast worse. | The chance of a low score rose from 51.4% to 69.8%, but the person got two answers right. |
| Nguyen Phu Trong | 2026-11-23 | 3122, 2, 6 and 7 | These two forecasts share two answers. | The first uses items 7, 8, and 9 to check the result. The next uses items 8, 9, and 10. Both use items 8 and 9. |

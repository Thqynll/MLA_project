"""Build the report from saved tables, historical probabilities and member evidence."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import DATA, FEATURES, FIT, ROOT, TABLES, read_table


def report_pages():
    m = read_table(TABLES / "metrics.csv").set_index(["scope", "model"])
    d = read_table(TABLES / "paired_differences.csv").set_index(["scope", "first", "second"])
    old = read_table(DATA / "reference/historical_predictions.csv")
    now = read_table(FIT / "oof_predictions.csv")
    cases = read_table(TABLES / "case_review_10.csv")
    reviews = read_table(ROOT / "Report/member_case_review.csv").fillna("")
    keys = ["participant_id", "block", "anchor"]
    if set(map(tuple, cases[keys].values)) != set(map(tuple, reviews[keys].values)):
        raise ValueError("The member review register must identify the ten selected cases.")
    cases = cases.merge(reviews, on=keys, validate="one_to_one")
    fields = ["reviewer", "review_date", "observed_evidence", "interpretation", "decision"]
    checked = cases[fields].apply(lambda c: c.astype(str).str.strip().ne("")).all(axis=1)
    anchor = now.query("participant_id == 2096 and block == 1 and anchor == 7").squeeze()
    historical = []
    models = ["History prior", "Outcomes LR", "Combined LR", "Process LR", "Combined CART", "Process CART"]
    for scope, group in [("all", old), ("0/3", old.query("recent_correct_count == 0")),
                         ("1/3", old.query("recent_correct_count == 1"))]:
        for name in ["Outcomes LR", "Combined LR"]:
            historical.append([scope, name, f"{log_loss(group.target, group[name]):.6f}",
                               f"{m.loc[(scope, name), 'log_loss']:.6f}"])
    def contrast(scope, first, second):
        row = d.loc[scope, first, second]
        return [scope, first, second, f"{row.log_loss_difference:+.6f}",
                f"[{row.log_loss_ci_low:+.6f}, {row.log_loss_ci_high:+.6f}]"]
    overall = [[name] + [f"{m.loc[('all', name), k]:.6f}" for k in ("log_loss", "brier", "auc")]
               for name in models]
    by_history = []
    for scope in ("0/3", "1/3", "2/3", "3/3"):
        row = m.loc[scope, "Outcomes LR"]
        by_history.append([scope, str(int(row.participants)), str(int(row.windows)), str(int(row.events)),
                           str(int(row.non_events)), f"{row.log_loss:.6f}", f"{m.loc[(scope, 'Combined LR'), 'log_loss']:.6f}"])
    detail = []
    for row in cases.itertuples():
        scores = "".join(str(int(getattr(row, f"history_correct_{i}"))) for i in (1, 2, 3))
        times = ", ".join(f"{getattr(row, f'history_seconds_{i}'):.1f}" for i in (1, 2, 3))
        detail.append([str(row.participant_id), f"{row.block}/{row.anchor}", scores, times,
                       f"{row.p_outcomes:.3f}", f"{row.p_combined:.3f}", str(row.label)])
    budget = read_table(TABLES / "table10_review_budgets.csv").query("model == 'Combined LR'")
    budgets = [[f"{v.budget:.0%}", str(v.selected_windows), str(v.events_selected),
                f"{v.precision:.4f}", f"{v.recall:.4f}"] for v in budget.itertuples()]
    mixed = contrast("1/3+2/3", "Combined LR", "Outcomes LR")
    relative = read_table(TABLES / "table08_relative_timing_differences.csv").query("scope == 'all'").squeeze()
    pages = [
        ("Abstract and research question", [
            ("h", "Response Process Analysis for Assessment Review"),
            ("p", "Group 04 | Machine Learning Applications | Le Hoang The Quyen, Tran Gia Thanh and Nguyen Phu Trong"),
            ("h", "Abstract"),
            ("p", f"We studied whether response times and selection actions add predictive information after recent correctness is observed. Five outer participant folds covered 995 adults and 9,950 development windows. Adding time to logistic regression changed pooled log loss by {d.loc[('all','Combined LR','Outcomes LR'),'log_loss_difference']:+.6f}. The gain was small relative to the contribution of score history. Action features showed an uncertain additional gain on log loss. These estimates describe an exploratory reanalysis of one assessment protocol."),
            ("h", "Problem and intended use"),
            ("p", "An assessment designer deciding whether to retain or review response logs needs to know when those logs improve a forecast based on scores. We estimate the probability of fewer than two correct answers on the next three items. The evidence applies to adults completing this fixed matrix-reasoning protocol. Classroom decisions, psychological diagnoses and intervention effects require separate evidence."),
            ("h", "Relation to the source study"),
            ("p", "Much et al. [1] released response, duration and action data collected for validation of psychometric models of speed and persistence. Group 04 defines a prospective information boundary within each block and compares predictive probabilities across feature sets and model families. The original preregistration concerns the source study."),]),
        ("Data and methods", [
            ("h", "Source, permission and input selection"),
            ("p", "Much et al. [1, 2] released data for 1,244 adults under CC BY 4.0. The authors replaced Prolific IDs with study-generated numeric IDs [1, sections 3.6-3.7]. With --release Data, Code/01_prepare_data.py verifies the three original CSVs and writes the MR01-MR20 response, action and key fields to Data/source/. The main and supplemental codebooks define these fields."),
            ("h", "Measurement and participant boundaries"),
            ("p", "Code/02_build_features.py replays ordered toggles, checks X_MR selections and Y_MR scores against the key, and matches T_MR milliseconds to final-submit cumulTime, including 41 repeated-submit pairs. Of 24,880 records, 143 blank scores were verified as omissions and counted as zero. Invalid times fail the audit. Valid long durations and all 232 zero-score participants are retained."),
            ("p", "Stage 02 validates Results/splits/, sets aside 249 test participants and writes 9,950 windows for 995 people to Results/features/development_windows.csv. Each person's blocks stay in one outer fold. Stage 03 tunes on three inner participant folds recorded in Results/fit/inner_folds.csv."),
            ("h", "Prediction time"),
            ("p", "After item j = 3 through 7, the event is fewer than two correct among the next three items. Only completed items in the current block supply responses, times and actions. Future correctness supplies the label. Final scores and post-study disruptions are excluded."),
            ("h", "One development prediction"),
            ("p", f"ID 2096, block 1, anchor 7 had recent correctness 0, 1, 0 and durations 20.528, 30.187 and 95.493 seconds. The history prior predicted {anchor.p_history_prior:.4f}, Outcomes LR {anchor.p_outcomes:.4f} and Combined LR {anchor.p_combined:.4f}. The next three items were all correct, giving label zero. This is a held-out fold prediction."),
            ("h", "Features and baselines"),
            ("p", "The history prior estimates a smoothed event rate per recent correct count. Outcomes LR distinguishes the most recent score, recent-three accuracy, completed-prefix accuracy and known block/anchor/instruction context. Combined LR adds latest, recent and prefix log(1 + seconds). Relative timing centers log time by training-item medians and completed person history. Process LR adds recent toggle counts, deselection fractions and first-action latencies.")]),
        ("Experimental setup and overall results", [
            ("p", "All observations for one person stay together. Of 1,244 participants, 249 belonged to a previously examined historical test cohort. The reconstruction excludes them from model fitting, tuning and evaluation. Five outer folds hold out 199 people each. Three inner folds select by pooled log loss. Encoders, scalers and item medians are fitted on each training fold. All LR variants share one model family."),
            ("p", "LR tunes C over 0.1 and 1.0. The reconstructed LR standardizes the full encoded matrix. CART uses Gini impurity, depths 3, 5 or 8, and minimum leaf sizes 20 or 50. CART uses the same inner assignments and feature sets. Model and fold seeds are 42. The score-only versus score-plus-time comparison changes one feature set while preserving evaluation units."),
            ("t", "Table 1. Development out-of-fold metrics", ["Model", "Log loss", "Brier", "AUC"], overall),
            ("p", "Lower log loss and Brier score indicate better probability estimates. AUC measures ranking. Compare the two LR rows to isolate the timing increment. Combined CART has higher log loss than Combined LR for the declared tuning grid."),
            ("t", "Table 2. Paired log-loss differences, first model minus second",
             ["Scope", "First", "Second", "Difference", "95% interval"],
             [contrast("all", a, b) for a,b in [("Outcomes LR","History prior"),("Combined LR","Outcomes LR"),
                                                  ("Process LR","Combined LR"),("Combined CART","Combined LR")]]),
            ("p", "Intervals use 2,000 paired participant-bootstrap draws, seed 2026. They condition on fitted predictions and omit refitting uncertainty. Comparisons use exploratory, unadjusted intervals.")]),
        ("Timing across recent score histories", [
            ("t", "Table 3. Counts and log loss by recent correctness", ["Recent", "People", "Windows", "Events", "Others", "Scores", "+ Time"], by_history),
            ("fig", "Figure 1. Timing increment in log loss with paired participant intervals."),
            ("p", f"At 0/3 there are 4,127 events and 38 non-events. Adding time worsens the point estimate, but its interval includes zero. AUC alone describes ranking against a small exceptional class; log loss penalizes confident errors. A common timing coefficient could contribute to the pattern. This mechanism has not been established. For pooled 1/3 and 2/3, the difference is {mixed[3]} {mixed[4]}. This focus followed feedback and was exploratory."),
            ("p", f"Replacing raw timing with relative timing gives an overall difference of {relative.log_loss_difference:+.6f} [{relative.log_loss_ci_low:+.6f}, {relative.log_loss_ci_high:+.6f}]. The evidence does not establish a general incremental advantage for centering or action summaries. People can appear in multiple history groups.")]),
        ("Error analysis and review workload", [
            ("p", "Within the 1/3 history group, rank the signed change (2y - 1) times (p_combined - p_outcomes). Take the five largest positive changes and the five most negative. Resolve ties by public ID, block and anchor. The first five rows move toward the observed label. Probabilities concern the future-score event."),
            ("t", "Table 4. Selected development cases", ["ID", "Block/j", "Scores", "Recent seconds", "Scores p", "+ Time p", "Label"], detail),
            ("p", f"These are selected extremes, so they cannot estimate typical benefit. Case-specific observations require inspection of the actual responses and durations. Completed member records: {checked.sum()} of 10. The separate member register preserves evidence and decisions across reruns."),
            ("t", "Table 5. Within-fold review capacity for Combined LR", ["Budget", "Selected", "Events", "Precision", "Recall"], budgets),
            ("p", "Each fold contributes the floor of its budget times its number of windows. Ties use ID, block and anchor. High precision occurs alongside many missed events; this retrospective ranking has no validated deployment threshold.")]),
        ("Historical comparison, discussion and conclusion", [
            ("t", "Table 6. Original Week 10 and current reconstruction", ["History", "Model", "Original log loss", "Current log loss"], historical),
            ("p", "Both runs use identical raw source files and share the 249 test IDs. They assign only 212 of 995 development participants the same numbered outer fold. The original used ID-sorted outer seed 20261021, inner seed 20261030 plus fold, numeric-only standardization and bootstrap seed 20261024. The reconstruction uses the saved seed-42 assignments and scales the encoded design. Changing bootstrap seeds affects intervals for fixed predictions. The specific contribution of each training difference has not been isolated."),
            ("p", "The original 0/3 timing interval excluded zero. The reconstructed interval includes it. Each conclusion must follow its own run. The historical forecasts in Table 6 are recovered from the original saved probabilities. The LR-CART comparison and the ten selected cases use current probabilities."),
            ("h", "Discussion and limitations"),
            ("p", "The source study frames timing in an experimental assessment setting [1]. Liang [3] examined timing and accuracy in gamified university quizzes. That setting differs in task, participants and feedback. Here, overlapping windows, an English-speaking adult volunteer sample and previously examined development data limit generalization. Action logs measure recorded behavior; they do not establish motive."),
            ("h", "Conclusion"),
            ("p", "Score history accounts for most of the observed predictive gain. Timing contributes a smaller improvement, particularly in mixed score histories. The additional value of actions and the applicability of these probabilities to other assessments remain uncertain.")]),
        ("References and reproducibility", [
            ("p", "[1] Much, S., Mutak, A., Pohl, S., and Ranger, J. (2025). Data From a Validation Study of Two Psychometric Models on Test-taking Behavior. Journal of Open Psychology Data, 13, 1. DOI 10.5334/jopd.124."),
            ("p", "[2] Much, S., Mutak, A., Pohl, S., and Ranger, J. (2024). Modeling speed-ability trade-off and test-taking persistence: Parameter validation for two psychometric models. Open Science Framework. DOI 10.17605/OSF.IO/9J6HM."),
            ("p", "[3] Liang, Z. (2024). More haste, less speed? Relationship between response time and response accuracy in gamified online quizzes in an undergraduate engineering course. Frontiers in Education, 9, 1412954. DOI 10.3389/feduc.2024.1412954."),
            ("h", "Reproduction and use of AI"),
            ("p", "Run the five numbered scripts in Code in order. Results/table_sources.csv identifies the generators for the analysis tables. Report/report_sources.csv identifies each displayed table and figure. Data/README.md documents the public release, transformations and license. AI_Log/AI_Use_Log.md records tool assistance and the outstanding individual source or code checks.")])]
    return pages

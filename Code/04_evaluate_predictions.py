"""Calculate paired uncertainty, review budgets and disagreement cases from held-out predictions."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from threadpoolctl import threadpool_limits

from config import BOOT_SEED, DATA, DRAWS, FEATURES, FIT, RESULTS, SPLITS, TABLES, read_table, record, save_table
from comparison_tables import metrics, paired
from history_tables import create_tables


def main():
    predictions = FIT / "oof_predictions.csv"
    folds = FEATURES / "development_folds.csv"
    oof, people = read_table(predictions), read_table(folds)
    test = read_table(SPLITS / "participant_split.csv").query("split == 'test'")
    if (len(oof) != 9950 or set(oof.participant_id) != set(people.ID)
            or set(oof.participant_id) & set(test.ID)
            or oof.duplicated(["participant_id", "block", "anchor"]).any()
            or not oof.outer_fold.eq(oof.participant_id.map(people.set_index("ID").outer_fold)).all()):
        raise ValueError("Invalid out-of-fold cohort or window assignment.")
    with threadpool_limits(limits=1):
        tables = create_tables(oof, people, DRAWS)
        tables["metrics.csv"] = metrics(oof)
        tables["paired_differences.csv"] = paired(oof, people.ID.tolist())
    sources = []
    date = datetime.now(timezone.utc).isoformat()
    for name, frame in tables.items():
        reference = DATA / "benchmarks" / name
        expected = read_table(reference)
        pd.testing.assert_frame_equal(frame.reset_index(drop=True).replace("", np.nan), expected,
                                      check_dtype=False, rtol=1e-10, atol=1e-12)
        save_table(frame, TABLES / name)
        if name == "metrics.csv":
            function = "comparison_tables.metrics"
        elif name == "paired_differences.csv":
            function = "comparison_tables.paired"
        elif "differences" in name:
            function = "history_tables.paired_differences"
        elif name == "case_review_10.csv":
            function = "history_tables.select_cases"
        elif name == "table10_review_budgets.csv":
            function = "history_tables.review_budgets"
        else:
            function = "history_tables.metric_table"
        sources.append({"output": "Results/tables/" + name, "function": function, "run_utc": date,
                        "source_file": "Code/" + function.split(".")[0] + ".py",
                        "entrypoint": "Code/04_evaluate_predictions.py", "input": "Results/fit/oof_predictions.csv"})
    save_table(pd.DataFrame(sources), RESULTS / "table_sources.csv")
    record("04_evaluate_predictions", [predictions, folds, *sorted((DATA / "benchmarks").glob("*.csv"))],
           [TABLES / name for name in tables] + [RESULTS / "table_sources.csv"],
           paired_participant_draws=DRAWS, bootstrap_seed=BOOT_SEED, matched_benchmark_tables=len(tables))


if __name__ == "__main__":
    main()

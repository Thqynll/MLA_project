"""Check local stage records, replay fitted models and validate the historical comparison."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from config import DATA, FEATURES, FIT, ROOT, RESULTS, TABLES, digest, read_table
from score_models import add_relative, item_medians
from tree_models import predict


def verify_records():
    for stage in ("01_prepare_data", "02_build_features", "03_train_models",
                  "04_evaluate_predictions", "05_export_results"):
        entry = json.loads((RESULTS / "logs" / (stage + ".json")).read_text(encoding="utf-8"))
        for group in ("inputs", "outputs"):
            for name, expected in entry[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Checksum mismatch in {stage}: {name}")


def replay():
    people = read_table(FEATURES / "development_folds.csv")
    items = read_table(FEATURES / "completed_items.csv")
    oof = read_table(FIT / "oof_predictions.csv")
    inner = read_table(FIT / "inner_folds.csv")
    count = 0
    for fold in range(1, 6):
        held_out = set(people.loc[people.outer_fold == fold, "ID"])
        training = set(people.ID) - held_out
        assignments = inner.loc[inner.outer_fold == fold]
        if set(assignments.ID) != training or assignments.ID.duplicated().any() or set(assignments.inner_fold) != {1,2,3}:
            raise ValueError("Invalid inner participant assignment.")
        frame = oof.loc[oof.outer_fold == fold]
        medians = item_medians(items, training)
        relative = add_relative(frame, medians)
        paths = sorted((FIT / "models").glob(f"fold_{fold}_*.joblib"))
        if len(paths) != 8:
            raise ValueError("Expected eight fitted models for each outer fold.")
        for path in paths:
            bundle = joblib.load(path)
            if set(bundle["validation_ids"]) != held_out or set(bundle["training_ids"]) != training:
                raise ValueError(f"Unexpected participant assignment in {path.name}.")
            slug = path.stem.removeprefix(f"fold_{fold}_")
            columns = bundle["columns"]
            if slug.endswith("_cart"):
                probability = predict(bundle["model"], frame, columns)
            else:
                pd.testing.assert_series_equal(medians, bundle["item_medians"], check_exact=False)
                probability = bundle["model"].predict_proba(relative[["context"] + columns])[:,1]
            np.testing.assert_allclose(probability, frame["p_"+slug], rtol=1e-10, atol=1e-12)
            count += 1
    return count


def historical_check():
    old = read_table(DATA / "reference/historical_predictions.csv")
    new = read_table(FIT / "oof_predictions.csv")
    joined = old.merge(new, left_on=["ID","block","anchor"], right_on=["participant_id","block","anchor"],
                       suffixes=("_original","_current"), validate="one_to_one")
    if len(joined) != 9950 or not joined.target.eq(joined.label).all():
        raise ValueError("Historical labels or windows differ.")
    same_folds = joined.loc[joined.outer_fold_original == joined.outer_fold_current, "ID"].nunique()
    if same_folds != 212:
        raise ValueError("Unexpected historical outer-fold agreement.")
    for row in read_table(DATA / "reference/historical_week10.csv").itertuples():
        group = old if row.scope == "all" else old.loc[old.recent_correct_count == int(row.scope[0])]
        np.testing.assert_allclose(log_loss(group.target, group[row.model]), row.original_log_loss, atol=1e-12)
    actual = read_table(TABLES / "metrics.csv").query("scope == 'all' and model == 'Combined LR'").squeeze()
    np.testing.assert_allclose(actual.log_loss, 0.304949831, atol=1e-8, rtol=0)


if __name__ == "__main__":
    verify_records()
    with threadpool_limits(limits=1):
        models = replay()
        historical_check()
    print(f"Verified all five stage records, {models} fitted models and 9,950 historical labels.")

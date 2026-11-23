"""Tune both model families on matched participant folds and save held-out probabilities."""
import platform
from importlib.metadata import version

from threadpoolctl import threadpool_limits

from config import FEATURES, FIT, LR_GRID, DEPTHS, LEAVES, MODEL_SEED, read_table, record, save_table
from score_models import run_nested_cv
from tree_models import run_cart


def main():
    paths = [FEATURES / name for name in ("development_windows.csv", "completed_items.csv", "development_folds.csv")]
    windows, items, people = map(read_table, paths)
    with threadpool_limits(limits=1):
        oof, lr, inner, medians = run_nested_cv(windows, items, people, FIT)
        oof, trees = run_cart(oof, people, inner, FIT)
    for frame, name in [(oof, "oof_predictions"), (lr, "lr_tuning"), (trees, "tree_tuning"),
                        (inner, "inner_folds"), (medians, "training_item_medians")]:
        save_table(frame, FIT / (name + ".csv"))
    outputs = sorted(FIT.glob("*.csv")) + sorted((FIT / "models").glob("*.joblib"))
    record("03_train_models", paths, outputs, python=platform.python_version(),
           versions={k: version(k) for k in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl")},
           seed=MODEL_SEED, lr_grid=LR_GRID, tree_depth=DEPTHS, tree_minimum_leaf=LEAVES,
           outer_folds=5, inner_folds=3, frozen_test_scored=False)


if __name__ == "__main__":
    main()

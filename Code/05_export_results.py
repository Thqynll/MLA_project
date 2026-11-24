"""Export PNG images of the six report tables and the timing figure."""
from shutil import rmtree
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

from config import DATA, RESULTS, TABLES, read_table, record

FIGURES = RESULTS / "figures"
INPUTS = [
    TABLES / "metrics.csv", TABLES / "paired_differences.csv",
    TABLES / "case_review_10.csv", TABLES / "table10_review_budgets.csv",
    DATA / "reference/historical_predictions.csv",
    TABLES / "table07_history_differences.csv",
]
MODELS = ("History prior", "Outcomes LR", "Combined LR", "Process LR",
          "Outcomes CART", "Combined CART", "Process CART")
PAIRS = (("Outcomes LR", "History prior"), ("Combined LR", "Outcomes LR"),
         ("Process LR", "Combined LR"), ("Combined CART", "Outcomes CART"),
         ("Combined CART", "Combined LR"))
HISTORIES = ("0/3", "1/3", "2/3", "3/3")


def effect(row):
    return [f"{row.log_loss_difference:+.6f}",
            f"[{row.log_loss_ci_low:+.6f}, {row.log_loss_ci_high:+.6f}]"]


def report_tables():
    metrics = read_table(TABLES / "metrics.csv").set_index(["scope", "model"]).set_flags(allows_duplicate_labels=False)
    paired = read_table(TABLES / "paired_differences.csv").set_index(
        ["scope", "first", "second"]).set_flags(allows_duplicate_labels=False)
    overall = []
    for name in MODELS:
        row = metrics.loc["all", name]
        overall.append([name, f"{row.log_loss:.6f}", f"{row.brier:.6f}", f"{row.auc:.6f}"])
    yield "table_01_performance", ["Model", "Log loss", "Brier score", "ROC AUC"], overall

    changes = [[first, second, *effect(paired.loc["all", first, second])] for first, second in PAIRS]
    yield "table_02_paired_changes", ["First model", "Second model", "Change", "95% interval"], changes

    histories = []
    for scope in HISTORIES:
        row = metrics.loc[scope, "Outcomes LR"]
        delta = effect(paired.loc[scope, "Combined LR", "Outcomes LR"])
        histories.append([scope, f"{int(row.windows):,}", f"{int(row.events):,}",
                          f"{int(row.non_events):,}", *delta])
    yield "table_03_score_history", ["Recent", "Windows", "Events", "Non-events", "Change", "95% interval"], histories

    cases = []
    for row in read_table(TABLES / "case_review_10.csv").itertuples():
        scores = "".join(str(int(getattr(row, f"history_correct_{i}"))) for i in (1, 2, 3))
        times = ", ".join(f"{getattr(row, f'history_seconds_{i}'):.1f}" for i in (1, 2, 3))
        cases.append([str(row.participant_id), f"{row.block}/{row.anchor}", scores, times,
                      f"{row.p_outcomes:.3f}", f"{row.p_combined:.3f}", str(row.label)])
    yield "table_04_selected_cases", ["Study ID", "Block/j", "Scores", "Recent seconds", "Scores p", "+ Time p", "Label"], cases

    budgets = []
    for row in read_table(TABLES / "table10_review_budgets.csv").query("model == 'Combined LR'").itertuples():
        budgets.append([f"{row.budget:.0%}", str(row.selected_windows), str(row.events_selected),
                        f"{row.precision:.4f}", f"{row.recall:.4f}"])
    yield "table_05_review_budget", ["Budget", "Selected", "Events", "Precision", "Recall"], budgets

    original = read_table(DATA / "reference/historical_predictions.csv")
    historical = []
    for scope in ("all", "0/3", "1/3"):
        group = original if scope == "all" else original.loc[original.recent_correct_count.eq(int(scope[0]))]
        for model in ("Outcomes LR", "Combined LR"):
            old_loss = log_loss(group.target, group[model], labels=[0, 1])
            historical.append([scope, model, f"{old_loss:.6f}", f"{metrics.loc[(scope, model), 'log_loss']:.6f}"])
    yield "table_06_historical_comparison", ["Recent history", "Model", "Original Week 10", "Reconstruction"], historical


def write_table(directory, stem, headers, rows):
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * (1 + len(rows)) + 0.12))
    ax.set_axis_off()
    widths = np.array([max(map(len, cells)) + 3 for cells in zip(headers, *rows)])
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths / widths.sum(),
                     cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (index, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#dce8eb" if index == 0 else ("#f3f5f6" if index % 2 else "white"))
        if index == 0:
            cell.set_text_props(weight="bold")
    fig.savefig(directory / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def timing_figure(directory):
    data = read_table(TABLES / "table07_history_differences.csv")
    data = data.query("first == 'Combined LR' and second == 'Outcomes LR'").set_index(
        "scope").set_flags(allows_duplicate_labels=False).loc[list(HISTORIES)]
    delta = data.log_loss_difference
    fig, ax = plt.subplots(figsize=(7.4, 2.65), layout="constrained")
    ax.axvline(0, color="#777777", linewidth=1)
    ax.errorbar(delta, data.index, xerr=np.vstack([delta - data.log_loss_ci_low, data.log_loss_ci_high - delta]),
                fmt="o", capsize=4, color="#204e64")
    ax.invert_yaxis()
    ax.set(xlabel="Log loss difference (Combined LR minus Outcomes LR)", ylabel="Recent correct")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "timing_by_history.png", dpi=320)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Use normal directory inheritance so Windows users can read the published PNGs.
    directory = FIGURES / f".export-{uuid4().hex}"
    directory.mkdir()
    try:
        for stem, headers, rows in report_tables():
            write_table(directory, stem, headers, rows)
        timing_figure(directory)
        outputs = []
        for asset in sorted(directory.iterdir()):
            outputs.append(asset.replace(FIGURES / asset.name))
    finally:
        rmtree(directory)
    record("05_export_results", INPUTS, outputs)


if __name__ == "__main__":
    main()

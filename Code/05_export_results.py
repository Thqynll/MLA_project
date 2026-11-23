"""Export the report, figure and provenance from the evaluated probabilities."""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from config import DATA, FIT, RESULTS, ROOT, TABLES, read_table, record, save_table
from report_content import report_pages

REPORT = ROOT / "Report"
FIGURES = RESULTS / "figures"


def timing_figure():
    frame = read_table(TABLES / "table07_history_differences.csv")
    frame = frame.query("first == 'Combined LR' and second == 'Outcomes LR'").sort_values("scope")
    x = frame.log_loss_difference
    fig, ax = plt.subplots(figsize=(7.4, 2.65), layout="constrained")
    ax.axvline(0, color="#777777", linewidth=1)
    ax.errorbar(x, frame.scope, xerr=np.vstack([x-frame.log_loss_ci_low, frame.log_loss_ci_high-x]),
                fmt="o", capsize=4, color="#204e64")
    ax.invert_yaxis()
    ax.set(xlabel="Log loss difference (Combined LR minus Outcomes LR)", ylabel="Recent correct")
    ax.spines[["top", "right"]].set_visible(False)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "timing_by_history.png", dpi=320)
    plt.close(fig)


def escape(value):
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def render(pages):
    latex = [r"\documentclass[10pt]{article}", r"\usepackage[margin=18mm]{geometry}",
             r"\usepackage{graphicx,booktabs}", r"\setlength{\parskip}{0.6em}", r"\begin{document}"]
    with PdfPages(REPORT / "Final_Report.pdf", metadata={"Title": "Response Process Analysis for Assessment Review",
                  "Author": "Group 04", "CreationDate": None, "ModDate": None}) as pdf:
        for index, (title, blocks) in enumerate(pages, 1):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = .94
            fig.text(.075, y, title, fontsize=17, weight="bold", color="#184555", va="top")
            y -= .066
            latex.append(r"\section*{" + escape(title) + "}")
            for kind, *values in blocks:
                caption = values[0]
                if kind == "h":
                    fig.text(.075, y, caption, fontsize=12.2, weight="bold", va="top")
                    y -= .034
                    latex.append(r"\subsection*{" + escape(caption) + "}")
                elif kind == "p":
                    lines = textwrap.wrap(caption, width=91)
                    fig.text(.075, y, "\n".join(lines), fontsize=9.9, linespacing=1.38, va="top")
                    y -= .0166 * len(lines) + .023
                    latex.extend([escape(caption), ""])
                elif kind == "t":
                    headers, rows = values[1:]
                    fig.text(.075, y, caption, fontsize=10.1, weight="bold", va="top")
                    y -= .027
                    height = .023 * (len(rows) + 1)
                    ax = fig.add_axes([.075, y-height, .85, height])
                    ax.axis("off")
                    widths = [max([len(str(h)), *[len(str(row[i])) for row in rows]])+3 for i,h in enumerate(headers)]
                    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", colLoc="left",
                                     colWidths=np.array(widths)/sum(widths), bbox=[0,0,1,1])
                    table.auto_set_font_size(False)
                    table.set_fontsize(8.7)
                    for (line, col), cell in table.get_celld().items():
                        cell.set_edgecolor("white")
                        cell.set_facecolor("#dde9ee" if line == 0 else ("#f2f5f6" if line%2 else "white"))
                    y -= height + .026
                    latex.extend([r"\noindent\textbf{" + escape(caption) + r"}\par",
                                  r"\begin{tabular}{" + "l"*len(headers) + "}", r"\toprule",
                                  " & ".join(map(escape, headers)) + r" \\", r"\midrule"])
                    latex.extend(" & ".join(map(escape, row)) + r" \\" for row in rows)
                    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
                else:
                    ax = fig.add_axes([.085,y-.23,.83,.23])
                    ax.imshow(plt.imread(FIGURES / "timing_by_history.png"))
                    ax.axis("off")
                    y -= .235
                    fig.text(.075, y, caption, fontsize=9, va="top")
                    y -= .04
                    latex.extend([r"\includegraphics[width=\linewidth]{../Results/figures/timing_by_history.png}",
                                  escape(caption), ""])
            if y < .045:
                raise ValueError(f"Report content exceeds page {index}.")
            fig.text(.075, .035, "GROUP 04  |  MACHINE LEARNING APPLICATIONS", fontsize=8, color="#456570")
            fig.text(.91, .035, str(index), fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
            if index < len(pages):
                latex.append(r"\clearpage")
    latex.append(r"\end{document}")
    (REPORT / "final_report.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main():
    timing_figure()
    render(report_pages())
    rows = [
        ("Table 1", "Results/tables/metrics.csv", "comparison_tables.metrics"),
        ("Table 2", "Results/tables/paired_differences.csv", "comparison_tables.paired"),
        ("Table 3", "Results/tables/metrics.csv", "comparison_tables.metrics"),
        ("Figure 1", "Results/tables/table07_history_differences.csv", "05_export_results.timing_figure"),
        ("Table 4", "Results/tables/case_review_10.csv", "history_tables.select_cases"),
        ("Table 5", "Results/tables/table10_review_budgets.csv", "history_tables.review_budgets"),
        ("Table 6", "Data/reference/historical_predictions.csv + Results/tables/metrics.csv", "report_content.report_pages"),
        ("Anchor example", "Results/fit/oof_predictions.csv", "report_content.report_pages"),
    ]
    save_table(pd.DataFrame(rows, columns=["report_item","input","source_function"]), REPORT / "report_sources.csv")
    record("05_export_results", [*sorted(TABLES.glob("*.csv")), DATA / "reference/historical_predictions.csv",
           FIT / "oof_predictions.csv", REPORT / "member_case_review.csv", *sorted((ROOT / "Code").glob("*.py"))],
           [REPORT / "Final_Report.pdf", REPORT / "final_report.tex", REPORT / "report_sources.csv",
            FIGURES / "timing_by_history.png"])


if __name__ == "__main__":
    main()

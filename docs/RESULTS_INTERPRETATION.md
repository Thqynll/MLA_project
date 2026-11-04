# Interpretation of the packaged reconstruction

This note describes the executed run delivered with the ZIP. If you rerun with changed inputs, code, or package versions, use the regenerated tables and manifest as the authoritative record.

## Main reporting focus

For the pooled 1/3 and 2/3 history groups, Combined minus Outcomes log loss is -0.007228, with a paired 95% participant-bootstrap interval of [-0.012818, -0.001421]. This is a small absolute timing increment. This reporting focus was chosen from prior feedback, so it remains exploratory rather than an original preregistered result.

## Conditional findings

- In the 1/3 group, Outcomes log loss is 0.409452 and Combined is 0.401148. The difference is -0.008304 [-0.014876, -0.002014].
- In the 2/3 group, the difference is -0.006359 [-0.014068, +0.001303]. Its individual interval contains zero.
- In the 0/3 group, the point difference is +0.000833 [-0.000093, +0.001896]. Timing is worse at the point estimate, but the new interval contains zero. The old report's interval excluded zero; do not repeat that stronger claim for this reconstruction.
- Relative minus Combined overall is +0.000021 [-0.000040, +0.000078]. There is no clear improvement on log loss.
- Process minus Combined overall is -0.001605 [-0.003375, +0.000145]. The separate action increment remains uncertain.
- In the 1/3 Relative-minus-Combined comparison, the upper CI bound is approximately +0.000007. Rounding it to +0.0000 does not make it exactly zero.

## Suggested presentation wording

In this fixed reasoning protocol, completed score history provides most of the predictive information. Response time adds a small improvement, clearest in the mixed-history participants. Relative timing does not clearly improve on raw timing, and the extra contribution of action summaries remains uncertain. In the nearly certain 0/3 group, added timing worsens the point estimate, but the reconstructed interval includes zero.

These intervals are conditional on saved OOF predictions, use no multiplicity correction, and do not establish that recording logs improves learning or enables a useful intervention. A shared LR coefficient may explain varying timing effects across history groups, but that mechanism has not been established by this analysis.

## Sources within the project

- Pooled focus: artifacts/week10_rebuilt/tables/mixed_history_differences.csv
- History-group metrics: artifacts/week10_rebuilt/tables/table06_history_outcomes_combined.csv
- History-group differences: artifacts/week10_rebuilt/tables/table07_history_differences.csv
- Overall differences: artifacts/week10_rebuilt/tables/table05_overall_differences.csv
- Relative timing: artifacts/week10_rebuilt/tables/table08_relative_timing_differences.csv
- Exact code, command, and date: artifacts/week10_rebuilt/table_sources.csv


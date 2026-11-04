# Relationship to previous reports

This package is an executed reconstruction, with fresh output dates and checksums. It implements the Week 10 development analysis and includes the earlier rebuilt Week 9 starter scripts.

The old Week 10 report's embedded script covered only Outcomes, Combined, and Relative LR in the 1/3 subgroup. It did not generate every reported prior, action-model table, or review-budget table. The exact code and provenance of that older full table run were not available for direct reconstruction.

The new run reproduces the visible legacy subgroup output to its displayed precision: Outcomes log loss about 0.4095, Combined about 0.4011, Relative about 0.4010. The old Table 6 instead printed 0.4093 and 0.4017. The precise cause of that historical discrepancy remains unverified; do not invent a seed or version explanation. Cite the new executable run and new numbers when using this package.

The direct Relative-minus-Combined interval is now included in table08_relative_timing_differences.csv for every group. Read full precision: the 1/3 upper bound is positive even though it rounds to +0.0000 at four decimals.

Every new table has a source file, function, run command, and run date in table_sources.csv. Table 07 has separate metrics and paired-difference CSVs. Tables 08 and 09 are new supporting tables for relative timing and action increments; the old report skipped those numbers.

## What this package does not claim

- It does not recreate the original full-sample Week 9 train/validation ID lists; the 746/249 split is explicitly reconstructed.
- It does not regenerate the historical 60-person pilot or the old frozen-test/CART appendix numbers. Those historical numbers are not inserted into the new result tables.
- It does not reopen or evaluate the 249-person frozen test cohort.
- It does not claim that team members have already read the ten selected cases.
- It does not finish bibliography verification for every source in the old Word report or provide a final revised Word report.
- Gradient boosting, time-by-history interaction LR, and instruction-specific follow-up models are deferred until after case review.

The project is ready for review of the main Week 10 pipeline. The final report still needs its numerical tables updated from this run, human case review, reference checks, and named member sign-off.

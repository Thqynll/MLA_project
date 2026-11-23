# End-to-end readiness audit

Reconstruction audit: 21 November 2026. Data provenance and distribution check: 22 November 2026. Scope: the English Group 04 package and the reconstructed development experiment.

| Gate | Evidence | Result |
| --- | --- | --- |
| Input identity and permission | All six original data and codebook files directly in Data/ match archived OSF SHA-256 values. The study reports CC BY 4.0 and replacement of Prolific identifiers with study-generated numeric IDs. The original main CSV includes the published demographics. Stage 01 selects only the ID, group and main-item response fields for its analysis projection. | Passed for the listed release files and projections |
| Data and label integrity | Stage 02 replayed the ordered actions, checked final-submit times and verified the public keys. It reconciled 24,880 person-item records, 143 omissions, 995 development participants, 9,950 windows and 6,587 events. | Passed |
| Matched model evaluation | Five outer and three inner participant folds, training-only preprocessing and 40 fitted LR/CART models. The frozen 249-person cohort was excluded from final prediction. Model replay agreed with the saved held-out probabilities. | Passed |
| Table and figure provenance | The 14 evaluation exports matched the November references at numerical tolerance. Report/report_sources.csv maps six displayed tables, the timing figure and the anchor example to their inputs. The historical comparison was recalculated from original October probabilities. | Passed |
| Independent reproduction | A new Windows CPython 3.14.0 virtual environment received the pinned dependencies from cached distribution wheels. A separate project directory began with code, the three original release CSVs supplied to stage 01, the frozen inputs, regression benchmarks and the empty member register. All five stages ran. The 14 resulting evaluation CSV files, window-level predictions, PDF, LaTeX source and chart matched the working run byte for byte. | Passed on the tested platform |
| Regression and syntax checks | Six unit tests passed. All Python sources compiled with bytecode directed outside the submission. The largest absolute difference across probability columns versus the saved November LR-CART predictions was zero. | Passed |
| Source and archive reconciliation | All 9,950 original and current window labels agreed. The original and current numbered outer folds agreed for 212 participants. Week 9, pilot and October results are kept distinct from the final matched model comparison. | Passed |
| Release hygiene | The public tree was checked for local absolute paths, common credential patterns, Vietnamese-specific characters, cache files and unapproved source columns. No matches were found in the applicable checks. The report text and figure labels are generated from English sources. The release check is scoped to the packaged files and the tested patterns. | Passed within the stated scope |
| Case analysis and member verification | Ten cases were selected by the recorded rule. At delivery, the member register had no completed evidence rows and the three individual source or code checks were pending. | Open for the group |
| Cross-platform confirmation | No independent Linux or macOS installation was exercised in this session. | Untested |

## Final interpretation

For the data provenance update, the three prepared CSVs were regenerated offline from the originals bundled directly in Data/. All six release files matched the archived OSF SHA-256 metadata. The Data/ review covered 27 CSV files, the README and the extracted text of both codebook PDFs. The credential and direct-contact patterns returned no matches. All participant-ID fields checked against the set of 1,244 public study IDs. The 14 evaluation tables and the held-out predictions remained byte-identical to the preceding package.

The technical package reproduces the final numerical results. The original and reconstructed 0/3 conclusions differ in interval coverage and are identified separately. Full course submission readiness still requires the group to enter genuine observations for the selected cases and document each member's source claim or code trace. After updating the case register, rerun the export and verification steps and include the updated disclosure in the submission.

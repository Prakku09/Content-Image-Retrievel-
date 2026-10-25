# Revised claims and statements

Changes apply to the paper-facing material available in the workspace: `ALL_VISUALS.md`, `FINAL_PAPER_FIGURE_AND_TABLE_INVENTORY.md`, and `README.md`. Historical experiment outputs remain unchanged.

1. **Corel split and leakage statement:** replaced the 900-gallery / 200-validation description with the corrected reference split (899 gallery, 100 held-out queries, 700 training, 199 validation); documented exclusion of `training_set/horses/761.jpg` and retention of `training_set/horses/776.jpg`.
2. **Corel model metrics:** replaced prior Corel M1-M6 values in available result tables with corrected Precision@10, Recall@10, F1@10, mAP, and ranking/sort timing; exact values appear in the generated table CSV.
3. **Figure assets/captions:** paper-facing links now reference versioned corrected Figure 2-7 assets generated from saved results, per-query predictions, and rankings. The old figure files remain untouched and must not be inserted as corrected figures.
4. **M5 versus M6 interpretation:** changed improvement statements to observed/descriptive comparisons. The corrected reference difference is +0.019258 mean paired AP, with 53 positive, 47 negative, and 0 tied query differences; this fixed-gallery result is not an inferential or robust superiority claim.
5. **Five-run summary:** added the descriptive mean seed-level difference +0.017592 and sample SD 0.004008. Clarified query/gallery overlap and category dependence; flowers dominate the positive difference, while foods and people/villages in Africa are negative in every seed.
6. **Transfer claims:** retained Oxford-IIIT Pet as a distinct frozen-transfer experiment with its own metrics and limitations. Removed any implication that one external dataset establishes universal generalization.
7. **Runtime claim:** labeled the table timing as distance calculation plus ranking/sorting per query, with feature extraction and MLP inference outside that measurement.
8. **Citation integrity:** no references or citations were added, removed, or rewritten. Since the actual manuscript/bibliography was absent, citation placement and citation-dependent claims in that document remain unreviewed.

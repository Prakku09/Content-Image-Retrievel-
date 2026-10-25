# Final paper consistency report

## Verified consistency

- Corrected Corel aggregate values in the paper-facing results table and new plots are generated from `paper_results/corrected_corel_v1/validation/corel_reference/results.json`.
- Corrected mAP values are M1 0.404462, M2 0.350816, M3 0.465036, M4 0.433659, M5 0.479794, and M6 0.460536.
- The corrected split is consistently documented as 899 gallery, 100 held-out queries, 700 MLP training images, and 199 validation images. The gallery-side near-duplicate `training_set/horses/761.jpg` is excluded; `training_set/horses/776.jpg` remains a query.
- The new plots use the same metric values as the corrected table. Figure 2 labels are rounded to three decimals and Figure 6 labels to four decimals; CSV table values preserve the source precision.
- Five-seed M5âˆ’M6 mean mAP difference (+0.017592) and sample SD (0.004008) match the reproducibility manifest. The text labels these results descriptive and notes shared images and category dependence.
- Oxford-IIIT Pet transfer values and limitations are separated from Corel results and do not support a universal generalization claim.
- Historical result files and the prior `figures/` assets were not overwritten. New corrected figures, including a replacement pipeline diagram without the obsolete historical mAP callout, are stored under `paper_results/corrected_corel_v1/assets_v4/`.
- The asset manifest records source and generated-file hashes; paper-specific source and artifact provenance is recorded in `paper_manifest.json`.

## Remaining discrepancies and limits

1. **The complete research manuscript was not present in the workspace.** The available paper-facing documents are `ALL_VISUALS.md`, `FINAL_PAPER_FIGURE_AND_TABLE_INVENTORY.md`, and `README.md`; no manuscript `.docx`, `.tex`, or PDF was found. Therefore the actual abstract, body tables, bibliography/citations, conclusion, and embedded figures could not be searched or modified. Replacement sections are provided in `revised_manuscript_sections.md`, but a fully integrated paper cannot be certified.
2. The historical binary figures in `figures/` remain unchanged by design. Paper-facing Markdown now points to the corrected versioned assets, but any manuscript file elsewhere that embeds these old images must replace them manually or be supplied for editing.
3. The corrected five-seed summary is descriptive, not independent-replication inference. The five splits share gallery content, and the 500 query observations contain only 399 unique images. The category differences are heterogeneous, with flowers strongly positive and two categories negative in all five runs.

## Release decision

The corrected machine-readable results, tables, and regenerated paper assets are consistent and verified. **Do not declare the paper finalized or ready for submission until the actual manuscript is supplied and every embedded value, statement, citation context, table, figure, caption, abstract claim, and conclusion has been reconciled against these artifacts.**

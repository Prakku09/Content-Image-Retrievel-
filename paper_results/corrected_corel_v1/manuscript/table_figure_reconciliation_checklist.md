# Table and figure reconciliation checklist

Source of truth: independently validated corrected Corel run `run_20261025T130024Z`; final corrected output assets are under `assets_v4/`. Historical assets under `figures/` were not overwritten.

| Item | Source/validation | Corrected action | Status |
|---|---|---|---|
| Table III: M1-M6 metrics | `results.json`; saved rankings independently recomputed with max absolute error 0 | Rebuilt as `assets_v4/table_III_corrected_metrics.csv`; P@10, R@10, F1@10, mAP and ranking/sort ms per query use source values | Verified |
| Fig. 2: grouped metrics | Same corrected aggregate metrics as Table III | Regenerated PNG and SVG from `results.json`; plot annotation precision is 3 decimals, table retains exact reported precision | Verified |
| Fig. 3: M5 weight distributions | `per_query_weights.csv` from corrected checkpoint; 100 held-out queries | Regenerated histograms (density scale), PNG and SVG | Verified |
| Fig. 4: category mean weights | `per_query_weights.csv`; category means recomputed from 10 query predictions/category | Regenerated stacked weight plot, PNG and SVG | Verified |
| Fig. 5: qualitative rankings | `rankings.npz`; first held-out beaches query; M3 and M5 top-10 rankings | Regenerated with actual corrected gallery/query images and relevance labels | Verified |
| Fig. 6: mAP comparison | `results.json` mAP for M1-M6 | Regenerated; labels rounded to four decimals, source table retains six mAP decimals | Verified |
| Fig. 7: category AP differences | `per_category_metrics.csv`; M5 minus M6 category means recomputed | Added plot with all 10 categories, including negative differences | Verified |
| Fig. 1: corrected method diagram | Corrected split, experiment configuration, and verified methodology | Replaced obsolete historical result/gain callout with corrected split and validation-only selection flow | Verified |
| Corrected split/method note | `split_manifest.csv`, `duplicate_audit.json`, `experiment_config.json` | Document 899 gallery, 100 queries, 700 train, 199 validation; exclude gallery `761.jpg`, retain query `776.jpg` | Verified |
| Five-run statement | Prespecified `seed_plan.json`, five validated `results.json` files and `summary_v1/reproducibility_manifest.json` | Report +0.017592 mean seed-level mAP difference, sample SD 0.004008; state identity overlap/dependence | Verified |
| Oxford transfer | Corrected frozen transfer `transfer_report.md` and independent validation | Kept separate and explicitly limited to one dataset/checkpoint; not universal generalization | Verified |
| Main manuscript integration | No `.docx`, `.tex`, manuscript PDF, or complete manuscript file was present in workspace | Corrected replacement sections and paper-facing Markdown inventory supplied; cannot verify/replace absent abstract, tables, figures, captions, or conclusion in the actual paper file | **Blocked: manuscript file required** |

## Figure-to-table numeric check

All figure source data are generated from the same corrected `results.json` or corrected per-query files. Figure 2 uses the four Table III metrics in the same order (Precision@10, Recall@10, F1@10, mAP); Figure 6 uses precisely the Table III mAP values. Their labels are rounded for legibility only. Exact numeric values are preserved in the CSV table and asset manifest. Category figures are recalculated from saved query-level weights and per-category metrics; no hand-entered values are used.

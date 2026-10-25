# Final IEEE Paper Figure and Table Inventory

This document inventories the paper assets available for the Corel-1K CBIR project. Historical files remain in `figures/`; corrected paper assets are in the versioned `paper_results/corrected_corel_v1/assets_v4/` directory.

---

## 1. Primary Figures for Main IEEE Paper

A set of publication-ready figures is available to support the methodology, quantitative evaluation, ablation analysis, and qualitative retrieval results.

* **Figure 1: Corrected Framework Block Diagram**
  * *Corrected source*: [`fig1_corrected_pipeline.png`](paper_results/corrected_corel_v1/assets_v4/fig1_corrected_pipeline.png) (300 DPI) & [`fig1_corrected_pipeline.svg`](paper_results/corrected_corel_v1/assets_v4/fig1_corrected_pipeline.svg) (Vector)
  * *Purpose*: Shows the corrected split, duplicate exclusion, gallery-fitted normalization, validation-only checkpoint selection, M1-M6, and evaluation.
  * *Status*: **Regenerated**; removes the obsolete historical result callout. The previous diagram remains preserved in `figures/`.

* **Figure 2: Model Performance Comparison (Grouped Bar Chart)**
  * *Corrected source*: [`fig2_corrected_grouped_metrics.png`](paper_results/corrected_corel_v1/assets_v4/fig2_corrected_grouped_metrics.png) (300 DPI) & [`fig2_corrected_grouped_metrics.svg`](paper_results/corrected_corel_v1/assets_v4/fig2_corrected_grouped_metrics.svg) (Vector)
  * *Purpose*: Multi-metric evaluation (Precision@10, Recall@10, F1-score@10, mAP) across M1â€“M6, including the M6 fixed mean-weight control.
  * *Status*: **Regenerated from independently validated corrected Corel run `run_20261025T130024Z`**. Values correspond to the corrected 899-image gallery and 100 held-out queries.

* **Figure 3: MLP-Predicted Weight Distributions**
  * *Corrected source*: [`fig3_corrected_weight_distributions.png`](paper_results/corrected_corel_v1/assets_v4/fig3_corrected_weight_distributions.png) (300 DPI) & [`fig3_corrected_weight_distributions.svg`](paper_results/corrected_corel_v1/assets_v4/fig3_corrected_weight_distributions.svg) (Vector)
  * *Purpose*: Shows histograms of corrected-run predicted colour and texture weights across 100 held-out queries. Variation alone is not evidence of retrieval improvement.
  * *Status*: **Regenerated from the corrected checkpoint's saved query weights**.

* **Figure 4: Category-Wise Mean Weights**
  * *Corrected source*: [`fig4_corrected_category_weights.png`](paper_results/corrected_corel_v1/assets_v4/fig4_corrected_category_weights.png) (300 DPI) & [`fig4_corrected_category_weights.svg`](paper_results/corrected_corel_v1/assets_v4/fig4_corrected_category_weights.svg) (Vector)
  * *Purpose*: Shows descriptive average weights per category; it makes no claim that category-level adaptation causes retrieval improvement.
  * *Status*: **Regenerated from saved corrected-run per-query predictions**.

* **Figure 5: M3 vs. M5 Qualitative Search Grid**
  * *Corrected source*: [`fig5_corrected_qualitative_example.png`](paper_results/corrected_corel_v1/assets_v4/fig5_corrected_qualitative_example.png) (300 DPI) & [`fig5_corrected_qualitative_example.svg`](paper_results/corrected_corel_v1/assets_v4/fig5_corrected_qualitative_example.svg) (Vector)
  * *Purpose*: Shows actual saved top-10 rankings for the first held-out beaches query in the corrected run. It is qualitative, not evidence of general improvement.
  * *Status*: **Regenerated from saved corrected rankings**.

---

## 2. Supplementary Figures

| Figure / Plot | File Path | Format | Role Recommendation |
| :--- | :--- | :---: | :--- |
| **Annotated Grouped Bar Chart** | [`fig2_corrected_grouped_metrics.png`](paper_results/corrected_corel_v1/assets_v4/fig2_corrected_grouped_metrics.png) | PNG (300 DPI) & SVG | **SUPPLEMENTARY MATERIAL** (corrected run; M5/M6 annotations) |
| **mAP-Only Bar Chart** | [`fig6_corrected_map.png`](paper_results/corrected_corel_v1/assets_v4/fig6_corrected_map.png) | PNG (300 DPI) & SVG | **SUPPLEMENTARY MATERIAL** (corrected run) |
| **Category AP Difference** | [`fig7_corrected_category_ap_difference.png`](paper_results/corrected_corel_v1/assets_v4/fig7_corrected_category_ap_difference.png) | PNG (300 DPI) & SVG | **SUPPLEMENTARY / category heterogeneity** |

---

## 3. IEEE Captions

Below are the formatted captions matching the IEEE style manual:

* **Fig. 1.** Corrected CBIR protocol and pipeline, showing the duplicate-audited split, gallery-only normalization, validation-only checkpoint selection, model fusion strategies, and evaluation.
* **Fig. 2.** Quantitative retrieval performance comparison across six CBIR models (M1â€“M6) on the corrected Corel-1K benchmark using Precision@10, Recall@10, F1-score@10, and mAP. M6 is the fixed mean-weight control using the M5 direct-weight distance.
* **Fig. 3.** Histograms of predicted colour and texture weights across the 100 held-out queries in the corrected Corel-1K run. Weight variation alone does not demonstrate retrieval improvement.
* **Fig. 4.** Category mean M5-predicted colour and texture weights for the corrected Corel-1K query set; descriptive only.
* **Fig. 5.** Top-10 rankings for the first held-out beaches query in the corrected saved outputs, comparing M3 and M5. Green denotes same-category relevance and red denotes other categories; this is one qualitative example.
* **Fig. 6.** Corrected mAP across M1-M6 on the 899-image gallery.
* **Fig. 7.** Category-level mean AP difference (M5 minus M6) on the corrected query set; positive and negative categories are both shown.

---

## 4. Table Inventory

### Table I: Dataset and Experimental Configuration
* **Description**: Corrected split: 899 gallery, 100 held-out queries, 700 MLP training, 199 validation images. `training_set/horses/761.jpg` is excluded after visual confirmation of its near-duplicate relationship to retained query `training_set/horses/776.jpg`.
* **Source**: Corrected split manifest and duplicate audit in [`run_20261025T130024Z`](paper_results/corrected_corel_v1/validation/corel_reference/).

### Table II: Feature Descriptors
* **Description**: Lists dimensions and parameters for HSV moments (9-dim) and GLCM texture features (4-dim).
* **Source**: [`features/feature_extractor.py`](features/feature_extractor.py)

### Table III: M1â€“M6 Retrieval Performance Comparison
* **Description**: Metrics (Precision@10, Recall@10, F1-score@10, mAP) and distance-plus-ranking time from the current evaluation run.
* **Source of truth**: [`results.json`](paper_results/corrected_corel_v1/validation/corel_reference/results.json), independently checked against saved rankings; exact values exported in [`table_III_corrected_metrics.csv`](paper_results/corrected_corel_v1/assets_v4/table_III_corrected_metrics.csv).

| Model | Descriptor / Strategy | Precision@10 | Recall@10 | F1-score@10 | mAP | Mean retrieval time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M1** | HSV Colour Moments (9-dim) | 0.6190 | 0.0689 | 0.1240 | 0.404462 | 0.082 ms |
| **M2** | GLCM Texture (4-dim) | 0.4680 | 0.0521 | 0.0937 | 0.350816 | 0.073 ms |
| **M3** | Equal feature-level fusion ($w_c=0.5, w_t=0.5$) | 0.6990 | 0.0778 | 0.1400 | 0.465036 | 0.079 ms |
| **M4** | Fixed feature-level fusion ($w_c=0.7, w_t=0.3$) | 0.6730 | 0.0749 | 0.1348 | 0.433659 | 0.077 ms |
| **M5** | Query-adaptive distance-level weighting | 0.7100 | 0.0790 | 0.1422 | 0.479794 | 0.167 ms |
| **M6** | Fixed mean-weight distance-level control | 0.6960 | 0.0774 | 0.1394 | 0.460536 | 0.153 ms |

Timing is distance calculation plus ranking/sorting only; feature extraction and MLP inference are timed separately. On the corrected reference split M5 has higher observed mAP than M6, but that single-split difference is descriptive. Across the five prespecified duplicate-group-disjoint runs the seed-level mean M5-M6 mAP difference was +0.017592 (sample SD 0.004008), also descriptive because runs share gallery content and query identities. Category gains are heterogeneous and driven primarily by flowers; do not claim robust statistical superiority.

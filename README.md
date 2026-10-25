# Content-Based Image Retrieval on Corel-1K using Adaptive Colour-Texture Fusion

This repository provides a Content-Based Image Retrieval (CBIR) research pipeline evaluated on the Corel-1K dataset. It implements statistical colour moments, Gray-Level Co-occurrence Matrix (GLCM) texture descriptors, four baselines (M1â€“M4), an MLP adaptive model (M5), and a fixed mean-weight control (M6).

---

## 1. Research Objective
The primary objective of this project is to model and evaluate Content-Based Image Retrieval performance using joint colour and texture descriptors. While traditional methods rely on hand-tuned, fixed-weight combinations (such as M3 and M4), this work introduces model **M5**, which leverages a metric-trained Multi-Layer Perceptron (MLP) to predict query-adaptive fusion weights. 

---

## 2. Corel-1K Dataset
The pipeline evaluates retrieval on the Corel-1K dataset:
- **Total Images**: 1,000 images
- **Categories**: 10 distinct category directories (`beaches`, `bus`, `dinosaurs`, `elephants`, `flowers`, `foods`, `horses`, `monuments`, `mountains_and_snow`, and `peolpe_and_villages_in_Africa`).
- **Images per Category**: Exactly 100 images per category.
- **Formats**: BMP, JPEG, JPG, and PNG are fully handled.

---

## 3. Feature Extraction

### Colour Features (HSV Space)
Images are converted to the HSV colour space, and for each of the three channels (Hue, Saturation, Value), the first three statistical moments are calculated over all pixels:
1. **Mean** (average channel intensity)
2. **Standard Deviation** (channel variance/dispersion)
3. **Skewness** (asymmetry of the intensity distribution)

This yields exactly **9 colour features** per image (3 HSV channels Ã— 3 moments).

### Texture Features (GLCM)
Images are converted to grayscale (`uint8` format) to extract spatial texture properties using the Gray-Level Co-occurrence Matrix (GLCM):
- **Distance**: 1 pixel
- **Angles**: 0Â°, 45Â°, 90Â°, and 135Â° (expressed in radians as 0, pi/4, pi/2, and 3*pi/4)
- **Extracted Properties**: Contrast, Correlation, Energy, and Homogeneity are computed for each orientation and then averaged across the four angles, yielding exactly **4 texture features** per image.

Each image is represented by a **13-dimensional** initial feature vector (9 colour + 4 texture).

---

## 4. Corrected Query-Gallery Split
The corrected paper evaluation and its reproducibility artifacts are packaged under `paper_results/corrected_corel_v1/`:
- **Seed**: Query and MLP seed 42.
- **Gallery Set**: 899 images after excluding `training_set/horses/761.jpg`, which the duplicate audit and visual review identified as a near-duplicate of held-out query `training_set/horses/776.jpg`.
- **Query Set**: 100 images; the identified query is retained.
- **MLP training/validation**: 700/199 gallery images, with held-out queries excluded from training, validation and checkpoint selection.
- Category labels are preserved in the split manifest.

The original 900-gallery outputs are historical and are not leakage-free estimates for this corrected protocol. The corrected run, manifests and validation are stored separately; do not overwrite or present old outputs as corrected results.

---

## 5. Min-Max Normalization
For the corrected protocol, Min-Max bounds are fitted strictly using the **899 corrected gallery features** and applied to gallery and query features. Held-out queries and the excluded near-duplicate are not used to fit normalization.

---

## 6. Retrieval Models (M1â€“M6)

### M1 â€” Colour Only
Retrieval is performed using only the 9 normalized HSV colour features. Ranking is based on ordinary Euclidean distance.

### M2 â€” Texture Only
Retrieval is performed using only the 4 normalized GLCM texture features. Ranking is based on ordinary Euclidean distance.

### M3 â€” Equal Fusion
Fuses normalised colour and texture features using equal weight coefficients:
- Colour Weight $w_C = 0.5$
- Texture Weight $w_T = 0.5$
- Equivalent distance ranking formula: $D_{M3} = \sqrt{0.25 D_c^2 + 0.25 D_t^2}$

### M4 â€” Fixed Fusion
Fuses colour and texture using a predetermined fixed baseline weight scheme:
- Colour Weight $w_C = 0.7$
- Texture Weight $w_T = 0.3$
- Equivalent distance ranking formula: $D_{M4} = \sqrt{0.49 D_c^2 + 0.09 D_t^2}$

### M5 â€” Proposed AI-Based Adaptive Weighting
M5 learns the relative contribution of colour and texture dynamically for each query.
- **MLP Input**: 13-dimensional concatenated normalized query feature vector.
- **MLP Outputs**: Raw activations mapping to $w_{colour}(q)$ and $w_{texture}(q)$.
- **Weight Normalization**: Softmax is applied to ensure weights are non-negative and sum to 1 within floating-point tolerance ($w_{colour}(q) + w_{texture}(q) \approx 1$).
- **Retrieval Distance**: 
  $$D_{M5}(q, x) = \sqrt{w_{colour}(q) \cdot D_{colour}(q,x)^2 + w_{texture}(q) \cdot D_{texture}(q,x)^2}$$

### M6 â€” Fixed Mean-Weight Control
M6 uses the mean of the two M5 predicted weights across the 100 held-out queries for every query. It uses the exact M5 direct distance formula and is an evaluation baseline, not another trained model. Query labels are not used to calculate the mean weights. The comparison evaluates whether per-query weight variation is associated with retrieval differences beyond one fixed pair; observed metric differences do not by themselves establish a general benefit.

### Formula distinction
M3 and M4 weight the feature blocks before concatenation, which induces the squared-distance formula
$$D(q,x) = \sqrt{w_{colour}^2 D_{colour}(q,x)^2 + w_{texture}^2 D_{texture}(q,x)^2}.$$
M5 and M6 use the weights directly in the squared distances, without squaring the coefficients. The two approaches are not numerically equivalent. The implementation intentionally preserves the existing M3/M4 baselines and reports the distinction rather than silently changing their methodology.
For ranking, M3's equal squared coefficients differ from direct 0.5/0.5 weights only by a common scale, so its ordering is unchanged. M4's squared coefficients have the relative ratio 0.49:0.09 (about 0.845:0.155 after normalization), not the direct 0.7:0.3 ratio. M6 uses the M5 distance formula so the M5/M6 comparison isolates per-query weight variation.

---

## 7. MLP Architecture & Metric Learning Training

### Network Architecture
- **Layers**: Input (13) -> Linear (16) -> LeakyReLU(0.1) -> Linear (8) -> LeakyReLU(0.1) -> Linear (2) -> Softmax
- **Parameters**: Exactly 378 trainable parameters (a lightweight network designed to support reproducibility and prevent overfitting).

### Training and Validation Protocol
- **Corrected reference run**: The 899-image gallery is partitioned into 700 MLP training and 199 validation images (seed 42). The held-out 100 queries are not included in training, validation, checkpoint selection, or normalization fitting. The removed duplicate pair and all paths/assignments are recorded in the corrected split manifest.
- **Triplet Margin Loss**: Model is trained using pre-generated triplet configurations (q, p, n), where p is a same-category image and n is a different-category image.
  $$\mathcal{L} = \max(0, D^2_{M5}(q, p) - D^2_{M5}(q, n) + 0.15)$$
- **Optimization**: Adam optimizer (learning rate = 0.001, batch size = 64).
- **Early Stopping**: Monitored strictly on internal validation triplet loss with a patience of 10 epochs.

---

## 8. Corrected Experimental Results

The verified corrected reference run's publication package, validation reports, and machine-readable results are in [`paper_results/corrected_corel_v1/`](paper_results/corrected_corel_v1/). Aggregate metrics were independently recomputed from saved rankings. Do not present the historical fixed-path `output/part2/` results or the old `figures/` performance plots as corrected estimates.

| Model | Precision@10 | Recall@10 | F1@10 | mAP |
|---|---:|---:|---:|---:|
| M1 â€” Colour only | 0.6190 | 0.0689 | 0.1240 | 0.404462 |
| M2 â€” Texture only | 0.4680 | 0.0521 | 0.0937 | 0.350816 |
| M3 â€” Equal feature-level fusion | 0.6990 | 0.0778 | 0.1400 | 0.465036 |
| M4 â€” Fixed 0.7/0.3 feature-level fusion | 0.6730 | 0.0749 | 0.1348 | 0.433659 |
| M5 â€” Query-adaptive distance-level fusion | 0.7100 | 0.0790 | 0.1422 | 0.479794 |
| M6 â€” Fixed mean-weight distance-level control | 0.6960 | 0.0774 | 0.1394 | 0.460536 |

The corrected reference split contains 899 gallery images, 100 held-out queries, 700 MLP-training images, and 199 validation images. `training_set/horses/761.jpg` was excluded after a visually confirmed near-duplicate audit; `training_set/horses/776.jpg` remains a held-out query. The split manifest and validation report are included in the paper-results package.

Across five prespecified duplicate-group-disjoint runs, the descriptive mean seed-level M5-M6 mAP difference is +0.017592 (sample SD 0.004008). The runs share gallery content and overlap in query identities, and category-level results are heterogeneous. This is not evidence of inferential or robust statistical superiority. The separate Oxford-IIIT Pet frozen-transfer result is limited to that dataset and checkpoint; it does not establish universal generalization.

Use [`corrected_corel_run.py`](corrected_corel_run.py) and its versioned output directory for the corrected protocol. The legacy `main.py` fixed-path pipeline refuses to overwrite existing artifacts unless `--allow-overwrite-results` is explicitly supplied; that option does not make the legacy split duplicate-safe. Historical files are retained for provenance.

---

## 9. Repository Structure

```text
â”œâ”€â”€ config.py                           # Central configuration source of truth
â”œâ”€â”€ main.py                             # Unified CLI runner (runs full pipeline or stages)
â”œâ”€â”€ sanity_check.py                     # Unified test suite
â”œâ”€â”€ README.md                           # Documentation
â”œâ”€â”€ FINAL_PAPER_FIGURE_AND_TABLE_INVENTORY.md  # Inventory of publication assets
â”œâ”€â”€ paper_results/
â”‚   â””â”€â”€ corrected_corel_v1/              # Corrected results, figures, tables, and provenance
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ dataset_loader.py               # Legacy Corel-1K loader and split
â”‚   â””â”€â”€ validate_dataset.py             # Dataset verification tool
â”‚
â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ feature_extractor.py            # HSV moments (9-dim) and GLCM texture (4-dim)
â”‚   â””â”€â”€ normalization.py                # MinMaxNormalizer fitted on gallery
â”‚
â”œâ”€â”€ model/
â”‚   â”œâ”€â”€ mlp_model.py                    # PyTorch AdaptiveWeightMLP architecture
â”‚   â””â”€â”€ mlp_trainer.py                  # Metric learning (triplet loss) trainer
â”‚
â”œâ”€â”€ retrieval/
â”‚   â”œâ”€â”€ retrieval.py                    # Concatenation and Euclidean distance ranking
â”‚   â””â”€â”€ evaluation.py                   # Precision@K, Recall@K, F1@K, mAP metrics
â”‚
â”œâ”€â”€ visualization/
â”‚   â”œâ”€â”€ visualization.py                # Retrieval grid and baseline plotting
â”‚   â”œâ”€â”€ visualize_part2.py              # M1-M6 comparisons, weight distributions
â”‚   â””â”€â”€ generate_grouped_bar_chart.py   # Legacy fixed-path chart generator
â”‚
â”œâ”€â”€ corrected_corel_run.py              # Corrected duplicate-audited evaluation runner
â”œâ”€â”€ validate_corrected_corel_run.py     # Independent ranking/metric validation
â”œâ”€â”€ paper_assets/
â”‚   â””â”€â”€ generate_corrected_paper_assets.py # Versioned corrected paper plots/tables
â”‚
â”œâ”€â”€ figures/                            # Historical assets; not corrected paper results
â”‚   â”œâ”€â”€ fig1_pipeline_diagram.svg       # Framework Block Diagram (Vector)
â”‚   â”œâ”€â”€ fig1_pipeline_diagram.png       # Framework Block Diagram (300 DPI)
â”‚   â”œâ”€â”€ cbir_model_comparison_grouped_bar.png # Grouped Bar Chart (300 DPI)
â”‚   â”œâ”€â”€ cbir_model_comparison_grouped_bar.svg # Grouped Bar Chart (Vector)
â”‚   â”œâ”€â”€ cbir_model_comparison_annotated.png   # Annotated Grouped Bar Chart (300 DPI)
â”‚   â”œâ”€â”€ cbir_model_comparison_annotated.svg   # Annotated Grouped Bar Chart (Vector)
â”‚   â”œâ”€â”€ fig2_model_performance_comparison.png # mAP Comparison (300 DPI)
â”‚   â”œâ”€â”€ fig3_weight_distributions.png   # Learned Weight Distributions KDE (300 DPI)
â”‚   â”œâ”€â”€ fig4_category_weights.png       # Category-Wise Mean Weights (300 DPI)
â”‚   â””â”€â”€ fig5_query_m3_vs_m5_beaches.png # M3 vs M5 Qualitative Search Grid (300 DPI)
â”‚
â””â”€â”€ output/                             # Local generated caches and historical experiment artifacts
```

---

## 10. Execution Instructions

### Run the corrected reference experiment
```bash
python corrected_corel_run.py --dataset-root "<Corel dataset>" --output-root "output/corrected_corel_v1"
python validate_corrected_corel_run.py "<versioned run directory>"
```

Generate and verify the versioned corrected paper assets:
```bash
python paper_assets/generate_corrected_paper_assets.py --run-dir "<validated run directory>" --output-dir "output/paper_final_v1/assets_new" --dataset-root "<Corel dataset>"
python paper_assets/validate_corrected_paper_assets.py --run-dir "<validated run directory>" --asset-dir "output/paper_final_v1/assets_new"
```

`main.py` is the legacy fixed-path pipeline. It is not the duplicate-safe corrected runner and refuses to overwrite existing artifacts unless explicitly opted into. Do not use it to produce final corrected paper results.

### Prespecified multi-seed descriptive runs

The versioned five-seed summary under `paper_results/corrected_corel_v1/validation/five_seed_summary/` records the completed, validated descriptive comparison. They must be described as dependent, descriptive evaluations, not independent replications. Earlier `output/statistical_evaluation_v1/` artifacts predate the duplicate-corrected protocol and are historical.

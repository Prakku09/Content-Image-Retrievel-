# Content-Based Image Retrieval on Corel-1K using Adaptive Colour-Texture Fusion

This repository provides a Content-Based Image Retrieval (CBIR) research pipeline evaluated on the Corel-1K dataset. It implements statistical colour moments, Gray-Level Co-occurrence Matrix (GLCM) texture descriptors, and evaluates four baseline models (M1–M4) against a proposed Multi-Layer Perceptron (MLP) adaptive distance-level fusion model (M5).

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

This yields exactly **9 colour features** per image (3 HSV channels × 3 moments).

### Texture Features (GLCM)
Images are converted to grayscale (`uint8` format) to extract spatial texture properties using the Gray-Level Co-occurrence Matrix (GLCM):
- **Distance**: 1 pixel
- **Angles**: 0°, 45°, 90°, and 135° (expressed in radians as 0, pi/4, pi/2, and 3*pi/4)
- **Extracted Properties**: Contrast, Correlation, Energy, and Homogeneity are computed for each orientation and then averaged across the four angles, yielding exactly **4 texture features** per image.

Each image is represented by a **13-dimensional** initial feature vector (9 colour + 4 texture).

---

## 4. Query-Gallery Split
To support evaluation:
- **Seed**: Fixed random seed (42) to support reproducibility.
- **Gallery Set**: 900 images total (90 images randomly selected per category).
- **Query Set**: 100 images total (10 images randomly selected per category).
- Category labels are permanently preserved and associated with every image.

---

## 5. Min-Max Normalization
To prevent data leakage:
- Min-Max bounds are fitted **strictly using the 900 gallery features**.
- The fitted bounds are applied consistently to scale both the gallery features and the query features to the range [0, 1].

---

## 6. Retrieval Models (M1–M5)

### M1 — Colour Only
Retrieval is performed using only the 9 normalized HSV colour features. Ranking is based on ordinary Euclidean distance.

### M2 — Texture Only
Retrieval is performed using only the 4 normalized GLCM texture features. Ranking is based on ordinary Euclidean distance.

### M3 — Equal Fusion
Fuses normalised colour and texture features using equal weight coefficients:
- Colour Weight $w_C = 0.5$
- Texture Weight $w_T = 0.5$
- Equivalent distance ranking formula: $D_{M3} = \sqrt{0.25 D_c^2 + 0.25 D_t^2}$

### M4 — Fixed Fusion
Fuses colour and texture using a predetermined fixed baseline weight scheme:
- Colour Weight $w_C = 0.7$
- Texture Weight $w_T = 0.3$
- Equivalent distance ranking formula: $D_{M4} = \sqrt{0.49 D_c^2 + 0.09 D_t^2}$

### M5 — Proposed AI-Based Adaptive Weighting
M5 learns the relative contribution of colour and texture dynamically for each query.
- **MLP Input**: 13-dimensional concatenated normalized query feature vector.
- **MLP Outputs**: Raw activations mapping to $w_{colour}(q)$ and $w_{texture}(q)$.
- **Weight Normalization**: Softmax is applied to ensure weights are non-negative and sum to exactly 1.0 ($w_{colour}(q) + w_{texture}(q) = 1$).
- **Retrieval Distance**: 
  $$D_{M5}(q, x) = \sqrt{w_{colour}(q) \cdot D_{colour}(q,x)^2 + w_{texture}(q) \cdot D_{texture}(q,x)^2}$$

---

## 7. MLP Architecture & Metric Learning Training

### Network Architecture
- **Layers**: Input (13) -> Linear (16) -> LeakyReLU(0.1) -> Linear (8) -> LeakyReLU(0.1) -> Linear (2) -> Softmax
- **Parameters**: Exactly 378 trainable parameters (a lightweight network designed to support reproducibility and prevent overfitting).

### Training and Validation Protocol
- **Internal Split**: The 900 gallery images are split into an internal training set (700 images, 70/cat) and validation set (200 images, 20/cat) using random seed 42.
- **Triplet Margin Loss**: Model is trained using pre-generated triplet configurations (q, p, n), where p is a same-category image and n is a different-category image.
  $$\mathcal{L} = \max(0, D^2_{M5}(q, p) - D^2_{M5}(q, n) + 0.15)$$
- **Optimization**: Adam optimizer (learning rate = 0.001, batch size = 64).
- **Early Stopping**: Monitored strictly on internal validation triplet loss with a patience of 10 epochs.

---

## 8. Final M1–M5 Quantitative Results

| Model | Mean Precision@10 | Mean Recall@10 | Mean F1-score@10 | mAP (Full Gallery) | Avg Retrieval Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 (Colour Only)** | 0.6200 | 0.0689 | 0.1240 | 0.4048 | 0.13 ms |
| **M2 (Texture Only)** | 0.4680 | 0.0520 | 0.0936 | 0.3508 | 0.09 ms |
| **M3 (Equal Fusion)** | 0.6990 | 0.0777 | 0.1398 | 0.4652 | 0.10 ms |
| **M4 (Fixed Fusion)** | 0.6730 | 0.0748 | 0.1346 | 0.4340 | 0.11 ms |
| **M5 (AI Adaptive)** | **0.7200** | **0.0800** | **0.1440** | **0.4845** | 0.15 ms |

---

## 9. Repository Structure

```text
├── config.py                           # Central configuration source of truth
├── main.py                             # Unified CLI runner (runs full pipeline or stages)
├── sanity_check.py                     # Unified test suite
├── README.md                           # Documentation
├── FINAL_PAPER_FIGURE_AND_TABLE_INVENTORY.md  # Inventory of publication assets
│
├── data/
│   ├── dataset_loader.py               # Corel-1K recursive loader & 100/900 split
│   └── validate_dataset.py             # Dataset verification tool
│
├── features/
│   ├── feature_extractor.py            # HSV moments (9-dim) and GLCM texture (4-dim)
│   └── normalization.py                # MinMaxNormalizer fitted on gallery
│
├── model/
│   ├── mlp_model.py                    # PyTorch AdaptiveWeightMLP architecture
│   └── mlp_trainer.py                  # Metric learning (triplet loss) trainer
│
├── retrieval/
│   ├── retrieval.py                    # Concatenation and Euclidean distance ranking
│   └── evaluation.py                   # Precision@K, Recall@K, F1@K, mAP metrics
│
├── visualization/
│   ├── visualization.py                # Retrieval grid and baseline plotting
│   ├── visualize_part2.py              # M1-M5 comparisons, weight distributions
│   └── generate_grouped_bar_chart.py   # Publication-grade grouped bar chart generator
│
├── figures/                            # High-resolution (300 DPI) & vector paper assets
│   ├── fig1_pipeline_diagram.svg       # Framework Block Diagram (Vector)
│   ├── fig1_pipeline_diagram.png       # Framework Block Diagram (300 DPI)
│   ├── cbir_model_comparison_grouped_bar.png # Grouped Bar Chart (300 DPI)
│   ├── cbir_model_comparison_grouped_bar.svg # Grouped Bar Chart (Vector)
│   ├── cbir_model_comparison_annotated.png   # Annotated Grouped Bar Chart (300 DPI)
│   ├── cbir_model_comparison_annotated.svg   # Annotated Grouped Bar Chart (Vector)
│   ├── fig2_model_performance_comparison.png # mAP Comparison (300 DPI)
│   ├── fig3_weight_distributions.png   # Learned Weight Distributions KDE (300 DPI)
│   ├── fig4_category_weights.png       # Category-Wise Mean Weights (300 DPI)
│   └── fig5_query_m3_vs_m5_beaches.png # M3 vs M5 Qualitative Search Grid (300 DPI)
│
└── output/                             # Generated feature caches & checkpoints
    ├── corel1k_features.pkl
    └── part2/
        ├── adaptive_weight_mlp.pt
        └── part2_results.pkl
```

---

## 10. Execution Instructions

### Run the Full Pipeline End-to-End
```bash
python main.py
```

### Run Specific Stages
```bash
# Run sanity checks
python main.py --stage sanity
# or directly:
python sanity_check.py

# Run only baseline models (M1-M4)
python main.py --stage part1

# Run only MLP training & M5 evaluation
python main.py --stage part2

# Regenerate publication figures into figures/
python main.py --stage figures
```

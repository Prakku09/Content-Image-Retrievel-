# Final IEEE Paper Figure and Table Inventory

This document provides a comprehensive inventory of all visual and tabular assets available for the final IEEE research paper on the Corel-1K CBIR project. All assets are located in the [`figures/`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/) folder.

---

## 1. Primary Figures for Main IEEE Paper

A set of publication-ready figures is available to support the methodology, quantitative evaluation, ablation analysis, and qualitative retrieval results.

* **Figure 1: Framework Block Diagram**
  * *Source*: [`figures/fig1_pipeline_diagram.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/fig1_pipeline_diagram.png) (300 DPI) & [`figures/fig1_pipeline_diagram.svg`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/fig1_pipeline_diagram.svg) (Lossless Vector)
  * *Purpose*: Explains the end-to-end system architecture: feature extraction (HSV + GLCM), query feature vector preparation (13-dim), feed-forward MLP inference, softmax-based weight calculation, and adaptive distance calculation.
  * *Status*: **READY** (300 DPI PNG & Scalable SVG).

* **Figure 2: Model Performance Comparison (Grouped Bar Chart)**
  * *Source*: [`figures/cbir_model_comparison_grouped_bar.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/cbir_model_comparison_grouped_bar.png) (300 DPI) & [`figures/cbir_model_comparison_grouped_bar.svg`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/cbir_model_comparison_grouped_bar.svg) (Lossless Vector)
  * *Purpose*: Multi-metric evaluation (Precision@10, Recall@10, F1-score@10, mAP) across all five models (M1–M5), highlighting proposed model M5 in patterned emerald green.
  * *Status*: **READY** (Single-column IEEE 3.5 in, Times New Roman typography, 300 DPI).

* **Figure 3: MLP-Predicted Weight Distributions**
  * *Source*: [`figures/fig3_weight_distributions.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/fig3_weight_distributions.png)
  * *Purpose*: Demonstrates the dynamic range and density of weights ($w_{\text{colour}}$ and $w_{\text{texture}}$) across the query set.
  * *Status*: **READY** (300 DPI, Seaborn KDE plot).

* **Figure 4: Category-Wise Mean Weights**
  * *Source*: [`figures/fig4_category_weights.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/fig4_category_weights.png)
  * *Purpose*: Shows average weight assignments per category, highlighting that the model automatically adapts to texture-centric (e.g., flowers) and colour-centric (e.g., mountains) queries.
  * *Status*: **READY** (300 DPI, rotated labels).

* **Figure 5: M3 vs. M5 Qualitative Search Grid**
  * *Source*: [`figures/fig5_query_m3_vs_m5_beaches.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/fig5_query_m3_vs_m5_beaches.png)
  * *Purpose*: Compares Equal Fusion (M3) and Adaptive Fusion (M5) side-by-side on a sample query, demonstrating how M5 corrects baseline retrieval errors.
  * *Status*: **READY** (Includes relevance borders and weight tags).

---

## 2. Supplementary Figures

| Figure / Plot | File Path | Format | Role Recommendation |
| :--- | :--- | :---: | :--- |
| **Annotated Grouped Bar Chart** | [`figures/cbir_model_comparison_annotated.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/cbir_model_comparison_annotated.png) | PNG (300 DPI) & SVG | **SUPPLEMENTARY MATERIAL** (includes M5 peak text callouts) |
| **mAP-Only Bar Chart** | [`figures/fig2_model_performance_comparison.png`](file:///c:/Users/777Pr/Downloads/Research%20implementation/figures/fig2_model_performance_comparison.png) | PNG (300 DPI) | **SUPPLEMENTARY MATERIAL** |

---

## 3. IEEE Captions

Below are the formatted captions matching the IEEE style manual:

* **Fig. 1.** Block diagram of the proposed query-adaptive colour-texture fusion CBIR framework.
* **Fig. 2.** Quantitative retrieval performance comparison across the five CBIR models (M1–M5) evaluated on the Corel-1K benchmark across Precision@10, Recall@10, F1-score@10, and mean Average Precision (mAP). Model M5 (patterned green) represents the proposed adaptive weighting method.
* **Fig. 3.** Probability density distributions of the predicted colour and texture weights ($w_{\text{colour}}$ and $w_{\text{texture}}$) across all evaluation queries.
* **Fig. 4.** Average stacked colour and texture weights across the ten semantic categories of the Corel-1K dataset.
* **Fig. 5.** Top-10 retrieved gallery images for a sample 'beaches' query comparing equal weight fusion (M3) and the proposed adaptive weight model (M5). Relevance borders highlight correct (green) and incorrect (red) matches.

---

## 4. Table Inventory

### Table I: Dataset and Experimental Configuration
* **Description**: Outlines the dataset partition and baseline variables (split, seed, query size, etc.).
* **Source**: [`config.py`](file:///c:/Users/777Pr/Downloads/Research%20implementation/config.py)

### Table II: Feature Descriptors
* **Description**: Lists dimensions and parameters for HSV moments (9-dim) and GLCM texture features (4-dim).
* **Source**: [`features/feature_extractor.py`](file:///c:/Users/777Pr/Downloads/Research%20implementation/features/feature_extractor.py)

### Table III: M1–M5 Retrieval Performance Comparison
* **Description**: Complete accuracy metrics (Precision@10, Recall@10, F1-score@10, mAP) and retrieval latency.
* **Source**: [`main.py`](file:///c:/Users/777Pr/Downloads/Research%20implementation/main.py) / [`output/part2/part2_results.pkl`](file:///c:/Users/777Pr/Downloads/Research%20implementation/output/part2/part2_results.pkl)

| Model | Descriptor / Strategy | Precision@10 | Recall@10 | F1-score@10 | mAP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **M1** | HSV Color Moments (9-dim) | 0.6200 | 0.0689 | 0.1240 | 0.4048 |
| **M2** | GLCM Texture (4-dim) | 0.4680 | 0.0520 | 0.0936 | 0.3508 |
| **M3** | Equal Fusion ($w_c=0.5, w_t=0.5$) | 0.6990 | 0.0777 | 0.1398 | 0.4652 |
| **M4** | Fixed Fusion ($w_c=0.7, w_t=0.3$) | 0.6730 | 0.0748 | 0.1346 | 0.4340 |
| **M5** | **Proposed AI Adaptive Weighting** | **0.7200** | **0.0800** | **0.1440** | **0.4845** |

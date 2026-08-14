# Corel-1K CBIR Pipeline Experiment Report (Part 1)

This report details the execution and baseline evaluation of the Content-Based Image Retrieval (CBIR) pipeline on the Corel-1K dataset.

## Experimental Setup

- **Dataset Path**: `C:\Users\777Pr\Downloads\archive (1)\dataset`
- **Random Seed**: `42`
- **Query/Gallery Split**: 10 query images per category (100 total) and 90 gallery images per category (900 total)
- **Features Extracted**:
  - **HSV Color (9-dim)**: Channel mean, standard deviation, and skewness for H, S, V.
  - **GLCM Texture (4-dim)**: Contrast, correlation, energy, and homogeneity averaged over 4 angles (0, 45, 90, 135 degrees) with distance = 1.
- **Normalization**: Min-Max normalization fitted strictly on the 900 gallery features.
- **Fusion Weighting**:
  - **M3 (Equal Fusion)**: Colour Weight = 0.5, Texture Weight = 0.5
  - **M4 (Fixed Fusion)**: Colour Weight = 0.7, Texture Weight = 0.3 (Fixed baseline, no claims of optimality)
- **Evaluation**: Evaluated at K = 10 for Precision, Recall, and F1. Mean Average Precision (mAP) computed over the complete 900-image gallery.

## Evaluation Results

| Model | Mean Precision@10 | Mean Recall@10 | Mean F1-score@10 | mAP (Full Gallery) | Avg Retrieval Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1** | 0.6200 | 0.0689 | 0.1240 | 0.4048 | 0.14 ms |
| **M2** | 0.4680 | 0.0520 | 0.0936 | 0.3508 | 0.11 ms |
| **M3** | 0.6990 | 0.0777 | 0.1398 | 0.4652 | 0.12 ms |
| **M4** | 0.6730 | 0.0748 | 0.1346 | 0.4340 | 0.12 ms |

## Analysis of Baseline Performance

- **M1 (Colour Only)**: Performs feature retrieval focusing purely on color distributions in HSV space.
- **M2 (Texture Only)**: Performs feature retrieval focusing on spatial gray-level variations.
- **M3 (Equal Fusion)**: Fuses both color and texture with equal weights, balancing both descriptors.
- **M4 (Fixed Weight Fusion)**: Combines representations using a fixed weighting scheme of 0.7 colour and 0.3 texture.

## Visual Artifacts

The following comparison plots have been generated and saved to the `output/plots/` directory:
- [Precision comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/precision_comparison.png)
- [Recall comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/recall_comparison.png)
- [F1-score comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/f1_comparison.png)
- [mAP comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/map_comparison.png)
- [Retrieval-time comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/time_comparison.png)

Additionally, visual retrieval examples showing M1-M4 side-by-side rankings for queries in different categories have been saved:
- [Beaches Query Comparison](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/query_beaches.png)
- [Dinosaurs Query Comparison](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/query_dinosaurs.png)
- [Flowers Query Comparison](file:///C:/Users/777Pr/Downloads/Research implementation/output/plots/query_flowers.png)

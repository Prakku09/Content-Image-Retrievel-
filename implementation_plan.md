# CBIR Project Cleanup and Reorganization Plan

This document provides a comprehensive audit (STEP 1) and source-of-truth determination (STEP 2) for reorganizing the repository without breaking any functionality.

---

## STEP 1: Complete Project File Audit

| Filename / Path | Purpose & Description | Used By / Caller | Recommended Action |
| :--- | :--- | :--- | :---: |
| **`config.py`** | Primary configuration (dataset path, categories, seeds, GLCM params, M3/M4 weights). | All modules (`dataset_loader`, `feature_extractor`, `retrieval`, `visualization`, etc.) | **KEEP & MERGE** (Merge `config_part2.py` into this) |
| **`config_part2.py`** | Part 2 MLP training and output path parameters. | `mlp_trainer.py`, `run_part2_experiment.py`, `visualize_part2.py` | **MERGE & DELETE** (Merged into `config.py`) |
| **`dataset_loader.py`** | Loads Corel-1K dataset, category filtering, deterministic query/gallery 100/900 split. | `run_part1_experiment.py`, `validate_dataset.py`, `sanity_check.py` | **KEEP** (Move to `data/dataset_loader.py`) |
| **`validate_dataset.py`** | Validates external dataset structure, file integrity, and per-class image counts. | Standalone CLI tool | **KEEP** (Move to `data/validate_dataset.py`) |
| **`feature_extractor.py`** | Extracts 9-dim HSV color moments and 4-dim GLCM texture features. | `run_part1_experiment.py`, `sanity_check.py` | **KEEP** (Move to `features/feature_extractor.py`) |
| **`normalization.py`** | `MinMaxNormalizer` class for fitting on gallery and transforming query/gallery features. | `run_part1_experiment.py`, `sanity_check.py` | **KEEP** (Move to `features/normalization.py`) |
| **`mlp_model.py`** | PyTorch `AdaptiveWeightMLP` architecture (13 $\to$ 16 $\to$ 8 $\to$ 2 Softmax). | `mlp_trainer.py`, `run_part2_experiment.py`, `sanity_check.py` | **KEEP** (Move to `model/mlp_model.py`) |
| **`mlp_trainer.py`** | PyTorch Triplet Margin Loss training loop, triplet dataset loader, validation & checkpointing. | `run_part2_experiment.py` | **KEEP** (Move to `model/mlp_trainer.py`) |
| **`retrieval.py`** | Concatenates feature representations for M1–M4 and performs Euclidean distance image ranking. | `run_part1_experiment.py`, `sanity_check.py` | **KEEP** (Move to `retrieval/retrieval.py`) |
| **`evaluation.py`** | Computes Precision@K, Recall@K, F1-score@K, and full-gallery mean Average Precision (mAP). | `run_part1_experiment.py`, `run_part2_experiment.py` | **KEEP** (Move to `retrieval/evaluation.py`) |
| **`visualization.py`** | Baseline plotting routines (M1–M4), metric summary tables, query retrieval grids. | `run_part1_experiment.py`, `run_part2_experiment.py` | **KEEP** (Move to `visualization/visualization.py`) |
| **`visualize_part2.py`** | M1–M5 metric comparison charts, weight KDE distribution, category stacked bar, M3 vs M5 grid. | `run_part2_experiment.py` | **KEEP** (Move to `visualization/visualize_part2.py`) |
| **`generate_grouped_bar_chart.py`** | Dedicated IEEE publication grouped bar chart generator (PNG 300 DPI + SVG). | Standalone generator | **KEEP** (Move to `visualization/generate_grouped_bar_chart.py`) |
| **`run_part1_experiment.py`** | Part 1 pipeline runner (extracts features, normalizes, evaluates M1–M4, generates report). | Top-level script | **MERGE** (Combine with Part 2 into unified `main.py`) |
| **`run_part2_experiment.py`** | Part 2 pipeline runner (trains MLP, generates weights, evaluates M5, outputs comparisons). | Top-level script | **MERGE** (Combine with Part 1 into unified `main.py`) |
| **`run_sanity_check.py`** | Sanity test for Part 1 (shapes, bounds, normalization, distance ranking). | Standalone test | **MERGE** (Combine with Part 2 sanity check into `sanity_check.py`) |
| **`run_part2_sanity_check.py`** | Sanity test for Part 2 (MLP shapes, non-negativity, sum to 1.0, adaptivity). | Standalone test | **MERGE & DELETE** (Merged into `sanity_check.py`) |
| **`render.html`** | Temporary HTML/SVG rendering scaffold used to draw Figure 1 block diagram. | `render_diagram.ps1` | **DELETE** (Scaffolding no longer needed) |
| **`render_300dpi.html`** | High-res HTML/SVG rendering scaffold variant. | `render_diagram.ps1` | **DELETE** (Scaffolding no longer needed) |
| **`render_diagram.ps1`** | PowerShell script automating WebBrowser screenshot rendering for Figure 1. | Standalone script | **DELETE** (Scaffolding no longer needed) |
| **`fig1_pipeline_diagram.svg`** | Lossless vector source diagram for Figure 1. | Paper asset | **KEEP** (Move to `figures/fig1_pipeline_diagram.svg`) |
| **`fig1_pipeline_diagram_300dpi.png`** | 300 DPI high-resolution raster export of Figure 1. | Paper asset | **KEEP** (Move to `figures/fig1_pipeline_diagram.png`) |
| **`fig1_pipeline_diagram.png`** (160 KB) | Low-resolution draft raster export of Figure 1. | Intermediate output | **DELETE** (Superseded by 300 DPI version) |
| **`cbir_model_comparison_grouped_bar.png`** | 300 DPI publication grouped bar chart (M1–M5 across 4 metrics). | Paper asset | **KEEP** (Move to `figures/cbir_model_comparison_grouped_bar.png`) |
| **`cbir_model_comparison_grouped_bar.svg`** | Vector format of publication grouped bar chart. | Paper asset | **KEEP** (Move to `figures/cbir_model_comparison_grouped_bar.svg`) |
| **`cbir_model_comparison_annotated.png`** | 300 DPI grouped bar chart with M5 value callouts. | Supplementary asset | **KEEP** (Move to `figures/cbir_model_comparison_annotated.png`) |
| **`cbir_model_comparison_annotated.svg`** | Vector format of annotated grouped bar chart. | Supplementary asset | **KEEP** (Move to `figures/cbir_model_comparison_annotated.svg`) |
| **`cbir_model_comparison_ieee_clean.png`** | Redundant draft generated during legend layout testing. | Intermediate draft | **DELETE** (Superseded by grouped_bar) |
| **`cbir_model_comparison_ieee_clean.svg`** | Redundant vector draft generated during legend layout testing. | Intermediate draft | **DELETE** (Superseded by grouped_bar) |
| **`output/plots/*`** | Part 1-only comparison plots (M1–M4 only). | Intermediate output | **DELETE** (Superseded by Part 2 M1–M5 plots) |
| **`output/part2/plots/*`** | Part 2 complete plots (`map_comparison.png`, `weight_distributions.png`, etc.). | Verified paper assets | **KEEP** (Key figures copied to `/figures`) |
| **`output/corel1k_features.pkl`** | Pre-extracted 13-dim normalized feature cache (HSV + GLCM). | `run_part2_experiment.py`, `mlp_trainer.py` | **KEEP** (In `output/corel1k_features.pkl`) |
| **`output/part2/adaptive_weight_mlp.pt`** | Trained PyTorch checkpoint for M5. | `run_part2_experiment.py` | **KEEP** (In `output/part2/` or `model/`) |
| **`output/part2/part2_results.pkl`** | Part 2 metric results, training loss curve, weight statistics. | `visualize_part2.py` | **KEEP** |
| **`output/*_report.md`** | Part 1 and Part 2 markdown experiment summary reports. | Documentation | **KEEP** |
| **`FINAL_PAPER_FIGURE_AND_TABLE_INVENTORY.md`** | Comprehensive IEEE manuscript figure, table, and caption guide. | Documentation | **KEEP** (Update paths to `/figures`) |
| **`README.md`** | Project documentation and instructions. | Documentation | **KEEP** (Update paths & structure) |

---

## STEP 2: Source of Truth Confirmation

### 1. Model Comparison Grouped Bar Charts
- **Candidates**:
  - `cbir_model_comparison_grouped_bar.png` / `.svg`
  - `cbir_model_comparison_annotated.png` / `.svg`
  - `cbir_model_comparison_ieee_clean.png` / `.svg`
- **Source of Truth**: 
  - **`cbir_model_comparison_grouped_bar.*`** is the **final source of truth** for the clean IEEE single-column grouped bar chart (no bar occlusion, 3-column top legend, 300 DPI, emerald hatch for M5).
  - `cbir_model_comparison_annotated.*` is kept as a dedicated annotated variant.
  - `cbir_model_comparison_ieee_clean.*` is an intermediate draft and will be **DELETED**.

### 2. Figure 1: Pipeline Architecture Diagram
- **Candidates**:
  - `fig1_pipeline_diagram.svg` (vector source)
  - `fig1_pipeline_diagram_300dpi.png` (high-res 300 DPI raster)
  - `fig1_pipeline_diagram.png` (160 KB low-res draft)
  - `render.html`, `render_300dpi.html`, `render_diagram.ps1` (rendering scaffolding)
- **Source of Truth**:
  - **`fig1_pipeline_diagram.svg`** is the master vector source.
  - **`fig1_pipeline_diagram_300dpi.png`** is the master 300 DPI raster (will be placed as `figures/fig1_pipeline_diagram.png`).
  - The 160 KB low-res draft and the HTML/PowerShell rendering scaffolding will be **DELETED**.

### 3. Metric Comparison Plots (Part 1 vs. Part 2)
- **Candidates**:
  - `output/plots/*` (Part 1 plots: M1–M4 only)
  - `output/part2/plots/*` (Part 2 plots: complete M1–M5 comparisons)
- **Source of Truth**:
  - **`output/part2/plots/*`** is the **final source of truth** because it contains all 5 models (M1–M5), the peak M5 performance (0.4845 mAP), weight distributions, and M3 vs. M5 search grids.
  - `output/plots/*` contains obsolete M1–M4-only graphs and will be **DELETED**.

### 4. Configuration Settings (`config.py` vs `config_part2.py`)
- **Source of Truth**:
  - `config.py` handles dataset paths, categories, GLCM params, seeds, and evaluation metrics ($K=10$).
  - `config_part2.py` adds MLP hyperparameters (learning rate = 0.001, margin = 0.15, batch size = 64, epochs = 100).
  - Both are completely compatible and will be **MERGED** into a single, clean `config.py`.

---

## STEP 3: Proposed Reorganized Folder Structure

```
Research implementation/
├── config.py                           # Single unified configuration source of truth
├── main.py                             # Unified top-level entry point (runs full pipeline or individual stages)
├── sanity_check.py                     # Unified sanity test suite (data, features, MLP, retrieval)
├── README.md                           # Updated repository documentation
├── FINAL_PAPER_FIGURE_AND_TABLE_INVENTORY.md  # Final paper figure and table mapping
│
├── data/
│   ├── dataset_loader.py               # Corel-1K loading & deterministic 100/900 train/val splits
│   └── validate_dataset.py             # Dataset verification tool
│
├── features/
│   ├── feature_extractor.py            # HSV color moments (9-dim) & GLCM texture (4-dim)
│   └── normalization.py                # MinMaxNormalizer (fitted on gallery)
│
├── model/
│   ├── mlp_model.py                    # PyTorch AdaptiveWeightMLP architecture
│   └── mlp_trainer.py                  # Metric learning (triplet loss) training loop
│
├── retrieval/
│   ├── retrieval.py                    # Feature concatenation & Euclidean distance retrieval
│   └── evaluation.py                   # Precision@10, Recall@10, F1-score@10, mAP metrics
│
├── visualization/
│   ├── visualization.py                # Retrieval grid and baseline plotting
│   ├── visualize_part2.py              # M1-M5 comparisons, weight KDEs, category weights
│   └── generate_grouped_bar_chart.py   # Publication-grade grouped bar chart generator
│
├── figures/                            # Clean folder for all final IEEE publication figures
│   ├── fig1_pipeline_diagram.svg       # Figure 1: Pipeline Architecture (Vector)
│   ├── fig1_pipeline_diagram.png       # Figure 1: Pipeline Architecture (300 DPI)
│   ├── fig2_model_performance_comparison.png # Figure 2: mAP Comparison (300 DPI)
│   ├── fig3_weight_distributions.png   # Figure 3: Weight Distributions KDE (300 DPI)
│   ├── fig4_category_weights.png       # Figure 4: Category-Wise Mean Weights (300 DPI)
│   ├── fig5_query_m3_vs_m5_beaches.png # Figure 5: M3 vs M5 Qualitative Search Grid (300 DPI)
│   ├── cbir_model_comparison_grouped_bar.png # Grouped Bar Chart (300 DPI)
│   ├── cbir_model_comparison_grouped_bar.svg # Grouped Bar Chart (Vector SVG)
│   ├── cbir_model_comparison_annotated.png   # Annotated Grouped Bar Chart (300 DPI)
│   └── cbir_model_comparison_annotated.svg   # Annotated Grouped Bar Chart (Vector SVG)
│
└── output/                             # Generated execution artifacts
    ├── corel1k_features.pkl            # Pre-extracted 13-dim feature matrix cache
    ├── part1_experiment_report.md
    └── part2/
        ├── adaptive_weight_mlp.pt      # Trained PyTorch checkpoint
        ├── part2_results.pkl           # Metric results and loss history
        ├── part2_experiment_report.md
        └── plots/                      # Raw experiment run plot cache
```

---

## STEP 4: Verification Plan

1. **Import and Path Integrity**:
   - Update all `sys.path` and package imports across submodules to ensure relative and top-level imports resolve smoothly.
2. **Execute Sanity Suite**:
   - Run `python sanity_check.py` to verify dataset loading, feature dimension extraction (9+4=13), MinMax normalization, MLP shape/adaptivity, and distance ranking.
3. **Execute Figure Generator**:
   - Run `python visualization/generate_grouped_bar_chart.py` to verify PNG and SVG generation into `figures/`.
4. **Execute Full Pipeline / Part 2 Evaluation**:
   - Run `python main.py` to verify end-to-end execution, producing the exact verified metrics:
     - M1: Precision=0.6200, mAP=0.4048
     - M2: Precision=0.4680, mAP=0.3508
     - M3: Precision=0.6990, mAP=0.4652
     - M4: Precision=0.6730, mAP=0.4340
     - M5: Precision=0.7200, mAP=0.4845

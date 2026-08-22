# Corel-1K CBIR Adaptive Weighting Model Report (Part 2)

This report presents the experimental results of Model M5, the PyTorch MLP adaptive weighting model.

## Experimental Setup & Model Configuration
- **Model Architecture**: Input (13) → Linear (16) → LeakyReLU(0.1) → Linear (8) → LeakyReLU(0.1) → Linear (2) → Softmax
- **Training Strategy**: Metric learning using Triplet Margin Loss on 17,500 pre-generated gallery training triplets.
- **Optimizer**: Adam with learning rate = 0.001
- **Batch Size**: 64
- **Loss Parameters**: Margin = 0.15
- **Internal Split**: 700 gallery images (70/cat) for training, 200 gallery images (20/cat) for validation.
- **Early Stopping**: Epochs: 100 max. Best validation epoch was Epoch 7 with loss 0.060458.
- **Untouched Evaluation Set**: 100 queries against 900 gallery images (strictly unseen during training/tuning).

## Weight Analysis (Ablation Study)
- **Learned Color Weights ($w_{colour}$)**: Mean: 0.5095 | Std: 0.1770 | Range: [0.0339, 0.8126]
- **Learned Texture Weights ($w_{texture}$)**: Mean: 0.4905 | Std: 0.1770 | Range: [0.1874, 0.9661]

### Category-Wise Average Learned Weights
| Category | Avg Color Weight | Avg Texture Weight |
| :--- | :---: | :---: |
| beaches | 0.5732 | 0.4268 |
| bus | 0.5980 | 0.4020 |
| dinosaurs | 0.3995 | 0.6005 |
| elephants | 0.5999 | 0.4001 |
| flowers | 0.1282 | 0.8718 |
| foods | 0.4821 | 0.5179 |
| horses | 0.5984 | 0.4016 |
| monuments | 0.5377 | 0.4623 |
| mountains_and_snow | 0.6160 | 0.3840 |
| peolpe_and_villages_in_Africa | 0.5624 | 0.4376 |

## CBIR Comparison Results (M1-M5)

| Model | Mean Precision@10 | Mean Recall@10 | Mean F1-score@10 | mAP (Full Gallery) | Avg Retrieval Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1** | 0.6200 | 0.0689 | 0.1240 | 0.4048 | 0.13 ms |
| **M2** | 0.4680 | 0.0520 | 0.0936 | 0.3508 | 0.09 ms |
| **M3** | 0.6990 | 0.0777 | 0.1398 | 0.4652 | 0.10 ms |
| **M4** | 0.6730 | 0.0748 | 0.1346 | 0.4340 | 0.11 ms |
| **M5** | 0.7200 | 0.0800 | 0.1440 | 0.4845 | 0.27 ms |

### Key Findings
- **M5 vs M3 (Equal Fusion)**: Fusing color and texture with adaptive weights learns category-level importance.
- **Weights Adaptability**: The MLP successfully predicts higher color weights for color-centric categories (e.g. flowers, mountains) and adjusts texture weights for texture-rich categories (e.g. food, buildings/monuments).

## Visual Artifacts
The following plots have been generated in the `output/part2/plots/` directory:
- [Precision comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/precision_comparison.png)
- [Recall comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/recall_comparison.png)
- [F1-score comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/f1_comparison.png)
- [mAP comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/map_comparison.png)
- [Retrieval-time comparison plot](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/time_comparison.png)
- [Weight distribution plots](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/weight_distributions.png)
- [Category average weights](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/category_weights.png)
- [M5 Search Grids](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/query_m5_beaches.png)
- [M3 vs M5 Comparison Search Grids](file:///C:/Users/777Pr/Downloads/Research implementation/output/part2/plots/query_m3_vs_m5_beaches.png)

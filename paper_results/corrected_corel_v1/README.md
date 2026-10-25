# Corrected Corel-1K Results

This package contains the verified corrected reference evaluation and paper-ready assets. Historical experiment outputs remain separate and are not represented here as current results.

## Evaluation protocol

- Corel-1K images: 1,000 across 10 categories.
- Retrieval gallery: 899 images.
- Held-out queries: 100 images.
- MLP training and validation: 700 and 199 gallery images.
- `training_set/horses/761.jpg` was excluded following the visually confirmed near-duplicate audit; `training_set/horses/776.jpg` remains a held-out query.
- Normalization and checkpoint selection used gallery/training and validation partitions, respectively; held-out query labels were not used for model selection.

## Corrected aggregate results

| Model | Precision@10 | Recall@10 | F1@10 | mAP |
|---|---:|---:|---:|---:|
| M1 | 0.6190 | 0.0689 | 0.1240 | 0.404462 |
| M2 | 0.4680 | 0.0521 | 0.0937 | 0.350816 |
| M3 | 0.6990 | 0.0778 | 0.1400 | 0.465036 |
| M4 | 0.6730 | 0.0749 | 0.1348 | 0.433659 |
| M5 | 0.7100 | 0.0790 | 0.1422 | 0.479794 |
| M6 | 0.6960 | 0.0774 | 0.1394 | 0.460536 |

The five-run M5âˆ’M6 mean mAP difference is +0.017592 (sample SD 0.004008), descriptive only. Runs share image content and category results are heterogeneous; this does not establish robust statistical superiority. Oxford-IIIT Pet is a separate frozen-transfer experiment and is not evidence of universal generalization.

## Package contents

- `assets_v4/`: seven corrected figures in PNG and SVG, three CSV tables, and the asset manifest.
- `manuscript/`: replacement manuscript sections, revised-claims list, reconciliation checklist, and final consistency report.
- `validation/corel_reference/`: corrected results, per-query and per-category metrics, split and duplicate audit, runtime breakdown, rankings, validation reports, and run provenance.
- `validation/five_seed_summary/`: descriptive seed summary, category differences, query overlap, and sanitized reproducibility manifest.
- `paper_manifest.json`: repository-relative provenance for this package.

The image datasets, feature caches, and model checkpoint are intentionally not included. The complete manuscript was not present in the source workspace; the consistency report records that manuscript integration remains unverified.

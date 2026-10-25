# Manuscript replacement sections â€” corrected CBIR evaluation

**Status:** This file contains corrected replacement text for the affected abstract, experimental protocol, results, discussion, and conclusion. The complete source manuscript was not present in the workspace; therefore this is not a fully integrated/revised manuscript, and citations and unaffected sections have not been edited.

## Abstract replacement

We evaluate a query-adaptive content-based image retrieval method that combines handcrafted colour and texture descriptors using an MLP-predicted pair of weights. The corrected Corel-1K experiment uses 899 gallery images and 100 held-out queries after excluding one gallery image identified as a near-duplicate of a retained query. The MLP uses 700 training and 199 validation images, and normalization is fitted on the corrected gallery. On this split, the adaptive model (M5) obtains mAP 0.479794, compared with 0.460536 for the fixed mean-weight control (M6). Across five prespecified duplicate-group-disjoint runs, the descriptive mean seed-level M5-minus-M6 mAP difference is +0.017592 (sample SD 0.004008). These runs share images and gallery content, and category results are heterogeneous; this evidence does not establish robust statistical superiority. A separate frozen transfer evaluation on Oxford-IIIT Pet is reported as dataset-specific evidence and does not establish universal generalization.

## Experimental protocol replacement

The Corel-1K data were audited for exact duplicates using SHA-256 and for near-duplicates using DCT perceptual hashes. The audit found no exact duplicate pairs and one near-duplicate candidate at Hamming distance 4: `training_set/horses/761.jpg` and `training_set/horses/776.jpg`. Visual review confirmed that the pair shares the same scene and pose. The gallery-side image `training_set/horses/761.jpg` was excluded from retrieval, normalization fitting, MLP training, and validation; `training_set/horses/776.jpg` was retained as a held-out query. The corrected split contains 899 gallery images and 100 queries. The gallery was divided into 700 MLP training images and 199 validation images. Held-out queries and their identified near-duplicate were excluded from model training and validation, checkpoint selection, and normalization fitting.

The original M1-M6 definitions were retained. M1 and M2 use colour-only and texture-only Euclidean distances. M3 and M4 retain feature-level weighted concatenation, inducing squared-distance coefficients \(w_c^2\) and \(w_t^2\). M5 uses query-specific MLP weights with the existing direct distance-level formula; M6 applies the mean pair predicted for the 100 queries by that run's M5 checkpoint. The corrected experiment was independently recomputed from saved rankings; maximum absolute disagreement for per-query AP, Precision@10, Recall@10, F1@10, and aggregate metrics was 0.

## Corrected Corel-1K results replacement

Table III reports the corrected Corel-1K retrieval results. Precision, recall, and F1 are measured at rank 10; AP is calculated over the complete 899-image gallery and mAP is the mean AP over 100 held-out queries. Timing is distance calculation plus ranking/sorting per query; feature extraction and MLP inference are reported separately and excluded from the ranking-time column.

| Model | Precision@10 | Recall@10 | F1@10 | mAP | Mean ranking/sort (ms/query) |
|---|---:|---:|---:|---:|---:|
| M1 | 0.6190 | 0.0689 | 0.1240 | 0.404462 | 0.082 |
| M2 | 0.4680 | 0.0521 | 0.0937 | 0.350816 | 0.073 |
| M3 | 0.6990 | 0.0778 | 0.1400 | 0.465036 | 0.079 |
| M4 | 0.6730 | 0.0749 | 0.1348 | 0.433659 | 0.077 |
| M5 | 0.7100 | 0.0790 | 0.1422 | 0.479794 | 0.167 |
| M6 | 0.6960 | 0.0774 | 0.1394 | 0.460536 | 0.153 |

For the corrected fixed query/gallery split, mean paired AP difference M5âˆ’M6 was +0.019258 (median +0.000214; 53 positive, 47 negative, and 0 tied query differences). These are descriptive paired-query results conditional on one shared gallery. No Corel significance test is reported here, and no statistical superiority is inferred from the observed metric difference.

## Prespecified multi-seed descriptive result replacement

Five prespecified query/MLP seed pairs (42, 123, 2025, 31415, 271828) were evaluated using duplicate-group-disjoint query/gallery and training/validation assignments. Each seed used 900 gallery images, 100 queries, 700 training images, and 200 validation images. M5âˆ’M6 mAP differences were positive in all five runs. Their mean was **+0.017592** and sample standard deviation was **0.004008**. This is a seed-level descriptive summary, not an inferential estimate: the runs reuse dataset identities, share gallery content, and their 500 query observations comprise only 399 unique images.

The result varies by category. Across seed-level category means, flowers showed a positive difference in all five runs (mean +0.179664 AP), while foods (mean âˆ’0.005707) and people/villages in Africa (mean âˆ’0.003361) were negative in all five. Other categories were mixed. The aggregate difference is therefore category-dependent and strongly concentrated in flowers; difficult categories and queries have not been removed.

## Oxford-IIIT Pet transfer replacement

Oxford-IIIT Pet is a separate frozen Corel-to-Oxford transfer experiment, using the official trainval/test partition, breed labels for relevance, the corrected Corel checkpoint and Corel normalization bounds. After removing 24 visually confirmed gallery-side duplicate copies, the evaluation used 3,656 gallery images and 3,669 queries across 37 breeds. Corrected transfer mAP was 0.033882 for M5, 0.033683 for M6, and 0.033829 for M7. M5 was lower than M6 and M7 on Precision@10, Recall@10, and F1@10. The paired M5âˆ’M6 mean AP difference was +0.000199 (category-stratified 95% bootstrap CI [0.000129, 0.000271], sign-flip permutation p=0.000010; secondary M5âˆ’M7 Holm-adjusted p=0.172488). These findings are conditional on this fixed dataset split and frozen checkpoint; they are not evidence of universal generalization.

| Model | Precision@10 | Recall@10 | F1@10 | mAP |
|---|---:|---:|---:|---:|
| M5 | 0.043990 | 0.004475 | 0.008122 | 0.033882 |
| M6 | 0.044481 | 0.004522 | 0.008209 | 0.033683 |
| M7 | 0.044699 | 0.004545 | 0.008251 | 0.033829 |

## Discussion and conclusion replacement

On the corrected Corel-1K reference split, M5 has a higher observed mAP than its M6 fixed mean-weight control. The five prespecified duplicate-group-disjoint runs also show higher M5 mAP descriptively, but the runs are not independent replications because they overlap in query identities and share images in their galleries. Category results show that improvements are not uniform and are strongly concentrated in flowers; some categories favor M6. The Oxford-IIIT Pet experiment shows a very small positive M5âˆ’M6 mean AP difference but lower M5 top-10 metrics, and it does not demonstrate broad transfer superiority.

Accordingly, the results support the limited statement that M5 achieved higher observed Corel mAP than M6 under the corrected evaluation and in the five reported descriptive runs. They do not support a claim of robust statistical superiority, category-consistent improvement, or universal generalization. The historical 900-gallery Corel values are superseded for the corrected protocol and must not be presented as leakage-free results.

## Table and figure source data

- Corrected table: [`table_III_corrected_metrics.csv`](assets_v4/table_III_corrected_metrics.csv).
- Corrected figures: [`assets_v4/`](assets_v4/).
- Corrected run, split and validation: [`run_20261025T130024Z`](../corrected_corel_v1/run_20261025T130024Z/).
- Five-run comparisons and provenance: [`summary_v1`](../corrected_corel_robustness_v1/summary_v1/).
- Oxford transfer report: [`transfer_report.md`](../external_transfer_oxford_pet_corrected_v1/run_20261025T131708149568Z/transfer_report.md).

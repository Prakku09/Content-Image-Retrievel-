# Corrected Corel-1K M1-M6 Evaluation and Validation

Run: `run_20261025T130024Z`. This is a newly computed corrected run. Historical outputs were not modified. The earlier partial attempt `../run_20261025T125909Z/` is marked incomplete and must not be used as a result.

## Duplicate audit and corrected split

- Scanned all 1,000 Corel images by SHA-256: **0 exact duplicate pairs**.
- Computed DCT perceptual hashes for all images. At Hamming distance <= 4, one candidate pair was found: `training_set/horses/761.jpg` (gallery) and `training_set/horses/776.jpg` (held-out query), distance 4. Full-resolution visual review confirmed the same scene and pose.
- Retained the query; excluded the gallery copy from the retrieval gallery, normalization reference, training, and validation. No other thresholded perceptual-hash candidate remained.
- Corrected protocol: **899 gallery, 100 held-out queries; 700 MLP train, 199 validation**. Train and validation are disjoint and cover all remaining gallery images. No held-out query or identified near-duplicate enters training, validation, normalization fitting, or checkpoint selection.
- Query split seed: 42; MLP/triplet sampling seed: 42.
- Explicit paths, category labels, assignments, duplicate groups, exclusions, and seeds are in `split_manifest.csv`.

## Corrected M1-M6 results

| Model | Precision@10 | Recall@10 | F1@10 | mAP | Mean ranking/sort (ms/query) |
|---|---:|---:|---:|---:|---:|
| M1 | 0.6190 | 0.0689 | 0.1240 | 0.4045 | 0.082 |
| M2 | 0.4680 | 0.0521 | 0.0937 | 0.3508 | 0.073 |
| M3 | 0.6990 | 0.0778 | 0.1400 | 0.4650 | 0.079 |
| M4 | 0.6730 | 0.0749 | 0.1348 | 0.4337 | 0.077 |
| M5 | 0.7100 | 0.0790 | 0.1422 | 0.4798 | 0.167 |
| M6 | 0.6960 | 0.0774 | 0.1394 | 0.4605 | 0.153 |

M5-M6 per-query AP differences: mean **0.019258**, median **0.000214**, sample SD **0.076450**; 53 positive, 47 negative, 0 tied. Category means favor M5 for 5/10 categories and M6 for 5/10; see `category_m5_minus_m6.csv` and `per_category_metrics.csv`. Flowers has a category mean difference of **+0.210336**; the other 90 queries together average **-0.001973**. Thus the aggregate positive difference is concentrated in flowers, rather than consistent across categories. These are descriptive results for one query/gallery split. No inferential test or superiority claim is made.

M6 uses the **new checkpoint's** mean query weights (0.578404 colour, 0.421596 texture). M3/M4 feature-level fusion and M5/M6 distance-level fusion definitions were retained unchanged.

## Training, normalization, cache, and runtime

- The existing 13->16->8->2 MLP, Adam optimizer, triplet-loss objective, and configured hyperparameters were unchanged. The best validation checkpoint was restored at epoch 8; best validation loss was 0.06175187. The trainer already used `copy.deepcopy(model.state_dict())` on validation improvement and restored that state, so no checkpoint-copy change was needed.
- Min-max bounds were fitted on the corrected 899-image gallery only. Held-out queries and the excluded duplicate were not used for normalization.
- Run-specific features and fingerprints live within this versioned directory. No historical feature cache or checkpoint was reused. M6 was computed from current M5 predictions.
- Feature extraction total was 44.984 seconds over gallery and queries; mean M5 forward pass was 0.113 ms/query. Mean ranking/sort values are in the table. `runtime_breakdown.csv` records per-query feature extraction, M5 inference, and ranking/sorting. Query latency adds query feature extraction, applicable M5 inference, and ranking; gallery preprocessing and training are excluded. Timings are single-run observations, not benchmark estimates.

## Independent validation and commands

Independent verification checked all six 100x899 rankings, all 600 per-query metric rows, manifest artifact hashes, split/index/path correspondence, scaler bounds, and M6 weight averaging. Maximum AP/Precision@10/Recall@10/F1@10 discrepancy from saved-ranking recomputation: **0.0**. M6 weights match the new M5 prediction mean. Full unit suite: **32 passed**. Python compilation passed; `git diff --check` passed.

Successful full-pipeline command:

```powershell
& 'python' corrected_corel_run.py --dataset-root '<Corel-1K dataset root>' --output-root '<repository-root>\output\corrected_corel_v1'
```

Independent metric/provenance validation:

```powershell
& 'python' validate_corrected_corel_run.py '<repository-root>\output\corrected_corel_v1\run_20261025T130024Z'
```

Unit tests:

```powershell
& 'python' -m unittest discover -s tests -v
```

Compilation and whitespace checks:

```powershell
& 'python' -m py_compile corrected_corel_run.py validate_corrected_corel_run.py model\mlp_trainer.py external_transfer.py retrieval\evaluation.py
git diff --check
```

The full Oxford frozen-transfer command used the corrected checkpoint, corrected scaler, and M5-derived M6 weights:

```powershell
& 'python' external_transfer.py --dataset-root '<Oxford-IIIT Pet dataset root>' --output-root '<repository-root>\output\external_transfer_oxford_pet_corrected_v1' --checkpoint '<repository-root>\output\corrected_corel_v1\run_20261025T130024Z\m5_checkpoint.pt' --scaler '<repository-root>\output\corrected_corel_v1\run_20261025T130024Z\corel_scaler_for_transfer.pkl' --corel-weights '<repository-root>\output\corrected_corel_v1\run_20261025T130024Z\corel_query_weights_for_transfer.csv'
```

It was independently verified with:

```powershell
& 'python' validate_external_run.py --run-dir '<repository-root>\output\external_transfer_oxford_pet_corrected_v1\run_20261025T131708149568Z'
```

## Historical paper values to revise

`paper_table_reconciliation.csv` compares the prior M1-M6 figures with this corrected split. All prior Corel result tables and performance claims based on the 900-image gallery are **superseded as estimates for the corrected protocol**. Preserve them as historical, but replace paper tables/figures with the corrected values and describe the one-image exclusion and corrected 899-image gallery. Do not claim the old run was leakage-free. The Oxford frozen-transfer evaluation has now been rerun with this corrected checkpoint; its results remain conditional on one dataset and checkpoint.

The corrected-checkpoint Oxford-IIIT Pet run reports mAP M5/M6/M7 = **0.033882/0.033683/0.033829**. M5 has lower Precision@10, Recall@10, and F1@10 than M6 and M7. Its primary paired M5-M6 mean AP difference is +0.000199 (category-stratified 95% bootstrap CI [0.000129, 0.000271], sign-flip permutation p=0.000010; secondary M5-M7 Holm-adjusted p=0.172488). The report documents the tests' assumptions and limitations. This is evidence conditional on this one official split and frozen checkpoint, not broad transfer or robust superiority. See the Oxford `transfer_report.md` for full details.

## Limitations

The corrected evaluation uses one fixed random query split and one checkpoint seed. Query APs share one gallery and are category-dependent; paired query differences do not represent independent gallery or training replications. The positive mean M5-M6 difference is descriptive and is concentrated in one category. No statistical significance or robust M5 superiority is claimed.

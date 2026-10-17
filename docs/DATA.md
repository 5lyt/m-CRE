# Data inventory

## Model inputs and supporting results

The four complete input tables are under data/processed/. Each contains one gene ID and eight ordered numerical feature columns; these are the supplied preprocessed modeling inputs. Candidate metadata under data/candidates/ contains the 1,362 requested-gene manifest and the 1,357 complete inputs. The stored pattern object is data/motifs/tfmodisco_results_1362.h5.

results/revised/Data7/ contains source/sample manifests, feature summaries, attribution sampling records, pattern counts and seqlet/gene membership. results/revised/Data5/ contains cross-species prediction metric aggregation records. Supporting Data 1-4 and 6 provide model performance, motif annotation and statistical comparisons.

## Ten-model comparison tables

The Figure 2a/d results are in data/model_metrics/ten_model_controls/. The summary CSV records pooled_oof_auroc, mean_fold_auroc, fold_std and n_folds for M0-M4, M4-T8, M4-R8, M4-P, T8-only and R8-only; two wide CSVs provide task-by-model views. fold_std is the sample SD of chromosome-fold AUROCs.

The historical tables were produced with DeepCRE_review_controls_v3. The separately supplied rerun implementation uses code/reproducibility/prepare_task_arrays.py, code/data_deal/build_control_features.py and code/model/train_comparisons.py. See the table-directory README for model definitions and provenance; numerical equivalence to historical runs is not established by the presence of a new runner.

## Separate model and motif-input assets

| Asset | Contents | Restore |
| --- | --- | --- |
| models_01_05.zip | Maize-leaf chromosome checkpoints 1-5 | Unzip both model ZIPs into one parent |
| models_06_10.zip | Maize-leaf chromosome checkpoints 6-10 | Preserves Zmay_leaf/ paths |
| motif_inputs_part1.zip | First 679 input rows, gene IDs and row indices | Unzip both parts into one parent; run analysis/restore_motif_arrays.py |
| motif_inputs_part2.zip | Remaining 678 aligned rows | Restores ohe_1362.npz and attr_1362.npz with N=1357 |

Exact asset names, sizes and SHA256 checksums are listed in data/large_asset_manifest.json. Obtain matching assets from the corresponding release/archive file list. The lightweight CSV/HDF5 checks do not require these separate model/array ZIPs. Recompressed NPZ bytes may differ while numerical arrays are identical.

## External inputs and versions

Reference genomes, annotations and raw SRA reads are public-source inputs; species configuration specifies reference filename patterns. Full training additionally requires complete processed expression-label tables and the validation whitelist. Example labels in this repository are partial examples. Exact checkpoint replay requires the preprocessing and cohort states described in REPRODUCIBILITY.md.

KEGG query dates and annotation versions are recorded with the enrichment results. Gene-ID mappings, effective backgrounds and software session records are retained for interpretation and reproducibility. SOURCE_FILES_SHA256.tsv identifies files in this source tree; large-asset checksums are maintained separately.

## Additional checkpoint and training provenance assets

Three additional ZIPs provide supplied M0 maize-leaf models and training state, M1-M3 maize-leaf checkpoints/metrics, and historical control-training records. See RELEASE_ASSETS.md for exact filenames, historical model-name mappings, coverage and restoration paths. Their checksums are in data/supplementary_asset_manifest.json. The M0/control states are model-specific and must not be substituted for the missing original M4 preprocessing state.

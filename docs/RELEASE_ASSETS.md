# Release assets

The source archive contains code, processed common feature tables, summary results, metadata and the stored TF-MoDISco HDF5 object. Model checkpoints and aligned motif inputs are distributed as separate ZIP assets. Check data/large_asset_manifest.json for the four core ZIPs and data/supplementary_asset_manifest.json for the three additional ZIPs. Match both file size and SHA256 before extraction. Repackaging changes ZIP checksums even when model bytes and numerical arrays are unchanged.

## Core checkpoints and inputs

- models_01_05.zip and models_06_10.zip contain the original maize-leaf M4 checkpoints for chromosomes 1-5 and 6-10. Extract both into the same parent; they retain Zmay_leaf/output_DeepCRE_mRNA_BiLSTM_chr_model_chrN.keras paths. All ten checkpoint files match the supplied earlier M4 archive byte for byte. Their headers identify Keras 2.10.0 and HDF5 storage despite the .keras extension. No TensorFlow inference or retraining was performed during packaging.
- motif_inputs_part1.zip contains original input rows 0-678; motif_inputs_part2.zip contains rows 679-1356. Each contains candidate_arrays.npz with sequences, attributions, row_index and gene_ids, plus gene_meta_part.csv. There are 1,357 complete rows from the requested 1,362-gene cohort. Both numerical arrays have shape (1357, 3020, 4), dtype float32 and A/C/G/T channels. Attributions retain positive and negative values.

Extract the two motif ZIPs into the same parent and run:

```bash
python analysis/restore_motif_arrays.py --parts-root /path/to/parts_parent --output-dir /path/to/restored_inputs
python analysis/verify_new_materials.py --array-dir /path/to/restored_inputs --output-dir /path/to/audit_output
```

Partition/reassembly preserved every numerical value and the row/gene order. All 2,576 stored seqlet intervals from 348 genes were matched to the sequence and signed arrays, including reverse complements. These are sequence attributions; their presence does not supply signed eight-feature per-gene attributions.

## Additional model and training records

- M0_Zmay_leaf_training_records.zip contains the supplied M0 maize-leaf models, per-fold scaler files/parameters, balanced/evaluation sample manifests, histories, thresholds and input fingerprint. Its original archive contents are unchanged. Extract into a separate M0 parent directory. M0 is an eight-feature-only model; these scalers do not recover original M4 preprocessing state.
- M1_M2_M3_Zmay_leaf_checkpoints.zip contains ten maize-leaf chromosome checkpoints per model plus original metrics/plots. M1/ is DeepCRE_chr, M2/ is DeepCRE_BiLSTM_chr and M3/ is DeepCRE_mRNA_chr. File bytes from the three supplied source archives are retained.
- control_training_records.zip contains supplied CSV/JSON, feature, cohort, scaler-parameter, prediction and figure records from the historical control pipeline. Three console .log files are omitted; retained source files are unchanged. The incomplete earlier controls_v3/M6_transcript8 directory is retained as an explicitly identified partial record. Complete transcript M6/M7 records are under controls_v3_m6m7. There are no control-model checkpoints in this archive.

Historical names map as follows: transcript M5 = M4-T8, transcript M6 = M4-P, transcript M7 = T8-only, regulatory M5 = M4-R8, regulatory M7 = R8-only. Pooled OOF and mean fold AUROC matched the published ten-model summary for all 40 control model/task rows within 1e-12. This is a check of supplied result records rather than a rerun of model training. The M0 maize-leaf fold mean also matches its supplied summary.

The availability of these M0 and control records does not establish complete original M4 scaler/background state or a verified family-wise benchmark. Scope and required inputs remain in REPRODUCIBILITY.md. Additional model architecture loading and inference require a compatible TensorFlow/Keras environment and were not validated during this packaging step.

## Integrity and availability

provenance/release_asset_provenance.json records checkpoint hashes, array validation and control-summary comparisons. Release-level SHA256SUMS.txt additionally identifies the exact source ZIP and all seven separate ZIP assets. Download from the corresponding GitHub Release or explicitly linked data archive. A GitHub Release attachment is not by itself evidence that the file is present in its Zenodo software record; consult the actual archival file inventory.

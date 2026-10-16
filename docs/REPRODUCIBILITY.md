# Reproduction instructions

Run commands from the repository root. `requirements-audit.txt` is sufficient for CSV/HDF5 verification. Training dependencies are listed separately; the original exact environment has not been recovered. The supplied checkpoint headers report Keras 2.10.0 with the TensorFlow backend, and the .keras-named files are HDF5. Use a compatible Python 3.10/Keras 2.x environment for checkpoint work; the header alone does not establish every package version.

## Inspect existing results

```bash
python -m pip install -r requirements-audit.txt
python analysis/verify_revision.py --output verified_core.json
python analysis/verify_new_materials.py --output-dir verified_new
python analysis/record_environment.py > current_environment.json
```

The core audit uses the already supplied M1–M4 folds and maize-leaf OOF records. The new-material audit checks all four feature tables, exact feature agreement for 1,357 candidate rows and the 28 stored patterns. It produces new audit files without overwriting the supplied results.

## Restore the existing signed arrays

Unzip motif_inputs_part1.zip and motif_inputs_part2.zip into the same parent directory, then:

```bash
python analysis/restore_motif_arrays.py --parts-root /path/to/parts_parent --output-dir /path/to/restored_inputs
python analysis/verify_new_materials.py --array-dir /path/to/restored_inputs --output-dir verified_with_intervals
```

The full interval check matches all 2,576 seqlet one-hot sequences and signed hypothetical-contribution intervals, including reverse complements. Recompressed NPZ file bytes can differ while numerical arrays remain identical.

## Model training

Obtain the exact reference FASTA/GTF and full processed expression-label files and validation whitelist specified by the inherited species configuration. The supplied common features are under data/processed/. Put/copy them with the external runtime input files. Do not use the partial examples as a full study input.

For the inherited M1–M4 drivers, set MCRE_DATA_DIR and MCRE_MODEL_DIR to explicit paths. For example, on Linux/macOS:

```bash
export MCRE_DATA_DIR=/path/to/full_runtime_inputs
export MCRE_MODEL_DIR=/path/to/new_m4_outputs
python code/model/DeepCRE_mRNA_BiLSTM_chr.py
```

PowerShell equivalents are `$env:MCRE_DATA_DIR = 'D:\mcre_inputs'` and `$env:MCRE_MODEL_DIR = 'D:\mcre_new_outputs'`. Other inherited models are DeepCRE_chr.py (M1), DeepCRE_BiLSTM_chr.py (M2) and DeepCRE_mRNA_chr.py (M3). Preserve each model in a separate output directory.

The updated M4 trainer saves scaler state plus actual balanced/scored gene manifests. The strict expected-gradients export reads those saved states and fails when they are absent:

```bash
python code/interpretation/attribute_expected_gradients.py
```

It saves signed auxiliary-feature attributions, matched gene IDs and feature values before taking absolute values. Supplied historical checkpoints lack these external scaler/sampling records; fitting a new scaler is a replay, not exact restoration of original predictions.

## Ten-comparison portable rerun

This is a new explicit implementation of the documented comparison protocol. It does not overwrite or certify the historical control training cohorts.

```bash
python code/reproducibility/prepare_task_arrays.py --task Zmay_leaf --data-dir /path/to/full_runtime_inputs --output prepared/Zmay_leaf.npz
python code/data_deal/build_control_features.py --kind regulatory8 --prepared prepared/Zmay_leaf.npz --output prepared/Zmay_regulatory8.csv
python code/data_deal/build_control_features.py --kind transcript8 --gff /path/to/annotation.gff3 --fasta /path/to/genome.fa --output prepared/Zmay_transcript8.csv
python code/model/train_comparisons.py --prepared prepared/Zmay_leaf.npz --model M0 --output-dir outputs/new_M0_Zmay_leaf
python code/model/train_comparisons.py --prepared prepared/Zmay_leaf.npz --model M4-T8 --control-features prepared/Zmay_transcript8.csv --output-dir outputs/new_M4T8_Zmay_leaf
```

The same command accepts M1, M2, M3, M4, M4-R8, M4-P, T8-only and R8-only. Control models require the corresponding eight-column CSV. Each control table must cover the prepared gene cohort; missing rows cause an error. New runs save configuration, folds, balanced IDs, scalers and raw OOF probabilities, enabling a real subsequent gene-paired audit. Their balance random stream is explicitly seeded per fold; they are not silently labeled as the original trained models.

## Raw maize descriptors

```bash
python code/data_deal/rebuild_maize_features.py --gff /path/to/maize.gff3 --fasta /path/to/maize.fa --output-dir outputs/rebuilt_raw_features
```

This preserves the supplied extractor's transcript definitions and explicitly writes both raw transcript rows and its calculated gene aggregate. It applies no undocumented transformation and does not replace data/processed/Zmay_mRNA_features_common.csv.

## Motif result reuse and a new run

```bash
python code/motifs/export_patterns_meme.py --h5 data/motifs/tfmodisco_results_1362.h5 --output outputs/patterns.meme
python code/motifs/run_tfmodisco_lite.py --input-dir /path/to/restored_inputs --output-h5 outputs/new_patterns.h5
python code/motifs/run_tomtom.py --query outputs/patterns.meme --database /path/to/actual_plant_profiles.meme --output-dir outputs/new_tomtom
```

The original builder scripts remain under code/motifs/ for provenance and prepared-input generation. They reconstruct preprocessing and must not be advertised as exact historical checkpoint replay. The TF-MoDISco wrapper retains the supplied direct score-as-hypothetical-contribution convention. The TOMTOM wrapper records its explicit new command, version and database checksum. The historical JASPAR release is not guessed: the supplied shell example named 2022 while the earlier paper said 2024. Original database/command logs remain needed to reproduce the exact historical annotation run. MEME export here uses stored sequence PFMs with an explicit pseudocount; it is a documented new export, not a claim to recover an unprovided old query file.

## WGCNA and enrichment

With WGCNA installed in R:

```bash
Rscript R/reproduce_network_from_saved_expression.R . outputs/WGCNA_replay
Rscript R/enrichment_from_saved_mapping.R .
Rscript R/enrichment_from_saved_mapping.R . --rerun
```

The network replay reads the actual 23-by-27,910 expression object, fixed corrected module assignments and original-module connectivity ranking. It recomputes indicator associations and full-neighbor TOM for the 500 selected genes without allocating the whole square matrix. It does not reselect modules. The enrichment default reads the supplied frozen exports; --rerun needs clusterProfiler and org.Zmays.eg.db and makes a new annotation query, recording the date and environment. A new query can differ as annotation databases change. These R entry points were reviewed as source but not executed here because this environment has no R runtime.

## Records still required for specific claims

Exact original checkpoint replay needs original scalers and balanced/background gene identities. Feature-value direction and fold-wise directional consistency need signed eight-feature per-gene attributions, not only the signed motif sequence arrays. A verified family-wise benchmark needs actual group/fold assignments, homolog mappings and gene-level M1/M4 predictions. Root high-class overlap needs the original root label file. Historical control provenance needs the original training manifests; new portable reruns are separately labeled. The revised manuscript does not claim that these specific missing records have been reconstructed.

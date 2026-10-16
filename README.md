# m-CRE

m-CRE classifies plant genes as relatively high or low expressed within a species and tissue, using paired gene-boundary DNA windows and eight transcript structural and compositional features. It also provides attribution and co-expression analyses for prioritizing candidate regulatory motifs.

## Installation and quick checks

Run commands from the repository root:

```bash
python -m pip install -r requirements-audit.txt
python analysis/verify_revision.py --output outputs/verified_core.json
python analysis/verify_new_materials.py --output-dir outputs/verified_new
python code/motifs/export_patterns_meme.py --h5 data/motifs/tfmodisco_results_1362.h5 --output outputs/patterns.meme
```

These checks inspect supplied CSV/HDF5 records and require no GPU or model retraining. Training dependencies are listed in requirements-training.txt.

## Code and data

| Component | Location |
| --- | --- |
| M1-M4 models and model comparisons | code/model/ |
| Expected-gradients attribution | code/interpretation/attribute_expected_gradients.py |
| Feature calculation and input preparation | code/data_deal/ and code/reproducibility/ |
| TF-MoDISco, MEME export and TOMTOM | code/motifs/ |
| WGCNA and enrichment | R/ and results/revised/WGCNA/ |
| Complete common feature inputs | data/processed/; 27,652 Atha, 34,081 Sbic, 34,413 Slyc and 39,745 Zmay genes |
| Ten-model performance tables | data/model_metrics/ten_model_controls/ |
| Candidate inputs and source metadata | data/candidates/ and data/metadata/ |
| Stored TF-MoDISco patterns | data/motifs/tfmodisco_results_1362.h5 |
| Supporting results and data definitions | results/revised/ |

The original source versions are retained under provenance/original_code/. The maize feature reconstruction utility imports the original extractor from that directory.

## Analysis definitions

- The requested motif cohort contains 1,362 genes; 1,357 have aligned complete inputs. Legacy filenames ending in 1362 retain their original names.
- Candidate replay yields 1,182 high and 175 low predictions. Its 165-gene overlap with the original maize-leaf OOF export contains 142 originally predicted-high genes; these records and denominators are reported separately.
- Attribution uses SHAP GradientExplainer expected gradients. Sequence channels follow A/C/G/T; the corrected channel summary is supplied with the results.
- The HDF5 contains 13 positive and 15 negative patterns, with 2,576 retained seqlets from 348 genes. The stored seqlet support is distinct from the full 1,357-gene input cohort.
- Figure 2a/d uses mean chromosome-fold AUROC and sample SD. Figure 2c/f uses AUROC over concatenated source-model prediction records. These summaries are distinct from original training OOF AUROC.
- Common feature tables contain preprocessed modeling values. Fold scaling is fitted on balanced training genes. Raw reconstructed features must not silently replace the common inputs.

## Documentation and availability

See [Reproduction instructions](docs/REPRODUCIBILITY.md) for commands, required inputs and validation scope, [Data inventory](docs/DATA.md) for assets and data definitions, and [Changes since v1.0.0](docs/RELEASE_NOTES.md) for software changes.

Versioned software is listed on the [GitHub releases page](https://github.com/5lyt/m-CRE/releases). The baseline v1.0.0 archive is identified by [DOI 10.5281/zenodo.20505253](https://doi.org/10.5281/zenodo.20505253); version-specific archival identifiers belong to their corresponding releases. Large model/input asset names and checksums are listed in data/large_asset_manifest.json.

The Academic Research License is provided in LICENSE.

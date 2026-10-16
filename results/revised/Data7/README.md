# Supplementary Data 7

This dataset records model inputs, attribution summaries and candidate/pattern provenance used in the accompanying study.

- The four `*_mRNA_features_common.csv` files contain the complete supplied numerical inputs with one gene ID and eight ordered features. `full_feature_input_audit.csv` reports the four species, row counts and column summaries; all 1,357 candidate feature vectors match the full maize table exactly.
- `sample_manifest_76_runs.csv`, `species_tissue_accession_mapping.csv` and `species_tissue_source_summary.csv` preserve the supplied source-study/sample mappings. The maize network uses the 23-run subset described in Methods.
- `candidate_1362_with_verified_module_statistics.csv` joins the requested manifest to corrected WGCNA statistics. `five_candidates_excluded_from_tfmodisco.csv` lists requested IDs absent from the complete prepared model-input records; the exact upstream reason was not recorded.
- `gene_meta_1362_used.csv` and `motif_input_1357_verified_metadata.csv` retain the actual 1,357-row input order despite the legacy filename. Original-OOF comparison tables separate original whitelist-scored predictions from later candidate replay with reconstructed preprocessing.
- Figure 3 sampling, positional CV and leaf–root similarity files summarize the global attribution export. The corrected nucleotide table maps the legacy A/T/C/G labels to the actual A/C/G/T channels. Signed candidate sequence profiles describe a separate population and do not establish the direction of the eight auxiliary-feature effects.
- `motif_h5_pattern_audit.csv`, `motif_h5_direct_row_check.csv` and `motif_seqlet_gene_membership.csv` describe 28 stored patterns, 2,576 retained seqlets and their 348 represented genes. Every stored sequence and signed-score interval matches its referenced input row and strand orientation. The HDF5 object itself is included in Supplementary Software 1 at `data/motifs/tfmodisco_results_1362.h5`.

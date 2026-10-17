# Changes since v1.0.0

- Add complete common eight-feature input tables for four species and the TF-MoDISco HDF5 result object.
- Add ten-model comparison summaries, candidate/OOF comparisons, source metadata and retained-seqlet membership tables.
- Correct attribution method naming to expected gradients and nucleotide-channel labels to A/C/G/T.
- Save training scaler and cohort state; export signed auxiliary-feature attribution before magnitude aggregation.
- Add model-comparison, control-feature, WGCNA/enrichment and motif-analysis entry points with reproduction commands.
- Document chromosome-fold, pooled OOF and cross-species prediction summaries separately.
- Distinguish 1,362 requested candidates, 1,357 complete inputs and the 348 genes represented by 2,576 retained seqlets.

New rerun programs are distinct from the historical runs that produced the supplied results. Historical model checkpoints and motif-input arrays retain their original numerical contents. See REPRODUCIBILITY.md for the scope of validation and requirements for exact checkpoint replay.

- Repackage supplied M4 chromosome checkpoints and signed motif inputs with explicit asset checksums and unchanged model bytes/array values.
- Add model-specific M0 state, M1-M3 maize-leaf checkpoint assets and historical control training records.
- Document verified 40-row control AUROC agreement and preserve the distinction between supplied M0/control state and missing original M4 state.

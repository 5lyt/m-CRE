# Changes since v1.0.0

- Add complete common eight-feature input tables for four species and the TF-MoDISco HDF5 result object.
- Add ten-model comparison summaries, candidate/OOF comparisons, source metadata and retained-seqlet membership tables.
- Correct attribution method naming to expected gradients and nucleotide-channel labels to A/C/G/T.
- Save training scaler and cohort state; export signed auxiliary-feature attribution before magnitude aggregation.
- Add model-comparison, control-feature, WGCNA/enrichment and motif-analysis entry points with reproduction commands.
- Document chromosome-fold, pooled OOF and cross-species prediction summaries separately.
- Distinguish 1,362 requested candidates, 1,357 complete inputs and the 348 genes represented by 2,576 retained seqlets.

New rerun programs are distinct from the historical runs that produced the supplied results. Historical model checkpoints and motif-input arrays retain their original numerical contents. See REPRODUCIBILITY.md for the scope of validation and requirements for exact checkpoint replay.

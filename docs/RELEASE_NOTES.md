# Release notes for m-CRE v1.1.0

Prepared revision of the existing v1.0.0 repository for the Scientific Reports manuscript. Publish this note only with the actual new tag.

- Preserve original M1–M4 source, example data, WGCNA objects, results and Academic Research License; consolidate revised entry points in one source tree.
- Add all four full common eight-feature input tables and the TF-MoDISco HDF5 object.
- Correct expected-gradients method naming and ACGT nucleotide-channel export; save signed auxiliary outputs before magnitude aggregation.
- Save training scaler and cohort state and provide strict saved-state attribution replay.
- Add candidate/OOF audits, exact seqlet interval checks, explicit cross-species AUROC aggregation and sample metadata.
- Add complete comparison rerun, control feature builders, WGCNA/enrichment replay, motif/MEME/TOMTOM entry points and commands.
- Distinguish 1,362 requested candidates, 1,357 actual inputs and 348 genes supporting the retained 2,576 seqlets.

This preparation does not retrain historical models or certify unrecorded preprocessing, family-wise partitions or the exact original motif database. Existing separate model/array assets are reused unchanged.

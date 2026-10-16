# Ten-model comparison results (revision controls)

This directory supplies the ten-model out-of-fold AUROC results shown in the
revised Figure 2 (panels a and d), including the five matched control models.

## Files

- `Figure2_10model_AUROC_summary.csv` — long format, one row per model-task.
  Columns: `model, task, species, tissue, pooled_oof_auroc, mean_fold_auroc, fold_std, n_folds`.
  - `pooled_oof_auroc`: one AUROC computed over concatenated out-of-fold gene predictions.
  - `mean_fold_auroc`: unweighted mean of per-chromosome-fold AUROCs.
  - `fold_std`: sample standard deviation of the per-fold AUROCs (the error bars in Figure 2a/d).
  - `n_folds`: number of held-out chromosomes.
- `Figure2_10model_mean_fold_AUROC_wide.csv` — task x model wide view of `mean_fold_auroc`.
- `Figure2_10model_pooled_OOF_AUROC_wide.csv` — task x model wide view of `pooled_oof_auroc`.

## Model labels

| Label | Meaning |
| --- | --- |
| M0 | eight transcript features only |
| M1 | sequence-only baseline (DeepCRE) |
| M2 | sequence + BiLSTM (L-CRE) |
| M3 | sequence + features, no BiLSTM |
| M4 | full model (sequence + features + BiLSTM) |
| M4-T8 | full model with transcript8 control features |
| M4-R8 | full model with regulatory8 control features |
| M4-P | full model with partition-wise permuted eight features |
| T8-only | transcript8 features only |
| R8-only | regulatory8 features only |

## Provenance

These numbers were produced by the reviewer-response control pipeline
`DeepCRE_review_controls_v3` (entry scripts `run_controls.py` / `legacy_controls.py`
and the feature builders `transcript_controls.py` / `control_math.py`), not by the
portable rerun entry point in this repository.

For an equivalent portable rerun in this repository, build the NPZ inputs with
`code/reproducibility/prepare_task_arrays.py`, build the transcript8/regulatory8
tables with `code/data_deal/build_control_features.py`, then run
`code/model/train_comparisons.py` with the matching `--model`. The portable rerun
may differ in low-order digits from these historical numbers; it is provided for
reproducibility, not as a byte-for-byte replacement.

`mean_fold_auroc +/- fold_std` is the statistic reported in Figure 2a/d; the error
is the sample standard deviation across chromosome folds, not a standard error or
confidence interval. `pooled_oof_auroc` is a separate statistic; do not mix the two.

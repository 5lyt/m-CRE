#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build TF-MoDISco-lite inputs for Zea mays leaf using the 1,362
WGCNA-supported high-expression candidate genes.

Outputs:
  - ohe_1362.npz: one-hot DNA sequences, key = "sequences", shape = (N, L, 4)
  - attr_1362.npz: signed sequence SHAP attributions, key = "attributions", shape = (N, L, 4)
  - gene_meta_1362_used.csv: gene_id / chromosome / label / prediction / WGCNA info for used genes

This script is designed for the newer chromosome-wise dual-input model:
  output_DeepCRE_mRNA_BiLSTM_chr_model_chr*.keras

Important:
  Do not take abs(SHAP) before TF-MoDISco. Signed attributions are needed so that
  positive/negative patterns can be separated correctly.
"""

import argparse
import gc
import glob
import os
import random
import re
import warnings
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from pyfaidx import Fasta
import pyranges as pr
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

SEED = 42
UPSTREAM = 1000
DOWNSTREAM = 500
GAP_LEN = 20
SEQ_TOTAL_LEN = 2 * (UPSTREAM + DOWNSTREAM) + GAP_LEN  # 3020 for the new model
PROMOTER_MASK_SLICE = slice(UPSTREAM, UPSTREAM + 3)
TERMINATOR_MASK_SLICE = slice(UPSTREAM + (DOWNSTREAM * 2) + 17, UPSTREAM + (DOWNSTREAM * 2) + 20)

MRNA_COLS = [
    "UTR5LEN", "CDSLEN", "INTRONLEN", "UTR3LEN",
    "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY"
]

EXPR_VALUE_CANDIDATES = [
    "log(MaxTPM+1)", "logMaxTPM", "log_MaxTPM_plus_1",
    "log_max_tpm_plus_1", "log_max_tpm"
]

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_gene_id(x) -> str:
    return str(x).strip()


def auto_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def standardize_gene_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {str(c).strip().lower(): c for c in df.columns}
    if "gene_id" in lower:
        df = df.rename(columns={lower["gene_id"]: "gene_id"})
    elif "gene.symbol" in lower:
        df = df.rename(columns={lower["gene.symbol"]: "gene_id"})
    elif "gene" in lower:
        df = df.rename(columns={lower["gene"]: "gene_id"})
    elif "geneid" in lower:
        df = df.rename(columns={lower["geneid"]: "gene_id"})
    else:
        df = df.rename(columns={df.columns[0]: "gene_id"})
    df["gene_id"] = df["gene_id"].astype(str).map(normalize_gene_id)
    return df


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower:
            return lower[key]
    return None


def find_expression_value_column(df: pd.DataFrame) -> Optional[str]:
    col = find_first_column(df, EXPR_VALUE_CANDIDATES)
    if col:
        return col
    for c in df.columns:
        name = str(c).lower()
        if "log" in name and "tpm" in name:
            return c
    return None


def load_expression_labels(path: str) -> pd.DataFrame:
    """Return gene_id, expression_value, label. label: low=0, high=1, medium=2."""
    df = standardize_gene_id_column(auto_read_table(path))

    # If explicit Category exists, use it directly.
    cat_col = find_first_column(df, ["Category", "category", "label_name", "class"])
    label_col = find_first_column(df, ["label"])
    value_col = find_expression_value_column(df)

    if cat_col is not None:
        cat = df[cat_col].astype(str).str.strip().str.lower()
        mapper = {"low": 0, "0": 0, "medium": 2, "mid": 2, "1/2": 2, "0.5": 2, "high": 1, "1": 1}
        df["label"] = cat.map(mapper)
    elif label_col is not None:
        vals = pd.to_numeric(df[label_col], errors="coerce")
        # Accept 0/1/2; if only 0/1, no medium class.
        df["label"] = vals
    elif value_col is not None:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col]).copy()
        q25, q75 = np.percentile(df[value_col].values, [25, 75])
        df["label"] = np.where(df[value_col] <= q25, 0, np.where(df[value_col] >= q75, 1, 2))
    else:
        raise ValueError(f"Could not infer expression label/category from columns: {list(df.columns)}")

    if value_col is None:
        df["expression_value"] = np.nan
    else:
        df["expression_value"] = pd.to_numeric(df[value_col], errors="coerce")

    out = df[["gene_id", "expression_value", "label"]].dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [(0, int(p)) if p.isdigit() else (1, p.lower()) for p in parts]


def resolve_unique_file(base_dir: str, patterns: List[str]) -> str:
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(base_dir, pat)))
    matches = sorted(set(matches), key=natural_sort_key)
    if not matches:
        raise FileNotFoundError(f"No file matched these patterns in {base_dir}: {patterns}")
    return matches[0]


def reverse_complement_onehot(encoded_seq: np.ndarray) -> np.ndarray:
    # Channels are A,C,G,T; reverse along length and complement by reversing channels.
    return encoded_seq[::-1, ::-1]


def onehot(seq: str) -> np.ndarray:
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
        "unk": [0, 0, 0, 0],
    }
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nt in enumerate(str(seq).upper()):
        arr[i] = mapping.get(nt, mapping["unk"])
    return arr


def load_species_feature_table(data_dir: str, valid_chroms: Optional[List[str]]) -> pd.DataFrame:
    fasta_path = resolve_unique_file(data_dir, ["Zea_mays*.fa", "Zea_mays*.fasta", "*Zea*mays*.fa", "*Zea*mays*.fasta"])
    gtf_path = resolve_unique_file(data_dir, ["Zea_mays*.gtf", "Zea_mays*.gff", "Zea_mays*.gff3", "*Zea*mays*.gtf", "*Zea*mays*.gff3"])
    mrna_path = resolve_unique_file(data_dir, ["Zmay_mRNA_features_common.csv", "*Zmay*mRNA*features*.csv", "*mRNA_features_common.csv"])

    print(f"[DATA] FASTA: {fasta_path}")
    print(f"[DATA] GTF/GFF: {gtf_path}")
    print(f"[DATA] mRNA features: {mrna_path}")

    fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_models = pr.read_gtf(gtf_path, as_df=True)
    gene_models = gene_models[gene_models["Feature"] == "gene"].copy()

    biotype_col = find_first_column(gene_models, ["gene_biotype", "gene_type", "biotype"])
    if biotype_col is not None:
        pc = gene_models[gene_models[biotype_col].astype(str) == "protein_coding"].copy()
        if len(pc) > 0:
            gene_models = pc

    needed_cols = ["Chromosome", "Start", "End", "Strand", "gene_id"]
    missing = [c for c in needed_cols if c not in gene_models.columns]
    if missing:
        raise ValueError(f"GTF/GFF is missing required columns from pyranges parse: {missing}")

    gene_models = gene_models[needed_cols].copy()
    gene_models["gene_id"] = gene_models["gene_id"].astype(str).map(normalize_gene_id)
    gene_models["Chromosome"] = gene_models["Chromosome"].astype(str)

    if valid_chroms:
        valid_set = set(map(str, valid_chroms))
        gm = gene_models[gene_models["Chromosome"].isin(valid_set)].copy()
        if len(gm) > 0:
            gene_models = gm

    gene_models = gene_models.drop_duplicates(subset=["gene_id"], keep="first")

    mrna_df = standardize_gene_id_column(pd.read_csv(mrna_path))
    missing_mrna = [c for c in MRNA_COLS if c not in mrna_df.columns]
    if missing_mrna:
        raise ValueError(f"mRNA feature file lacks columns: {missing_mrna}")
    mrna_df = mrna_df[["gene_id"] + MRNA_COLS].dropna(subset=MRNA_COLS).copy()
    mrna_genes = set(mrna_df["gene_id"])

    seq_records = []
    for row in gene_models.itertuples(index=False):
        chrom = str(row.Chromosome)
        start = int(row.Start)
        end = int(row.End)
        strand = str(row.Strand)
        gene_id = normalize_gene_id(row.gene_id)
        if gene_id not in mrna_genes:
            continue

        if strand == "+":
            prom_start, prom_end = start - UPSTREAM, start + DOWNSTREAM
            term_start, term_end = end - DOWNSTREAM, end + UPSTREAM
        else:
            prom_start, prom_end = end - DOWNSTREAM, end + UPSTREAM
            term_start, term_end = start - UPSTREAM, start + DOWNSTREAM

        if prom_start <= 0 or term_start <= 0:
            continue

        try:
            prom_seq = fasta[chrom][prom_start:prom_end]
            term_seq = fasta[chrom][term_start:term_end]
            if strand == "+":
                encoded = np.concatenate([onehot(prom_seq), np.zeros((GAP_LEN, 4), dtype=np.float32), onehot(term_seq)], axis=0)
            else:
                encoded = np.concatenate([
                    reverse_complement_onehot(onehot(prom_seq)),
                    np.zeros((GAP_LEN, 4), dtype=np.float32),
                    reverse_complement_onehot(onehot(term_seq)),
                ], axis=0)
        except Exception:
            continue

        if encoded.shape != (SEQ_TOTAL_LEN, 4):
            continue

        seq_records.append({"gene_id": gene_id, "chromosome": chrom, "encoded_seq": encoded})

    seq_df = pd.DataFrame(seq_records)
    base_df = seq_df.merge(mrna_df, on="gene_id", how="inner").reset_index(drop=True)
    print(f"[DATA] usable genes with sequence + mRNA features: {len(base_df)}")
    return base_df


def make_task_dataframe(base_df: pd.DataFrame, expr_df: pd.DataFrame) -> pd.DataFrame:
    label_df = expr_df[expr_df["label"].isin([0, 1])][["gene_id", "expression_value", "label"]].copy()
    df = base_df.merge(label_df, on="gene_id", how="inner").dropna(subset=MRNA_COLS).reset_index(drop=True)
    print(f"[DATA] task genes after merging labels: {len(df)}; high={int((df['label']==1).sum())}; low={int((df['label']==0).sum())}")
    return df


def balance_train_data(train_seqs, train_mrna, train_labels):
    low = np.where(train_labels == 0)[0]
    high = np.where(train_labels == 1)[0]
    n = min(len(low), len(high))
    if n == 0:
        return train_seqs, train_mrna, train_labels
    rng = np.random.RandomState(SEED)
    sel_low = rng.choice(low, n, replace=False)
    sel_high = rng.choice(high, n, replace=False)
    Xs = np.concatenate([train_seqs[sel_low], train_seqs[sel_high]], axis=0)
    Xm = np.concatenate([train_mrna[sel_low], train_mrna[sel_high]], axis=0)
    y = np.concatenate([train_labels[sel_low], train_labels[sel_high]], axis=0)
    return shuffle(Xs, Xm, y, random_state=SEED)


def choose_background(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx_parts = []
    for lab in [0, 1]:
        lab_idx = np.where(y == lab)[0]
        take = min(len(lab_idx), max_n // 2)
        if take > 0:
            idx_parts.append(rng.choice(lab_idx, size=take, replace=False))
    idx = np.concatenate(idx_parts) if idx_parts else rng.choice(len(y), size=max_n, replace=False)
    if len(idx) < max_n:
        remain = np.setdiff1d(np.arange(len(y)), idx)
        extra = rng.choice(remain, size=min(max_n - len(idx), len(remain)), replace=False)
        idx = np.concatenate([idx, extra])
    return np.sort(idx)


def normalize_shap_outputs(shap_values):
    """Return sequence shap array with shape (n, L, 4)."""
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            seq_sv = shap_values[0]
        elif len(shap_values) == 1 and isinstance(shap_values[0], list) and len(shap_values[0]) == 2:
            seq_sv = shap_values[0][0]
        else:
            seq_sv = shap_values[0]
    elif isinstance(shap_values, tuple):
        seq_sv = shap_values[0]
    else:
        seq_sv = shap_values

    arr = np.asarray(seq_sv)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Unexpected sequence SHAP shape: {arr.shape}; expected (n, L, 4)")
    return arr.astype(np.float32)


def find_model_path(model_dir: str, chrom: str) -> str:
    chrom = str(chrom)
    candidates = []
    # model_dir may be either .../Zmay_leaf or the parent output directory.
    search_dirs = [model_dir, os.path.join(model_dir, "Zmay_leaf")]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        patterns = [
            os.path.join(d, f"output_DeepCRE_mRNA_BiLSTM_chr_model_chr{chrom}.keras"),
            os.path.join(d, f"*model_chr{chrom}.keras"),
            os.path.join(d, f"*chr{chrom}.keras"),
            os.path.join(d, f"*model_chr{chrom}.h5"),
            os.path.join(d, f"*chr{chrom}.h5"),
        ]
        for pat in patterns:
            candidates.extend(glob.glob(pat))
    candidates = sorted(set(candidates), key=natural_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No model found for chromosome {chrom} under {model_dir}")
    return candidates[0]


def compute_shap_for_candidates(args, data_df: pd.DataFrame, core_info: pd.DataFrame) -> None:
    import shap

    ensure_dir(args.out_dir)

    core_genes = set(core_info["gene_id"].astype(str))
    candidate_df = data_df[data_df["gene_id"].isin(core_genes)].copy().reset_index(drop=True)

    missing_gene_ids = sorted(core_genes - set(candidate_df["gene_id"]), key=natural_sort_key)
    if missing_gene_ids:
        pd.DataFrame({"gene_id": missing_gene_ids}).to_csv(os.path.join(args.out_dir, "gene_ids_missing_from_model_data.csv"), index=False)
        print(f"[WARN] {len(missing_gene_ids)} of 1,362 genes were not found in model data; see gene_ids_missing_from_model_data.csv")

    if args.max_genes and len(candidate_df) > args.max_genes:
        candidate_df = candidate_df.iloc[:args.max_genes].copy()
        print(f"[TEST] Using only first {len(candidate_df)} genes because --max-genes was set.")

    X_seq_all = np.stack(data_df["encoded_seq"].values).astype(np.float32)
    X_mrna_all = data_df[MRNA_COLS].values.astype(np.float32)
    y_all = data_df["label"].values.astype(np.int32)

    all_ohe, all_attr, all_meta = [], [], []
    chroms = sorted(candidate_df["chromosome"].astype(str).unique().tolist(), key=natural_sort_key)
    print(f"[RUN] candidate genes available for SHAP: {len(candidate_df)} across chromosomes: {chroms}")

    for round_i, chrom in enumerate(chroms, start=1):
        cand_chr = candidate_df[candidate_df["chromosome"].astype(str) == str(chrom)].copy()
        if cand_chr.empty:
            continue

        model_path = find_model_path(args.model_dir, chrom)
        print(f"\n[CHR {chrom}] model: {model_path}")
        print(f"[CHR {chrom}] candidate genes before prediction filter: {len(cand_chr)}")

        train_idx = data_df.index[data_df["chromosome"].astype(str) != str(chrom)].to_numpy()
        cand_idx = cand_chr.index.to_numpy()

        X_seq_train = X_seq_all[train_idx].copy()
        X_mrna_train = X_mrna_all[train_idx].copy()
        y_train = y_all[train_idx]

        X_seq_train_bal, X_mrna_train_bal, y_train_bal = balance_train_data(X_seq_train, X_mrna_train, y_train)
        X_seq_train_bal[:, PROMOTER_MASK_SLICE, :] = 0
        X_seq_train_bal[:, TERMINATOR_MASK_SLICE, :] = 0

        scaler = StandardScaler()
        X_mrna_train_bal = scaler.fit_transform(X_mrna_train_bal).astype(np.float32)

        X_seq_cand = np.stack(cand_chr["encoded_seq"].values).astype(np.float32)
        X_seq_cand[:, PROMOTER_MASK_SLICE, :] = 0
        X_seq_cand[:, TERMINATOR_MASK_SLICE, :] = 0
        X_mrna_cand = scaler.transform(cand_chr[MRNA_COLS].values.astype(np.float32)).astype(np.float32)
        y_cand = cand_chr["label"].values.astype(np.int32)

        model = tf.keras.models.load_model(model_path, compile=False)
        model_seq_len = int(model.inputs[0].shape[1])
        if model_seq_len != X_seq_cand.shape[1]:
            raise ValueError(
                f"Model input length is {model_seq_len}, but generated sequence length is {X_seq_cand.shape[1]}. "
                "Do not mix the old 3022-bp pipeline with the new 3020-bp chromosome-wise model."
            )

        y_prob = model.predict([X_seq_cand, X_mrna_cand], batch_size=args.predict_batch_size, verbose=0).reshape(-1)
        y_pred = (y_prob >= args.pred_threshold).astype(int)

        if args.require_predicted_high:
            keep = y_pred == 1
            print(f"[CHR {chrom}] kept predicted-high genes: {int(keep.sum())}/{len(keep)}")
            if not np.any(keep):
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                continue
            cand_chr = cand_chr.iloc[np.where(keep)[0]].copy()
            X_seq_cand = X_seq_cand[keep]
            X_mrna_cand = X_mrna_cand[keep]
            y_cand = y_cand[keep]
            y_prob = y_prob[keep]
            y_pred = y_pred[keep]

        bg_idx = choose_background(y_train_bal, min(args.background, len(y_train_bal)), SEED + round_i)
        X_seq_bg = X_seq_train_bal[bg_idx].astype(np.float32)
        X_mrna_bg = X_mrna_train_bal[bg_idx].astype(np.float32)

        explainer = shap.GradientExplainer(model, [X_seq_bg, X_mrna_bg])
        attr_parts = []
        for start in range(0, len(X_seq_cand), args.shap_batch_size):
            end = start + args.shap_batch_size
            sv = explainer.shap_values([X_seq_cand[start:end], X_mrna_cand[start:end]])
            attr_parts.append(normalize_shap_outputs(sv))
        seq_attr = np.concatenate(attr_parts, axis=0).astype(np.float32)

        if args.multiply_by_onehot:
            # Optional conversion to actual observed-base contribution scores.
            # Default is False because TF-MoDISco-lite can use the signed per-base-channel attribution matrix directly.
            seq_attr = seq_attr * X_seq_cand

        meta = cand_chr.drop(columns=["encoded_seq"], errors="ignore").copy()
        meta["model_chr"] = chrom
        meta["model_path"] = model_path
        meta["pred_prob_high"] = y_prob
        meta["pred_label"] = y_pred
        meta["true_label"] = y_cand
        meta["used_for_tfmodisco"] = True

        all_ohe.append(X_seq_cand.astype(np.float32))
        all_attr.append(seq_attr)
        all_meta.append(meta)

        print(f"[CHR {chrom}] saved SHAP shape: {seq_attr.shape}; mean_abs_attr={np.mean(np.abs(seq_attr)):.6g}")
        del model, explainer, X_seq_bg, X_mrna_bg, X_seq_cand, X_mrna_cand, seq_attr
        tf.keras.backend.clear_session()
        gc.collect()

    if not all_ohe:
        raise RuntimeError("No TF-MoDISco inputs were generated. Check model paths and gene overlap.")

    ohe = np.concatenate(all_ohe, axis=0).astype(np.float32)
    attr = np.concatenate(all_attr, axis=0).astype(np.float32)
    meta_df = pd.concat(all_meta, ignore_index=True)

    # Add WGCNA information back to the exact output order.
    core_cols = [c for c in core_info.columns if c not in meta_df.columns or c == "gene_id"]
    meta_df = meta_df.merge(core_info[core_cols], on="gene_id", how="left")

    if ohe.shape != attr.shape or ohe.ndim != 3 or ohe.shape[-1] != 4:
        raise ValueError(f"Invalid final shapes: one_hot={ohe.shape}, attributions={attr.shape}; expected both (N, L, 4).")

    np.savez_compressed(os.path.join(args.out_dir, "ohe_1362.npz"), sequences=ohe)
    np.savez_compressed(os.path.join(args.out_dir, "attr_1362.npz"), attributions=attr)
    meta_df.to_csv(os.path.join(args.out_dir, "gene_meta_1362_used.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "n_requested_core_genes": int(len(core_genes)),
        "n_available_in_model_data": int(len(candidate_df)),
        "n_used_for_tfmodisco": int(ohe.shape[0]),
        "one_hot_shape": list(ohe.shape),
        "attribution_shape": list(attr.shape),
        "mean_abs_attribution": float(np.mean(np.abs(attr))),
        "require_predicted_high": bool(args.require_predicted_high),
        "multiply_by_onehot": bool(args.multiply_by_onehot),
    }
    pd.Series(summary).to_json(os.path.join(args.out_dir, "tfmodisco_input_summary.json"), force_ascii=False, indent=2)

    print("\n[DONE] TF-MoDISco-lite input files:")
    print(f"  one-hot:      {os.path.join(args.out_dir, 'ohe_1362.npz')}  shape={ohe.shape}")
    print(f"  attributions: {os.path.join(args.out_dir, 'attr_1362.npz')} shape={attr.shape}")
    print(f"  metadata:     {os.path.join(args.out_dir, 'gene_meta_1362_used.csv')}")
    print(f"  mean_abs_attr={np.mean(np.abs(attr)):.6g}")


def parse_args():
    p = argparse.ArgumentParser(description="Build TF-MoDISco-lite inputs for Zmay_leaf 1,362 WGCNA-supported high-expression genes.")
    p.add_argument("--data-dir", required=True, help="Directory containing Zea_mays FASTA/GTF and Zmay_mRNA_features_common.csv")
    p.add_argument("--model-dir", required=True, help="Directory containing chromosome-wise Zmay_leaf .keras models, or its parent directory")
    p.add_argument("--gene-info-csv", required=True, help="CSV containing the 1,362 WGCNA-supported gene_id list")
    p.add_argument("--expression-file", default=None, help="Zmay leaf expression/label file. If omitted, uses data-dir/Zmay_leaf_expression_label_common.csv or data-dir/zea_counts.csv")
    p.add_argument("--out-dir", required=True, help="Output directory for .npz and metadata")
    p.add_argument("--valid-chroms", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated maize chromosomes to keep")
    p.add_argument("--background", type=int, default=64, help="Background samples per chromosome for SHAP")
    p.add_argument("--shap-batch-size", type=int, default=32, help="Batch size for SHAP computation")
    p.add_argument("--predict-batch-size", type=int, default=128, help="Batch size for model prediction")
    p.add_argument("--pred-threshold", type=float, default=0.5, help="High-class probability threshold")
    p.add_argument("--require-predicted-high", action="store_true", help="Keep only genes predicted as high by the chromosome-wise model")
    p.add_argument("--multiply-by-onehot", action="store_true", help="Use actual observed-base contribution scores = signed SHAP * one-hot")
    p.add_argument("--max-genes", type=int, default=None, help="Small test mode: only run first N available genes")
    return p.parse_args()


def main():
    args = parse_args()
    valid_chroms = [x.strip() for x in args.valid_chroms.split(",") if x.strip()]

    gene_info = standardize_gene_id_column(pd.read_csv(args.gene_info_csv))
    if len(gene_info) != 1362:
        print(f"[WARN] gene-info CSV has {len(gene_info)} rows, not 1,362. Continuing with supplied rows.")

    if args.expression_file is None:
        candidates = [
            os.path.join(args.data_dir, "Zmay_leaf_expression_label_common.csv"),
            os.path.join(args.data_dir, "zea_counts.csv"),
        ]
        args.expression_file = next((x for x in candidates if os.path.exists(x)), None)
        if args.expression_file is None:
            raise FileNotFoundError("Please provide --expression-file; neither Zmay_leaf_expression_label_common.csv nor zea_counts.csv was found in data-dir.")

    print(f"[DATA] expression labels: {args.expression_file}")
    expr_df = load_expression_labels(args.expression_file)
    base_df = load_species_feature_table(args.data_dir, valid_chroms)
    task_df = make_task_dataframe(base_df, expr_df)
    compute_shap_for_candidates(args, task_df, gene_info)


if __name__ == "__main__":
    main()

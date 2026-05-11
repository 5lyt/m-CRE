import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import gc
import re
import glob
import json
import pickle
import random
import traceback
import warnings
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import pyranges as pr
from pyfaidx import Fasta

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =========================================================
# 0. Global configuration
# =========================================================
BASE_PATH = "/home"
DATA_DIR = os.path.join(BASE_PATH, "data")

MODEL_OUTPUT_ROOT = os.path.join(BASE_PATH, "output_DeepCRE_mRNA_BiLSTM_chr")
MODEL_OUTPUT_FALLBACKS = [
    MODEL_OUTPUT_ROOT,
    os.path.join(BASE_PATH, "outputs_output_DeepCRE_mRNA_BiLSTM_chr"),
]

OUTPUT_PREFIX = "output_DeepCRE_mRNA_BiLSTM_chr"
MODEL_GLOB_PATTERN = f"{OUTPUT_PREFIX}_model_chr*.keras"
OUT_DIR = os.path.join(MODEL_OUTPUT_ROOT, "cross_species_prediction_chr_models")

VALIDATION_PICKLE_CANDIDATES = [
    os.path.join(DATA_DIR, "validation_genes.pickle"),
    os.path.join(BASE_PATH, "validation_genes.pickle"),
    os.path.join(BASE_PATH, "data", "validation_genes.pickle"),
]

SEED = 42
BATCH_SIZE = 64
THRESHOLD = 0.5

UPSTREAM = 1000
DOWNSTREAM = 500
GAP_LEN = 20
SEQ_TOTAL_LEN = 2 * (UPSTREAM + DOWNSTREAM) + GAP_LEN  # 3020

PROMOTER_MASK_SLICE = slice(UPSTREAM, UPSTREAM + 3)
TERMINATOR_MASK_SLICE = slice(
    UPSTREAM + (DOWNSTREAM * 2) + 17,
    UPSTREAM + (DOWNSTREAM * 2) + 20,
)

MRNA_COLS = [
    "UTR5LEN", "CDSLEN", "INTRONLEN", "UTR3LEN",
    "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY",
]

SPECIES_ORDER = ["Zmay", "Sbic", "Slyc", "Atha"]
TISSUE_ORDER = ["leaf", "root"]

TASK_ORDER = [
    "Atha_leaf", "Atha_root",
    "Sbic_leaf", "Sbic_root",
    "Slyc_leaf", "Slyc_root",
    "Zmay_leaf", "Zmay_root",
]

SPECIES_CONFIG = {
    "Atha": {
        "display_name": "Arabidopsis thaliana",
        "fasta_glob": "Arabidopsis_thaliana*.fa",
        "gtf_glob": "Arabidopsis_thaliana*.gtf",
        "mrna_file": "Atha_mRNA_features_common.csv",
        "valid_chroms": [str(i) for i in range(1, 6)],
    },
    "Sbic": {
        "display_name": "Sorghum bicolor",
        "fasta_glob": "Sorghum_bicolor*.fa",
        "gtf_glob": "Sorghum_bicolor*.gtf",
        "mrna_file": "Sbic_mRNA_features_common.csv",
        "valid_chroms": [str(i) for i in range(1, 11)],
    },
    "Slyc": {
        "display_name": "Solanum lycopersicum",
        "fasta_glob": "Solanum_lycopersicum*.fa",
        "gtf_glob": "Solanum_lycopersicum*.gtf",
        "mrna_file": "Slyc_mRNA_features_common.csv",
        "valid_chroms": [str(i) for i in range(1, 13)],
    },
    "Zmay": {
        "display_name": "Zea mays",
        "fasta_glob": "Zea_mays*.fa",
        "gtf_glob": "Zea_mays*.gtf",
        "mrna_file": "Zmay_mRNA_features_common.csv",
        "valid_chroms": [str(i) for i in range(1, 11)],
    },
}

VALIDATION_KEY_ALIASES = {
    "Atha": ["Atha", "atha", "ara", "ARA", "arabidopsis", "Arabidopsis", "arabidopsis_thaliana", "Arabidopsis_thaliana", "ath"],
    "Sbic": ["Sbic", "sbic", "sor", "SOR", "sorghum", "Sorghum", "sorghum_bicolor", "Sorghum_bicolor", "sbi"],
    "Slyc": ["Slyc", "slyc", "sol", "SOL", "tomato", "Tomato", "solanum", "Solanum", "solanum_lycopersicum", "Solanum_lycopersicum", "sly"],
    "Zmay": ["Zmay", "zmay", "zea", "ZEA", "Zea", "maize", "Maize", "zea_mays", "Zea_mays", "zma"],
}

EXPRESSION_VALUE_COL_CANDIDATES = [
    "log(MaxTPM+1)",
    "logMaxTPM",
    "log_MaxTPM_plus_1",
    "log_max_tpm_plus_1",
    "log_max_tpm",
]

# Cross-species target-set choice:
# all_labeled: use all available low/high genes in target species.
# whitelist_only: use only target genes in validation_genes.pickle whitelist.
# Diagonal same-species evaluation always uses held-out chromosome ∩ whitelist.
CROSS_SPECIES_TARGET_MODE = "all_labeled"

SAVE_PREDICTIONS = True
SAVE_ROC_POINTS = True
SAVE_EXCEL = True
MAKE_HEATMAPS = True
PREDICTION_CSV_SUFFIX = ".csv.gz"


def set_global_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] TensorFlow GPUs: {gpus}")
    except Exception as e:
        print(f"[WARN] GPU memory growth setup failed: {e}")


set_global_determinism(SEED)
configure_tensorflow()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_gene_id(gene_id: Any) -> str:
    return str(gene_id).strip()


def natural_sort_key(value: Any) -> list:
    text = str(value)
    parts = re.split(r"(\d+)", text)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    return key


def get_terminal_digit_token(value: str) -> Optional[str]:
    m = re.search(r"(\d+)$", str(value))
    if m:
        return str(int(m.group(1)))
    return None


def resolve_task_parts(task_name: str) -> Tuple[str, str]:
    species_key, tissue = task_name.split("_", 1)
    if species_key not in SPECIES_CONFIG:
        raise ValueError(f"Unknown species key: {species_key}")
    if tissue not in {"leaf", "root"}:
        raise ValueError(f"Unknown tissue: {tissue}")
    return species_key, tissue


def resolve_model_root() -> str:
    for root in MODEL_OUTPUT_FALLBACKS:
        if root and os.path.isdir(root):
            ok = any(os.path.isdir(os.path.join(root, t)) for t in TASK_ORDER)
            if ok:
                print(f"[INFO] Using model output root: {root}")
                return root
    raise FileNotFoundError("Could not find a valid model output root. Tried:\n" + "\n".join(MODEL_OUTPUT_FALLBACKS))


def auto_read_table(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(file_path, sep="\t")
    return pd.read_csv(file_path, sep=None, engine="python")


def standardize_gene_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gene_id" in df.columns:
        pass
    elif df.index.name == "gene_id":
        df = df.reset_index()
    elif len(df.columns) > 0 and str(df.columns[0]).lower() in {"gene", "geneid", "gene_id", "id", "index", "unnamed: 0"}:
        df = df.rename(columns={df.columns[0]: "gene_id"})
    else:
        df = df.reset_index().rename(columns={"index": "gene_id"})
    df["gene_id"] = df["gene_id"].astype(str).map(normalize_gene_id)
    return df


def find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def find_expression_value_column(df: pd.DataFrame) -> str:
    col = find_first_matching_column(df, EXPRESSION_VALUE_COL_CANDIDATES)
    if col is not None:
        return col
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if "log" in cl and "tpm" in cl:
            candidates.append(c)
    if candidates:
        return candidates[0]
    raise ValueError(f"No expression value column found. Candidates={EXPRESSION_VALUE_COL_CANDIDATES}; columns={list(df.columns)}")


def resolve_unique_file(base_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matched: {os.path.join(base_dir, pattern)}")
    if len(matches) > 1:
        print(f"[WARN] Pattern {pattern} matched multiple files; using first: {matches[0]}")
    return matches[0]


def reverse_complement_onehot(encoded_seq: np.ndarray) -> np.ndarray:
    return encoded_seq[::-1, ::-1]


def onehot(seq: str) -> np.ndarray:
    code = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
        "unk": [0, 0, 0, 0],
    }
    seq = str(seq).upper()
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        encoded[i, :] = code.get(nt, code["unk"])
    return encoded


def resolve_validation_pickle_path() -> str:
    for path in VALIDATION_PICKLE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("validation_genes.pickle not found. Tried:\n" + "\n".join(VALIDATION_PICKLE_CANDIDATES))


def _normalize_gene_collection(obj: Any) -> set:
    if obj is None:
        return set()
    if isinstance(obj, dict):
        genes = set()
        for v in obj.values():
            genes.update(_normalize_gene_collection(v))
        return genes
    if isinstance(obj, (set, list, tuple, np.ndarray, pd.Index, pd.Series)):
        return {normalize_gene_id(x) for x in obj if pd.notna(x)}
    return set()


def load_validation_gene_whitelist(species_key: str) -> Tuple[set, str, str]:
    pickle_path = resolve_validation_pickle_path()
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"validation_genes.pickle top-level object should be dict, got {type(payload).__name__}")
    aliases = VALIDATION_KEY_ALIASES.get(species_key, [species_key])
    matched_key = None
    gene_set = set()
    for alias in aliases:
        if alias in payload:
            candidate_set = _normalize_gene_collection(payload[alias])
            if candidate_set:
                matched_key = alias
                gene_set = candidate_set
                break
    if not gene_set:
        lower_map = {str(k).strip().lower(): k for k in payload.keys()}
        for alias in aliases:
            real_key = lower_map.get(str(alias).strip().lower())
            if real_key is not None:
                candidate_set = _normalize_gene_collection(payload[real_key])
                if candidate_set:
                    matched_key = real_key
                    gene_set = candidate_set
                    break
    if not gene_set:
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            nested_keys = {str(k).strip().lower(): k for k in value.keys()}
            for alias in aliases:
                nested_real_key = nested_keys.get(str(alias).strip().lower())
                if nested_real_key is not None:
                    candidate_set = _normalize_gene_collection(value[nested_real_key])
                    if candidate_set:
                        matched_key = f"{key}->{nested_real_key}"
                        gene_set = candidate_set
                        break
            if gene_set:
                break
    if not gene_set:
        raise KeyError(f"No validation whitelist key found for species {species_key}. aliases={aliases}; keys={list(payload.keys())[:20]}")
    print(f"[INFO] {species_key}: validation whitelist key={matched_key}, genes={len(gene_set)}")
    return gene_set, pickle_path, str(matched_key)


def resolve_species_files(species_key: str) -> dict:
    cfg = SPECIES_CONFIG[species_key]
    fasta_path = resolve_unique_file(DATA_DIR, cfg["fasta_glob"])
    gtf_path = resolve_unique_file(DATA_DIR, cfg["gtf_glob"])
    mrna_path = os.path.join(DATA_DIR, cfg["mrna_file"])
    if not os.path.exists(mrna_path):
        raise FileNotFoundError(f"mRNA feature file not found: {mrna_path}")
    return {"fasta_path": fasta_path, "gtf_path": gtf_path, "mrna_path": mrna_path}


def resolve_expression_file(task_name: str) -> str:
    expr_path = os.path.join(DATA_DIR, f"{task_name}_expression_label_common.csv")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"Expression label file not found: {expr_path}")
    return expr_path


def load_expression_labels(task_name: str) -> pd.DataFrame:
    expr_path = resolve_expression_file(task_name)
    expr_df = auto_read_table(expr_path)
    expr_df = standardize_gene_id_column(expr_df)
    value_col = find_expression_value_column(expr_df)
    expr_df[value_col] = pd.to_numeric(expr_df[value_col], errors="coerce")
    expr_df = expr_df.dropna(subset=[value_col]).copy()
    q25 = np.percentile(expr_df[value_col].values, 25)
    q75 = np.percentile(expr_df[value_col].values, 75)
    def _to_label(v):
        if v <= q25:
            return 0
        if v >= q75:
            return 1
        return 2
    expr_df["label"] = expr_df[value_col].map(_to_label).astype(int)
    expr_df = expr_df[["gene_id", value_col, "label"]].copy().rename(columns={value_col: "expression_value"})
    print(f"[INFO] {task_name}: expression column={value_col}, Q25={q25:.6f}, Q75={q75:.6f}, counts={expr_df['label'].value_counts().sort_index().to_dict()}")
    return expr_df


def load_species_feature_table(species_key: str, validation_gene_set: set) -> pd.DataFrame:
    cfg = SPECIES_CONFIG[species_key]
    paths = resolve_species_files(species_key)
    print("\n" + "=" * 80)
    print(f"[INFO] Rebuilding species base features: {species_key} | {cfg['display_name']}")
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    print("=" * 80)
    fasta = Fasta(paths["fasta_path"], as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_models = pr.read_gtf(paths["gtf_path"], as_df=True)
    gene_models = gene_models[gene_models["Feature"] == "gene"].copy()
    biotype_col = find_first_matching_column(gene_models, ["gene_biotype", "gene_type", "biotype"])
    if biotype_col is not None:
        gene_models = gene_models[gene_models[biotype_col].astype(str) == "protein_coding"].copy()
    else:
        print(f"[WARN] {species_key}: no gene_biotype/gene_type/biotype column; skip protein_coding filter.")
    required_gtf_cols = ["Chromosome", "Start", "End", "Strand", "gene_id"]
    missing_gtf_cols = [c for c in required_gtf_cols if c not in gene_models.columns]
    if missing_gtf_cols:
        raise ValueError(f"GTF missing required columns: {missing_gtf_cols}")
    gene_models = gene_models[required_gtf_cols].copy()
    gene_models["gene_id"] = gene_models["gene_id"].astype(str).map(normalize_gene_id)
    gene_models["Chromosome"] = gene_models["Chromosome"].astype(str)
    valid_chroms = set(cfg.get("valid_chroms") or [])
    if valid_chroms:
        filtered = gene_models[gene_models["Chromosome"].isin(valid_chroms)].copy()
        if len(filtered) == 0:
            print(f"[WARN] {species_key}: chromosome filter removed all genes; using all chromosomes. examples={sorted(gene_models['Chromosome'].astype(str).unique())[:10]}")
        else:
            gene_models = filtered
    if gene_models["gene_id"].duplicated().any():
        n_dup = int(gene_models["gene_id"].duplicated().sum())
        print(f"[WARN] {species_key}: duplicated gene_id rows={n_dup}; keeping first occurrence.")
        gene_models = gene_models.drop_duplicates(subset=["gene_id"], keep="first").copy()
    mrna_df = pd.read_csv(paths["mrna_path"])
    mrna_df = standardize_gene_id_column(mrna_df)
    missing_mrna_cols = [c for c in MRNA_COLS if c not in mrna_df.columns]
    if missing_mrna_cols:
        raise ValueError(f"mRNA feature file missing columns {missing_mrna_cols}: {paths['mrna_path']}")
    mrna_df = mrna_df[["gene_id"] + MRNA_COLS].dropna(subset=MRNA_COLS).copy()
    mrna_gene_set = set(mrna_df["gene_id"])
    seq_data = []
    total_gene_models = len(gene_models)
    for idx, row in enumerate(gene_models.itertuples(index=False), start=1):
        chrom = str(row.Chromosome)
        start = int(row.Start)
        end = int(row.End)
        strand = str(row.Strand)
        gene_id = normalize_gene_id(row.gene_id)
        if gene_id not in mrna_gene_set:
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
                encoded_seq = np.concatenate([onehot(prom_seq), np.zeros((GAP_LEN, 4), dtype=np.float32), onehot(term_seq)], axis=0)
            else:
                encoded_seq = np.concatenate([reverse_complement_onehot(onehot(prom_seq)), np.zeros((GAP_LEN, 4), dtype=np.float32), reverse_complement_onehot(onehot(term_seq))], axis=0)
        except Exception:
            continue
        if encoded_seq.shape[0] != SEQ_TOTAL_LEN:
            continue
        seq_data.append({"gene_id": gene_id, "chromosome": chrom, "encoded_seq": encoded_seq, "is_val_gene": gene_id in validation_gene_set})
        if idx % 5000 == 0:
            print(f"[{species_key}] scanned {idx}/{total_gene_models}; valid samples={len(seq_data)}")
    core_df = pd.DataFrame(seq_data)
    if core_df.empty:
        raise ValueError(f"{species_key}: no valid sequence samples were built.")
    base_df = core_df.merge(mrna_df, on="gene_id", how="inner").dropna(subset=MRNA_COLS).reset_index(drop=True)
    print(f"[INFO] {species_key}: base samples={len(base_df)}, whitelist hits={int(base_df['is_val_gene'].sum())}")
    print(base_df["chromosome"].value_counts().sort_index())
    return base_df


def make_task_dataframe(base_df: pd.DataFrame, expr_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    label_df = expr_df.loc[expr_df["label"] != 2, ["gene_id", "expression_value", "label"]].copy()
    merged_df = base_df.merge(label_df, on="gene_id", how="inner").dropna(subset=MRNA_COLS).reset_index(drop=True)
    print(f"[INFO] {task_name}: samples={len(merged_df)}, counts={merged_df['label'].value_counts().sort_index().to_dict()}, whitelist hits={int(merged_df['is_val_gene'].sum())}")
    return merged_df


def resolve_cv_chromosomes(data_df: pd.DataFrame, species_key: str) -> List[str]:
    actual_chroms = [str(x) for x in data_df["chromosome"].dropna().astype(str).unique().tolist()]
    actual_sorted = sorted(actual_chroms, key=natural_sort_key)
    preferred = [str(x) for x in SPECIES_CONFIG[species_key].get("valid_chroms", [])]
    if not preferred:
        return actual_sorted
    actual_by_token = {}
    for chrom in actual_sorted:
        token = get_terminal_digit_token(chrom)
        if token is not None and token not in actual_by_token:
            actual_by_token[token] = chrom
    resolved = []
    used = set()
    for chrom in preferred:
        if chrom in actual_sorted and chrom not in used:
            resolved.append(chrom)
            used.add(chrom)
            continue
        alias = actual_by_token.get(chrom)
        if alias is not None and alias not in used:
            resolved.append(alias)
            used.add(alias)
    if resolved:
        extras = [c for c in actual_sorted if c not in used]
        resolved.extend(extras)
        return resolved
    print(f"[WARN] {species_key}: preferred chromosomes {preferred} did not match actual names; using actual={actual_sorted}")
    return actual_sorted


def make_chromosome_split(df: pd.DataFrame, valid_chrom: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chrom_mask = df["chromosome"].astype(str) == str(valid_chrom)
    val_mask = chrom_mask & df["is_val_gene"].astype(bool)
    train_mask = ~chrom_mask
    return df.loc[train_mask].copy(), df.loc[val_mask].copy()


def balance_train_data(train_seqs, train_mrna, train_labels, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
    low_train = np.where(train_labels == 0)[0]
    high_train = np.where(train_labels == 1)[0]
    min_class = min(len(low_train), len(high_train))
    if min_class == 0:
        return train_seqs, train_mrna, train_labels
    selected_low = np.random.choice(low_train, min_class, replace=False)
    selected_high = np.random.choice(high_train, min_class, replace=False)
    x_seq_balanced = np.concatenate([np.take(train_seqs, selected_low, axis=0), np.take(train_seqs, selected_high, axis=0)], axis=0)
    x_mrna_balanced = np.concatenate([np.take(train_mrna, selected_low, axis=0), np.take(train_mrna, selected_high, axis=0)], axis=0)
    y_balanced = np.concatenate([np.take(train_labels, selected_low), np.take(train_labels, selected_high)], axis=0)
    x_seq_balanced, x_mrna_balanced, y_balanced = shuffle(x_seq_balanced, x_mrna_balanced, y_balanced, random_state=SEED)
    return x_seq_balanced, x_mrna_balanced, y_balanced


def fit_source_scaler_for_chrom_model(source_df: pd.DataFrame, valid_chrom: str, fold_round: int) -> Tuple[StandardScaler, dict]:
    train_df, val_df = make_chromosome_split(source_df, valid_chrom)
    if train_df.empty:
        raise ValueError(f"No training samples for valid_chrom={valid_chrom}")
    train_idx = train_df.index.to_numpy()
    X_seq = np.stack(source_df["encoded_seq"].values).astype(np.float32)
    X_mrna = source_df[MRNA_COLS].values.astype(np.float32)
    y = source_df["label"].values.astype(np.int32)
    X_seq_train = X_seq[train_idx].copy()
    X_mrna_train = X_mrna[train_idx].copy()
    y_train = y[train_idx]
    balance_seed = SEED + int(fold_round) - 1
    _, X_mrna_train_bal, y_train_bal = balance_train_data(X_seq_train, X_mrna_train, y_train, seed=balance_seed)
    scaler = StandardScaler()
    scaler.fit(X_mrna_train_bal)
    meta = {
        "valid_chromosome": str(valid_chrom),
        "fold_round": int(fold_round),
        "train_size": int(len(train_df)),
        "source_val_size": int(len(val_df)),
        "train_pos_raw": int((y_train == 1).sum()),
        "train_neg_raw": int((y_train == 0).sum()),
        "train_balanced_size": int(len(y_train_bal)),
        "train_pos_balanced": int((y_train_bal == 1).sum()),
        "train_neg_balanced": int((y_train_bal == 0).sum()),
        "balance_seed": int(balance_seed),
    }
    return scaler, meta


def apply_model_input_processing(df: pd.DataFrame, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq = np.stack(df["encoded_seq"].values).astype(np.float32)
    X_mrna = df[MRNA_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X_seq[:, PROMOTER_MASK_SLICE, :] = 0
    X_seq[:, TERMINATOR_MASK_SLICE, :] = 0
    X_mrna = scaler.transform(X_mrna).astype(np.float32)
    return X_seq, X_mrna, y


def parse_valid_chrom_from_model_path(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"_model_chr(.+?)\.keras$", name)
    if not m:
        raise ValueError(f"Cannot parse chromosome from model file name: {name}")
    return m.group(1)


def discover_saved_models_for_task(model_root: str, task_name: str) -> List[dict]:
    task_dir = os.path.join(model_root, task_name)
    paths = sorted(glob.glob(os.path.join(task_dir, MODEL_GLOB_PATTERN)), key=natural_sort_key)
    models = []
    for p in paths:
        chrom = parse_valid_chrom_from_model_path(p)
        models.append({"path": p, "valid_chromosome": str(chrom)})
    return models


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0:
        return {"N": 0, "TN": 0, "FP": 0, "FN": 0, "TP": 0, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan}
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "N": int(len(y_true)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_score),
    }


def select_target_df_for_evaluation(source_species: str, target_species: str, target_df: pd.DataFrame, valid_chrom: str) -> pd.DataFrame:
    if source_species == target_species:
        _, val_df = make_chromosome_split(target_df, valid_chrom)
        return val_df.copy()
    if CROSS_SPECIES_TARGET_MODE == "whitelist_only":
        return target_df[target_df["is_val_gene"].astype(bool)].copy()
    return target_df.copy()


def prediction_frame(df: pd.DataFrame, y_score: np.ndarray, y_pred: np.ndarray, meta: dict) -> pd.DataFrame:
    cols = ["gene_id", "chromosome", "label", "expression_value", "is_val_gene"]
    out = df[cols].copy()
    for k, v in meta.items():
        out[k] = v
    out["pred_prob"] = y_score
    out["pred_label"] = y_pred
    return out


def make_metric_matrix(agg_df: pd.DataFrame, tissue: str, metric: str) -> pd.DataFrame:
    mat = pd.DataFrame(index=SPECIES_ORDER, columns=SPECIES_ORDER, dtype=float)
    sub = agg_df[agg_df["tissue"] == tissue]
    for _, row in sub.iterrows():
        mat.loc[row["train_species"], row["test_species"]] = row[metric]
    return mat


def save_matrices_for_tissue(agg_df: pd.DataFrame, tissue: str, out_dir: str) -> Dict[str, str]:
    ensure_dir(out_dir)
    metrics = ["accuracy", "auc", "f1", "precision", "recall", "TN", "FP", "FN", "TP", "N"]
    matrix_paths = {}
    for metric in metrics:
        mat = make_metric_matrix(agg_df, tissue, metric)
        path = os.path.join(out_dir, f"{tissue}_cross_species_{metric}_matrix.csv")
        mat.to_csv(path, encoding="utf-8-sig")
        matrix_paths[metric] = path
    if SAVE_EXCEL:
        excel_path = os.path.join(out_dir, f"{tissue}_cross_species_all_matrices.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for metric in metrics:
                make_metric_matrix(agg_df, tissue, metric).to_excel(writer, sheet_name=metric[:31])
        matrix_paths["excel"] = excel_path
    return matrix_paths


def plot_heatmap_matrix(mat: pd.DataFrame, title: str, out_prefix: str, fmt: str = ".3f") -> Tuple[str, str]:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    values = mat.astype(float).values
    finite = values[np.isfinite(values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmin == vmax:
        vmin = max(0.0, vmin - 0.1)
        vmax = min(1.0, vmax + 0.1) if vmax <= 1 else vmax + 0.1
    im = ax.imshow(values, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns, fontsize=10)
    ax.set_yticklabels(mat.index, fontsize=10)
    ax.set_xlabel("Test species", fontsize=11)
    ax.set_ylabel("Train species", fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    mean_val = np.nanmean(values) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt = "NA" if not np.isfinite(val) else format(val, fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if np.isfinite(val) and val >= mean_val else "#333333")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    png = out_prefix + ".png"
    pdf = out_prefix + ".pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def build_roc_rows(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in pred_df.groupby(["tissue", "train_species", "test_species"]):
        tissue, train_sp, test_sp = keys
        y_true = sub["label"].astype(int).values
        y_score = sub["pred_prob"].astype(float).values
        auc = safe_auc(y_true, y_score)
        fpr, tpr = safe_roc_curve(y_true, y_score)
        rows.append(pd.DataFrame({"tissue": tissue, "train_species": train_sp, "test_species": test_sp, "fpr": fpr, "tpr": tpr, "auc": auc}))
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["tissue", "train_species", "test_species", "fpr", "tpr", "auc"])


def rebuild_all_task_data() -> Dict[str, pd.DataFrame]:
    print("\n" + "#" * 90)
    print("[STEP] Rebuilding all task data")
    print("#" * 90)
    validation_info = {}
    base_by_species = {}
    task_data = {}
    for sp in SPECIES_ORDER:
        gene_set, pickle_path, matched_key = load_validation_gene_whitelist(sp)
        validation_info[sp] = {"pickle_path": pickle_path, "matched_key": matched_key, "n_genes": len(gene_set)}
        base_by_species[sp] = load_species_feature_table(sp, gene_set)
    for task in TASK_ORDER:
        sp, _ = resolve_task_parts(task)
        expr_df = load_expression_labels(task)
        task_data[task] = make_task_dataframe(base_by_species[sp], expr_df, task)
    ensure_dir(OUT_DIR)
    with open(os.path.join(OUT_DIR, "validation_whitelist_info.json"), "w", encoding="utf-8") as f:
        json.dump(validation_info, f, ensure_ascii=False, indent=2)
    return task_data


def run_cross_species_prediction() -> dict:
    model_root = resolve_model_root()
    ensure_dir(OUT_DIR)
    task_data = rebuild_all_task_data()
    task_chrom_order = {}
    task_models = {}
    failures = []
    for task, df in task_data.items():
        sp, _ = resolve_task_parts(task)
        chrom_order = resolve_cv_chromosomes(df, sp)
        task_chrom_order[task] = chrom_order
        models = discover_saved_models_for_task(model_root, task)
        task_models[task] = models
        print(f"[INFO] {task}: discovered {len(models)} saved chromosome models.")
        for m in models:
            print(f"       chr={m['valid_chromosome']} | {m['path']}")
    all_metric_rows = []
    all_pred_frames = []
    for tissue in TISSUE_ORDER:
        print("\n" + "#" * 90)
        print(f"[STEP] Cross-species prediction for tissue={tissue}")
        print("#" * 90)
        for train_sp in SPECIES_ORDER:
            source_task = f"{train_sp}_{tissue}"
            source_df = task_data[source_task]
            chrom_order = task_chrom_order[source_task]
            chrom_to_round = {str(chrom): i + 1 for i, chrom in enumerate(chrom_order)}
            if not task_models[source_task]:
                failures.append({"source_task": source_task, "error": "no saved .keras models found"})
                continue
            for model_info in task_models[source_task]:
                model_path = model_info["path"]
                valid_chrom = str(model_info["valid_chromosome"])
                fold_round = int(chrom_to_round.get(valid_chrom, len(chrom_to_round) + 1))
                print("\n" + "-" * 90)
                print(f"[MODEL] source={source_task}, valid_chrom={valid_chrom}, fold_round={fold_round}")
                print(f"[MODEL] path={model_path}")
                try:
                    scaler, scaler_meta = fit_source_scaler_for_chrom_model(source_df=source_df, valid_chrom=valid_chrom, fold_round=fold_round)
                    K.clear_session()
                    model = load_model(model_path, compile=False)
                    for test_sp in SPECIES_ORDER:
                        target_task = f"{test_sp}_{tissue}"
                        target_df_all = task_data[target_task]
                        eval_df = select_target_df_for_evaluation(train_sp, test_sp, target_df_all, valid_chrom)
                        if eval_df.empty:
                            failures.append({"source_task": source_task, "target_task": target_task, "valid_chromosome": valid_chrom, "model_path": model_path, "error": "empty evaluation dataframe"})
                            continue
                        X_seq_test, X_mrna_test, y_test = apply_model_input_processing(eval_df, scaler)
                        y_score = model.predict([X_seq_test, X_mrna_test], batch_size=BATCH_SIZE, verbose=0).ravel()
                        y_pred = (y_score >= THRESHOLD).astype(int)
                        metrics = evaluate_binary(y_test, y_score, threshold=THRESHOLD)
                        metric_row = {
                            "tissue": tissue,
                            "train_species": train_sp,
                            "test_species": test_sp,
                            "source_task": source_task,
                            "target_task": target_task,
                            "valid_chromosome": valid_chrom,
                            "fold_round": fold_round,
                            "model_path": model_path,
                            "target_mode": "heldout_chromosome_whitelist" if train_sp == test_sp else CROSS_SPECIES_TARGET_MODE,
                            **scaler_meta,
                            **metrics,
                        }
                        all_metric_rows.append(metric_row)
                        if SAVE_PREDICTIONS:
                            pred_meta = {
                                "tissue": tissue,
                                "train_species": train_sp,
                                "test_species": test_sp,
                                "source_task": source_task,
                                "target_task": target_task,
                                "valid_chromosome": valid_chrom,
                                "fold_round": fold_round,
                                "model_path": model_path,
                                "target_mode": metric_row["target_mode"],
                            }
                            all_pred_frames.append(prediction_frame(eval_df, y_score, y_pred, pred_meta))
                        print(f"[RESULT] {train_sp}->{test_sp} | {tissue} | chr={valid_chrom} | N={metrics['N']} | AUC={metrics['auc']:.4f} | ACC={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | TN/FP/FN/TP={metrics['TN']}/{metrics['FP']}/{metrics['FN']}/{metrics['TP']}")
                        del X_seq_test, X_mrna_test, y_test, y_score, y_pred
                        gc.collect()
                    del model
                    K.clear_session()
                    gc.collect()
                except Exception as e:
                    err = traceback.format_exc()
                    print(f"[ERROR] source={source_task}, valid_chrom={valid_chrom}: {repr(e)}")
                    failures.append({"source_task": source_task, "valid_chromosome": valid_chrom, "model_path": model_path, "error": repr(e), "traceback": err})
                    K.clear_session()
                    gc.collect()
                    continue
    metrics_by_model_df = pd.DataFrame(all_metric_rows)
    metrics_by_model_path = os.path.join(OUT_DIR, "cross_species_metrics_by_chrom_model.csv")
    metrics_by_model_df.to_csv(metrics_by_model_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {metrics_by_model_path}")
    if SAVE_PREDICTIONS and all_pred_frames:
        pred_df = pd.concat(all_pred_frames, ignore_index=True)
        pred_path = os.path.join(OUT_DIR, f"cross_species_predictions{PREDICTION_CSV_SUFFIX}")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig", compression="gzip")
        print(f"[SAVE] {pred_path}")
        if SAVE_ROC_POINTS:
            roc_df = build_roc_rows(pred_df)
            roc_path = os.path.join(OUT_DIR, "cross_species_roc_points_aggregated.csv")
            roc_df.to_csv(roc_path, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {roc_path}")
    else:
        pred_df = pd.DataFrame()
        pred_path = None
        roc_path = None
    agg_rows = []
    if not pred_df.empty:
        for keys, sub in pred_df.groupby(["tissue", "train_species", "test_species"]):
            tissue, train_sp, test_sp = keys
            y_true = sub["label"].astype(int).values
            y_score = sub["pred_prob"].astype(float).values
            metrics = evaluate_binary(y_true, y_score, threshold=THRESHOLD)
            agg_rows.append({"tissue": tissue, "train_species": train_sp, "test_species": test_sp, "aggregation": "concatenated_predictions_across_chromosome_models", **metrics})
    else:
        for keys, sub in metrics_by_model_df.groupby(["tissue", "train_species", "test_species"]):
            tissue, train_sp, test_sp = keys
            agg_rows.append({
                "tissue": tissue,
                "train_species": train_sp,
                "test_species": test_sp,
                "aggregation": "mean_metrics_by_chromosome_model",
                "N": int(sub["N"].sum()),
                "TN": int(sub["TN"].sum()),
                "FP": int(sub["FP"].sum()),
                "FN": int(sub["FN"].sum()),
                "TP": int(sub["TP"].sum()),
                "accuracy": float(sub["accuracy"].mean(skipna=True)),
                "precision": float(sub["precision"].mean(skipna=True)),
                "recall": float(sub["recall"].mean(skipna=True)),
                "f1": float(sub["f1"].mean(skipna=True)),
                "auc": float(sub["auc"].mean(skipna=True)),
            })
    agg_df = pd.DataFrame(agg_rows)
    agg_path = os.path.join(OUT_DIR, "cross_species_metrics_aggregated.csv")
    agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {agg_path}")
    mean_std_path = None
    if not metrics_by_model_df.empty:
        mean_std_rows = []
        metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "N", "TN", "FP", "FN", "TP"]
        for keys, sub in metrics_by_model_df.groupby(["tissue", "train_species", "test_species"]):
            tissue, train_sp, test_sp = keys
            row = {"tissue": tissue, "train_species": train_sp, "test_species": test_sp, "n_chrom_models": int(sub["model_path"].nunique())}
            for m in metric_cols:
                row[f"{m}_mean_by_model"] = sub[m].mean(skipna=True)
                row[f"{m}_std_by_model"] = sub[m].std(skipna=True)
            mean_std_rows.append(row)
        mean_std_df = pd.DataFrame(mean_std_rows)
        mean_std_path = os.path.join(OUT_DIR, "cross_species_metrics_mean_std_by_chrom_model.csv")
        mean_std_df.to_csv(mean_std_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {mean_std_path}")
    matrix_dir = os.path.join(OUT_DIR, "matrices")
    heatmap_dir = os.path.join(OUT_DIR, "heatmaps")
    ensure_dir(matrix_dir)
    ensure_dir(heatmap_dir)
    matrix_paths = {}
    heatmap_paths = []
    for tissue in TISSUE_ORDER:
        tissue_matrix_dir = os.path.join(matrix_dir, tissue)
        matrix_paths[tissue] = save_matrices_for_tissue(agg_df, tissue, tissue_matrix_dir)
        if MAKE_HEATMAPS:
            for metric in ["auc", "accuracy", "f1"]:
                mat = make_metric_matrix(agg_df, tissue, metric)
                out_prefix = os.path.join(heatmap_dir, f"{tissue}_cross_species_{metric}_heatmap")
                heatmap_paths.extend(plot_heatmap_matrix(mat, title=f"{tissue.capitalize()} cross-species {metric.upper()}", out_prefix=out_prefix, fmt=".3f"))
    failures_path = None
    if failures:
        failures_df = pd.DataFrame(failures)
        failures_path = os.path.join(OUT_DIR, "cross_species_prediction_failures.csv")
        failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] Failures saved to {failures_path}")
    manifest = {
        "model_root": model_root,
        "output_dir": OUT_DIR,
        "model_glob_pattern": MODEL_GLOB_PATTERN,
        "data_dir": DATA_DIR,
        "species_order": SPECIES_ORDER,
        "tissue_order": TISSUE_ORDER,
        "target_mode_cross_species": CROSS_SPECIES_TARGET_MODE,
        "target_mode_diagonal": "heldout_chromosome_whitelist",
        "threshold": THRESHOLD,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_TOTAL_LEN,
        "mask_slices": {"promoter": [PROMOTER_MASK_SLICE.start, PROMOTER_MASK_SLICE.stop], "terminator": [TERMINATOR_MASK_SLICE.start, TERMINATOR_MASK_SLICE.stop]},
        "outputs": {
            "metrics_by_chrom_model": metrics_by_model_path,
            "metrics_aggregated": agg_path,
            "metrics_mean_std_by_chrom_model": mean_std_path,
            "predictions": pred_path,
            "roc_points": roc_path,
            "matrices": matrix_paths,
            "heatmaps": heatmap_paths,
            "failures": failures_path,
        },
    }
    manifest_path = os.path.join(OUT_DIR, "cross_species_prediction_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {manifest_path}")
    print("\n" + "=" * 90)
    print("[DONE] Cross-species prediction completed.")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 90)
    return manifest


def main():
    run_cross_species_prediction()


if __name__ == "__main__":
    main()

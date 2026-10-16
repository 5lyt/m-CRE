import os
import gc
import re
import json
import glob
import time
import random
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pyranges as pr
from pyfaidx import Fasta
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.stats import variation

warnings.filterwarnings("ignore")

# =========================
# 0. 全局配置
# =========================
BASE_PATH = "/home"
DATA_DIR = os.path.join(BASE_PATH, "data")
MODEL_OUTPUT_ROOT = os.path.join(BASE_PATH, "output_DeepCRE_mRNA_BiLSTM_chr")
INTERP_ROOT = os.path.join(MODEL_OUTPUT_ROOT, "interpretability_deepshap")

VALIDATION_PICKLE_CANDIDATES = [
    os.path.join(DATA_DIR, "validation_genes.pickle"),
    os.path.join(BASE_PATH, "validation_genes.pickle"),
    os.path.join(BASE_PATH, "data", "validation_genes.pickle"),
]

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

MRNA_COLS = [
    "UTR5LEN", "CDSLEN", "INTRONLEN", "UTR3LEN",
    "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY"
]

EXPRESSION_VALUE_COL_CANDIDATES = [
    "log(MaxTPM+1)",
    "logMaxTPM",
    "log_MaxTPM_plus_1",
    "log_max_tpm_plus_1",
    "log_max_tpm",
]

SEED = 42
UPSTREAM = 1000
DOWNSTREAM = 500
GAP_LEN = 20
SEQ_TOTAL_LEN = 2 * (UPSTREAM + DOWNSTREAM) + GAP_LEN
PROMOTER_MASK_SLICE = slice(UPSTREAM, UPSTREAM + 3)
TERMINATOR_MASK_SLICE = slice(UPSTREAM + (DOWNSTREAM * 2) + 17, UPSTREAM + (DOWNSTREAM * 2) + 20)

# -------- DeepSHAP 计算参数（可按显存调小） --------
BACKGROUND_PER_CHROM = 64
EXPLAIN_PER_CHROM = 256
EXPLAIN_BATCH_SIZE = 64
MAX_EXPLAIN_BACKGROUND_FROM_VAL = False
ONLY_CORRECT_PREDICTIONS = False
USE_ABS_SHAP = True
SAVE_PER_CHROM_RAW = False

# 画图配置
REGION_ANNOTATIONS = [
    (0, 1000, "Promoter Region", "#fff7bc"),
    (1000, 1500, "5' UTR", "#fdd49e"),
    (1500, 1520, "N Spacer", "#d9d9d9"),
    (1520, 2020, "Coding Sequence", "#fdbb84"),
    (2020, 3020, "3' UTR", "#fc8d59"),
]
NUCLEOTIDE_ORDER = ["A", "T", "C", "G"]
NUCLEOTIDE_COLORS = {
    "A": "#E64B35",
    "T": "#4DBBD5",
    "C": "#00A087",
    "G": "#3C5488",
}

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# =========================
# 1. 基础工具
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_gene_id(gene_id):
    return str(gene_id).strip()


def resolve_task_parts(task_name: str):
    species_key, tissue = task_name.split("_", 1)
    return species_key, tissue


def auto_read_table(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=None, engine="python")


def standardize_gene_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gene_id" in df.columns:
        pass
    elif df.index.name == "gene_id":
        df = df.reset_index()
    elif len(df.columns) > 0 and str(df.columns[0]).lower() in {"gene", "geneid", "id", "index", "unnamed: 0"}:
        df = df.rename(columns={df.columns[0]: "gene_id"})
    else:
        df = df.reset_index().rename(columns={"index": "gene_id"})
    df["gene_id"] = df["gene_id"].astype(str).map(normalize_gene_id)
    return df


def find_first_matching_column(df: pd.DataFrame, candidates):
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
        if ("log" in cl) and ("tpm" in cl):
            candidates.append(c)
    if candidates:
        return candidates[0]
    raise ValueError(f"表达量文件中未找到可用列: {list(df.columns)}")


def resolve_unique_file(base_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"未找到匹配文件: {os.path.join(base_dir, pattern)}")
    return matches[0]


def reverse_complement_onehot(encoded_seq: np.ndarray) -> np.ndarray:
    return encoded_seq[::-1, ::-1]


def onehot(seq: str) -> np.ndarray:
    code = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "unk": [0, 0, 0, 0],
    }
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        encoded[i, :] = code.get(nt, code["unk"])
    return encoded


def natural_sort_key(value):
    text = str(value)
    parts = re.split(r"(\d+)", text)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    return key


def get_terminal_digit_token(value: str):
    m = re.search(r"(\d+)$", str(value))
    if m:
        return str(int(m.group(1)))
    return None


def resolve_cv_chromosomes(data_df: pd.DataFrame, species_key: str):
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
    return actual_sorted

# =========================
# 2. 数据解析
# =========================
def resolve_validation_pickle_path() -> str:
    for path in VALIDATION_PICKLE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("未找到 validation_genes.pickle")


def _normalize_gene_collection(obj):
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


def load_validation_gene_whitelist(species_key: str):
    pickle_path = resolve_validation_pickle_path()
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)
    aliases = VALIDATION_KEY_ALIASES.get(species_key, [species_key])
    matched_key = None
    gene_set = set()

    for alias in aliases:
        if alias in payload:
            candidate = _normalize_gene_collection(payload[alias])
            if candidate:
                matched_key = alias
                gene_set = candidate
                break

    if not gene_set:
        lower_map = {str(k).strip().lower(): k for k in payload.keys()}
        for alias in aliases:
            real_key = lower_map.get(str(alias).strip().lower())
            if real_key is not None:
                candidate = _normalize_gene_collection(payload[real_key])
                if candidate:
                    matched_key = real_key
                    gene_set = candidate
                    break

    if not gene_set:
        raise KeyError(f"在 validation_genes.pickle 中未找到 {species_key} 对应白名单")
    return gene_set, pickle_path, matched_key


def resolve_species_files(species_key: str) -> dict:
    cfg = SPECIES_CONFIG[species_key]
    return {
        "fasta_path": resolve_unique_file(DATA_DIR, cfg["fasta_glob"]),
        "gtf_path": resolve_unique_file(DATA_DIR, cfg["gtf_glob"]),
        "mrna_path": os.path.join(DATA_DIR, cfg["mrna_file"]),
    }


def resolve_expression_file(task_name: str) -> str:
    expr_path = os.path.join(DATA_DIR, f"{task_name}_expression_label_common.csv")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(expr_path)
    return expr_path


def load_expression_labels(task_name: str) -> pd.DataFrame:
    expr_df = auto_read_table(resolve_expression_file(task_name))
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
    expr_df = expr_df[["gene_id", value_col, "label"]].copy()
    expr_df = expr_df.rename(columns={value_col: "expression_value"})
    return expr_df


def load_species_feature_table(species_key: str, validation_gene_set: set) -> pd.DataFrame:
    paths = resolve_species_files(species_key)
    fasta = Fasta(paths["fasta_path"], as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_models = pr.read_gtf(paths["gtf_path"], as_df=True)
    gene_models = gene_models[gene_models["Feature"] == "gene"].copy()

    biotype_col = find_first_matching_column(gene_models, ["gene_biotype", "gene_type", "biotype"])
    if biotype_col is not None:
        gene_models = gene_models[gene_models[biotype_col].astype(str) == "protein_coding"].copy()

    gene_models = gene_models[["Chromosome", "Start", "End", "Strand", "gene_id"]].copy()
    gene_models["gene_id"] = gene_models["gene_id"].astype(str).map(normalize_gene_id)
    gene_models["Chromosome"] = gene_models["Chromosome"].astype(str)

    cfg = SPECIES_CONFIG[species_key]
    valid_chroms = set(cfg.get("valid_chroms") or [])
    filtered_gene_models = gene_models[gene_models["Chromosome"].isin(valid_chroms)].copy()
    if len(filtered_gene_models) > 0:
        gene_models = filtered_gene_models

    gene_models = gene_models.drop_duplicates(subset=["gene_id"], keep="first").copy()

    mrna_df = pd.read_csv(paths["mrna_path"])
    mrna_df = standardize_gene_id_column(mrna_df)
    mrna_df = mrna_df[["gene_id"] + MRNA_COLS].dropna(subset=MRNA_COLS).copy()
    mrna_gene_set = set(mrna_df["gene_id"])

    seq_data = []
    for row in gene_models.itertuples(index=False):
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
                encoded_seq = np.concatenate([
                    onehot(prom_seq),
                    np.zeros((GAP_LEN, 4), dtype=np.float32),
                    onehot(term_seq),
                ], axis=0)
            else:
                encoded_seq = np.concatenate([
                    reverse_complement_onehot(onehot(prom_seq)),
                    np.zeros((GAP_LEN, 4), dtype=np.float32),
                    reverse_complement_onehot(onehot(term_seq)),
                ], axis=0)
        except Exception:
            continue

        if encoded_seq.shape[0] != SEQ_TOTAL_LEN:
            continue

        seq_data.append({
            "gene_id": gene_id,
            "chromosome": chrom,
            "encoded_seq": encoded_seq,
            "is_val_gene": gene_id in validation_gene_set,
        })

    core_df = pd.DataFrame(seq_data)
    base_df = core_df.merge(mrna_df, on="gene_id", how="inner")
    base_df = base_df.dropna(subset=MRNA_COLS).copy().reset_index(drop=True)
    return base_df


def make_task_dataframe(base_df: pd.DataFrame, expr_df: pd.DataFrame) -> pd.DataFrame:
    label_df = expr_df.loc[expr_df["label"] != 2, ["gene_id", "expression_value", "label"]].copy()
    merged_df = base_df.merge(label_df, on="gene_id", how="inner")
    merged_df = merged_df.dropna(subset=MRNA_COLS).copy().reset_index(drop=True)
    return merged_df


def make_chromosome_split(df: pd.DataFrame, valid_chrom: str):
    chrom_mask = df["chromosome"].astype(str) == str(valid_chrom)
    val_mask = chrom_mask & df["is_val_gene"].astype(bool)
    train_mask = ~chrom_mask
    return df.loc[train_mask].copy(), df.loc[val_mask].copy()


def balance_train_data(train_seqs, train_mrna, train_labels):
    low_train = np.where(train_labels == 0)[0]
    high_train = np.where(train_labels == 1)[0]
    min_class = min(len(low_train), len(high_train))
    if min_class == 0:
        return train_seqs, train_mrna, train_labels
    selected_low = np.random.choice(low_train, min_class, replace=False)
    selected_high = np.random.choice(high_train, min_class, replace=False)
    x_seq_balanced = np.concatenate([
        np.take(train_seqs, selected_low, axis=0),
        np.take(train_seqs, selected_high, axis=0),
    ], axis=0)
    x_mrna_balanced = np.concatenate([
        np.take(train_mrna, selected_low, axis=0),
        np.take(train_mrna, selected_high, axis=0),
    ], axis=0)
    y_balanced = np.concatenate([
        np.take(train_labels, selected_low, axis=0),
        np.take(train_labels, selected_high, axis=0),
    ], axis=0)
    x_seq_balanced, x_mrna_balanced, y_balanced = shuffle(
        x_seq_balanced, x_mrna_balanced, y_balanced, random_state=SEED
    )
    return x_seq_balanced, x_mrna_balanced, y_balanced

# =========================
# 3. DeepSHAP 计算
# =========================
def import_shap():
    try:
        import shap  # noqa
        return shap
    except Exception as e:
        raise ImportError(
            "当前环境无法导入 shap。请先安装 shap，例如: pip install shap==0.45.1"
        ) from e


def choose_sample_indices(y: np.ndarray, max_n: int, random_state: int = SEED):
    if len(y) <= max_n:
        return np.arange(len(y))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=max_n, random_state=random_state)
        idx = next(splitter.split(np.zeros(len(y)), y))[1]
        return np.sort(idx)
    except Exception:
        idx = np.random.RandomState(random_state).choice(len(y), size=max_n, replace=False)
        return np.sort(idx)


def _canonicalize_shap_array(arr, input_name: str):
    """将不同 shap/tf 版本下的输出统一为当前脚本可用形状。

    期望:
    - 序列输入: (n_samples, 3020, 4)
    - mRNA 输入: (n_samples, 8)

    常见异常:
    - 单输出模型会多出一个尾部输出维: (..., 1)
    - 某些版本返回 list/tuple 嵌套 ndarray
    """
    arr = np.asarray(arr)

    # 单输出二分类模型常见: (n, ..., 1)
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)

    if input_name == "seq":
        if arr.ndim != 3:
            raise ValueError(f"序列 SHAP 维度异常，期望 3 维 (n, 3020, 4)，实际 {arr.shape}")
        if arr.shape[-1] != 4:
            raise ValueError(f"序列 SHAP 最后一维应为 4 个碱基通道，实际 {arr.shape}")
    elif input_name == "mrna":
        if arr.ndim != 2:
            raise ValueError(f"mRNA SHAP 维度异常，期望 2 维 (n, 8)，实际 {arr.shape}")
    return arr


def normalize_shap_outputs(shap_values):
    """兼容多种 shap 输出形态，并消除单输出模型的尾部 singleton 维度。"""
    seq_sv, mrna_sv = None, None

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            seq_sv, mrna_sv = shap_values[0], shap_values[1]
        elif len(shap_values) == 1 and isinstance(shap_values[0], list) and len(shap_values[0]) == 2:
            seq_sv, mrna_sv = shap_values[0][0], shap_values[0][1]
        elif len(shap_values) == 1 and isinstance(shap_values[0], np.ndarray):
            raise ValueError("检测到单输入 SHAP 输出，和当前双输入模型不匹配")
    elif isinstance(shap_values, tuple) and len(shap_values) == 2:
        seq_sv, mrna_sv = shap_values
    else:
        # 某些 shap 版本可能返回 Explanation；这里显式报错，避免静默错用
        raise ValueError(f"无法识别 shap_values 输出类型: {type(shap_values)}")

    seq_sv = _canonicalize_shap_array(seq_sv, "seq")
    mrna_sv = _canonicalize_shap_array(mrna_sv, "mrna")
    return seq_sv, mrna_sv


def compute_deepshap_for_chrom(model, X_seq_bg, X_mrna_bg, X_seq_exp, X_mrna_exp):
    shap = import_shap()
    explainer = shap.GradientExplainer(model, [X_seq_bg, X_mrna_bg])
    seq_parts = []
    mrna_parts = []
    for start in range(0, len(X_seq_exp), EXPLAIN_BATCH_SIZE):
        end = start + EXPLAIN_BATCH_SIZE
        sv = explainer.shap_values([X_seq_exp[start:end], X_mrna_exp[start:end]])
        seq_sv, mrna_sv = normalize_shap_outputs(sv)
        seq_parts.append(np.array(seq_sv))
        mrna_parts.append(np.array(mrna_sv))
    seq_sv = np.concatenate(seq_parts, axis=0)
    mrna_sv = np.concatenate(mrna_parts, axis=0)
    return seq_sv, mrna_sv


def get_correct_prediction_mask(model, X_seq, X_mrna, y_true):
    y_prob = model.predict([X_seq, X_mrna], batch_size=128, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred == y_true, y_prob, y_pred

# =========================
# 4. 可视化与保存
# =========================
def add_region_spans(ax, y_text_factor=0.92):
    ymax = ax.get_ylim()[1]
    for start, end, label, color in REGION_ANNOTATIONS:
        ax.axvspan(start, end, alpha=0.2, color=color, zorder=0)
        ax.text((start + end) / 2, ymax * y_text_factor, label, ha="center", va="center", fontsize=9)


def save_sequence_importance_plot(seq_df: pd.DataFrame, out_png: str, out_pdf: str, title: str, column: str = "Total_SHAP"):
    plt.figure(figsize=(18, 5))
    sns.lineplot(x="Position", y=column, data=seq_df, linewidth=1.5, errorbar=None)
    ax = plt.gca()
    add_region_spans(ax)
    plt.title(title)
    plt.xlabel("Genomic Position (bp)")
    plt.ylabel(column)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def save_nucleotide_plot(seq_df: pd.DataFrame, out_png: str, out_pdf: str, title: str):
    plt.figure(figsize=(18, 5))
    for nt in NUCLEOTIDE_ORDER:
        sns.lineplot(x="Position", y=nt, data=seq_df, linewidth=1.0, color=NUCLEOTIDE_COLORS[nt], label=nt, errorbar=None)
    ax = plt.gca()
    add_region_spans(ax, y_text_factor=0.90)
    plt.title(title)
    plt.xlabel("Genomic Position (bp)")
    plt.ylabel("Mean |SHAP|")
    plt.legend(title="Nucleotide")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def save_feature_barplot(feature_df: pd.DataFrame, out_png: str, out_pdf: str, title: str):
    plt.figure(figsize=(9, 5.5))
    plot_df = feature_df.sort_values("SHAP", ascending=False).copy()
    ax = sns.barplot(data=plot_df, x="SHAP", y="Feature", hue="Feature", palette="viridis", dodge=False, legend=False)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(float(row["SHAP"]) + max(plot_df["SHAP"].max() * 0.01, 1e-6), i, f"{row['SHAP']:.4f}", va="center", fontsize=8)
    plt.title(title)
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def save_cv_plot(cv_df: pd.DataFrame, out_png: str, out_pdf: str, title: str):
    plt.figure(figsize=(18, 4.5))
    sns.lineplot(x="Position", y="CV", data=cv_df, linewidth=1.2, color="darkred", errorbar=None)
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("CV")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def save_multispecies_lineplot(profile_map, out_png: str, out_pdf: str, title: str, ylabel: str, region_spans=True):
    plt.figure(figsize=(18, 5))
    for species_key, df in profile_map.items():
        sns.lineplot(x="Position", y=df.columns[-1], data=df, linewidth=1.3, label=species_key, errorbar=None)
    ax = plt.gca()
    if region_spans:
        add_region_spans(ax)
    plt.title(title)
    plt.xlabel("Genomic Position (bp)")
    plt.ylabel(ylabel)
    plt.legend(title="Species")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def save_multispecies_feature_plot(feature_map, out_png: str, out_pdf: str, title: str):
    plt.figure(figsize=(10, 6))
    long_df = []
    for species_key, fdf in feature_map.items():
        tmp = fdf.copy()
        tmp["Species"] = species_key
        long_df.append(tmp)
    long_df = pd.concat(long_df, ignore_index=True)
    sns.lineplot(data=long_df, x="Feature", y="SHAP", hue="Species", marker="o", errorbar=None)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Mean |SHAP|")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def corr_heatmap_from_vectors(vector_map, out_png: str, out_pdf: str, title: str, method: str = "pearson"):
    species = list(vector_map.keys())
    mat = np.zeros((len(species), len(species)), dtype=float)
    for i, s1 in enumerate(species):
        for j, s2 in enumerate(species):
            v1 = np.asarray(vector_map[s1]).ravel()
            v2 = np.asarray(vector_map[s2]).ravel()
            if method == "spearman":
                mat[i, j] = pd.Series(v1).corr(pd.Series(v2), method="spearman")
            else:
                mat[i, j] = pd.Series(v1).corr(pd.Series(v2), method="pearson")
    heat_df = pd.DataFrame(mat, index=species, columns=species)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, square=True, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return heat_df


def calculate_position_stability(seq_importances_foldwise):
    seq_array = np.array(seq_importances_foldwise)

    # 兼容未提前 squeeze 的历史输出: (n_fold, pos, nt, 1)
    if seq_array.ndim == 4 and seq_array.shape[-1] == 1:
        seq_array = np.squeeze(seq_array, axis=-1)

    if seq_array.ndim != 3:
        raise ValueError(f"seq_importances_foldwise 维度异常: {seq_array.shape}")
    fold_position_mean = seq_array.mean(axis=2)
    mean_shap_per_position = fold_position_mean.mean(axis=0)
    cv_per_position = variation(fold_position_mean, axis=0)
    cv_per_position = np.nan_to_num(cv_per_position, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({
        "Position": np.arange(1, seq_array.shape[1] + 1),
        "Mean_SHAP": mean_shap_per_position.flatten(),
        "CV": cv_per_position.flatten(),
    })

# =========================
# 5. 单任务 SHAP 分析
# =========================
def analyze_single_task(task_name: str, base_df: pd.DataFrame):
    species_key, tissue = resolve_task_parts(task_name)
    expr_df = load_expression_labels(task_name)
    data_df = make_task_dataframe(base_df, expr_df)
    cv_chromosomes = resolve_cv_chromosomes(data_df, species_key)

    task_model_dir = os.path.join(MODEL_OUTPUT_ROOT, task_name)
    task_interp_dir = os.path.join(INTERP_ROOT, task_name)
    ensure_dir(task_interp_dir)
    ensure_dir(os.path.join(task_interp_dir, "per_chrom"))

    X_seq = np.stack(data_df["encoded_seq"].values).astype(np.float32)
    X_mrna_raw = data_df[MRNA_COLS].values.astype(np.float32)
    y = data_df["label"].values.astype(np.int32)

    fold_seq_profiles = []
    fold_feature_profiles = []
    per_chrom_records = []

    for fold_round, valid_chrom in enumerate(cv_chromosomes, start=1):
        model_path = os.path.join(task_model_dir, f"output_DeepCRE_mRNA_BiLSTM_chr_model_chr{valid_chrom}.keras")
        if not os.path.exists(model_path):
            print(f"[WARN] 模型不存在，跳过 {task_name} chr{valid_chrom}: {model_path}")
            continue

        train_df, val_df = make_chromosome_split(data_df, valid_chrom)
        if len(val_df) == 0 or len(train_df) == 0:
            print(f"[WARN] {task_name} chr{valid_chrom} 训练/验证样本为空，跳过")
            continue

        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
        X_seq_train = X_seq[train_idx].copy()
        X_seq_val = X_seq[val_idx].copy()
        X_mrna_train = X_mrna_raw[train_idx].copy()
        X_mrna_val = X_mrna_raw[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_seq_train_bal, X_mrna_train_bal, y_train_bal = balance_train_data(X_seq_train, X_mrna_train, y_train)
        X_seq_train_bal[:, PROMOTER_MASK_SLICE, :] = 0
        X_seq_train_bal[:, TERMINATOR_MASK_SLICE, :] = 0
        X_seq_val[:, PROMOTER_MASK_SLICE, :] = 0
        X_seq_val[:, TERMINATOR_MASK_SLICE, :] = 0

        scaler = StandardScaler()
        X_mrna_train_bal = scaler.fit_transform(X_mrna_train_bal).astype(np.float32)
        X_mrna_val = scaler.transform(X_mrna_val).astype(np.float32)

        model = tf.keras.models.load_model(model_path, compile=False)

        correct_mask = np.ones(len(y_val), dtype=bool)
        y_prob = np.full(len(y_val), np.nan)
        y_pred = np.full(len(y_val), -1)
        if ONLY_CORRECT_PREDICTIONS:
            correct_mask, y_prob, y_pred = get_correct_prediction_mask(model, X_seq_val, X_mrna_val, y_val)
            if correct_mask.sum() == 0:
                print(f"[WARN] {task_name} chr{valid_chrom} 无正确预测样本，跳过")
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                continue

        bg_source_seq = X_seq_val if MAX_EXPLAIN_BACKGROUND_FROM_VAL else X_seq_train_bal
        bg_source_mrna = X_mrna_val if MAX_EXPLAIN_BACKGROUND_FROM_VAL else X_mrna_train_bal
        bg_source_y = y_val if MAX_EXPLAIN_BACKGROUND_FROM_VAL else y_train_bal

        bg_idx = choose_sample_indices(bg_source_y, min(BACKGROUND_PER_CHROM, len(bg_source_y)), random_state=SEED + fold_round)
        exp_candidates = np.where(correct_mask)[0]
        exp_idx_local = choose_sample_indices(y_val[exp_candidates], min(EXPLAIN_PER_CHROM, len(exp_candidates)), random_state=SEED + 1000 + fold_round)
        exp_idx = exp_candidates[exp_idx_local]

        X_seq_bg = bg_source_seq[bg_idx].astype(np.float32)
        X_mrna_bg = bg_source_mrna[bg_idx].astype(np.float32)
        X_seq_exp = X_seq_val[exp_idx].astype(np.float32)
        X_mrna_exp = X_mrna_val[exp_idx].astype(np.float32)
        y_exp = y_val[exp_idx]
        genes_exp = val_df.iloc[exp_idx]["gene_id"].astype(str).tolist()

        seq_sv, mrna_sv = compute_deepshap_for_chrom(model, X_seq_bg, X_mrna_bg, X_seq_exp, X_mrna_exp)
        print(f"[SHAP] {task_name} chr{valid_chrom} seq_sv={seq_sv.shape}, mrna_sv={mrna_sv.shape}")
        if USE_ABS_SHAP:
            seq_sv = np.abs(seq_sv)
            mrna_sv = np.abs(mrna_sv)

        # 这里的 seq_sv 已在 normalize_shap_outputs 中规范为 (n_samples, 3020, 4)
        seq_profile = np.mean(seq_sv, axis=0)  # (3020, 4)
        feat_profile = np.mean(mrna_sv, axis=0).reshape(-1)  # (8,)

        fold_seq_profiles.append(seq_profile)
        fold_feature_profiles.append(feat_profile)

        per_chrom_records.append({
            "task": task_name,
            "species": species_key,
            "tissue": tissue,
            "fold": fold_round,
            "val_chromosome": valid_chrom,
            "n_background": len(bg_idx),
            "n_explained": len(exp_idx),
            "n_val_total": len(y_val),
            "n_val_used": len(y_exp),
            "n_val_pos_used": int((y_exp == 1).sum()),
            "n_val_neg_used": int((y_exp == 0).sum()),
            "only_correct_predictions": ONLY_CORRECT_PREDICTIONS,
            "model_path": model_path,
        })

        if SAVE_PER_CHROM_RAW:
            raw_dir = os.path.join(task_interp_dir, "per_chrom", f"chr_{valid_chrom}")
            ensure_dir(raw_dir)
            np.save(os.path.join(raw_dir, "seq_shap.npy"), seq_sv)
            np.save(os.path.join(raw_dir, "mrna_shap.npy"), mrna_sv)
            pd.DataFrame({"gene_id": genes_exp, "label": y_exp}).to_csv(os.path.join(raw_dir, "explained_samples.csv"), index=False)

        del model, X_seq_bg, X_mrna_bg, X_seq_exp, X_mrna_exp, seq_sv, mrna_sv
        tf.keras.backend.clear_session()
        gc.collect()

    if not fold_seq_profiles:
        raise RuntimeError(f"任务 {task_name} 没有成功完成任何染色体 SHAP 分析")

    mean_seq_shap = np.mean(np.stack(fold_seq_profiles, axis=0), axis=0)
    mean_feature_shap = np.mean(np.stack(fold_feature_profiles, axis=0), axis=0)
    position_stability = calculate_position_stability(fold_seq_profiles)

    seq_df = pd.DataFrame(mean_seq_shap, columns=NUCLEOTIDE_ORDER)
    seq_df["Position"] = np.arange(1, len(seq_df) + 1)
    seq_df["Total_SHAP"] = seq_df[NUCLEOTIDE_ORDER].mean(axis=1)
    seq_df = seq_df[["Position"] + NUCLEOTIDE_ORDER + ["Total_SHAP"]]

    feature_df = pd.DataFrame({"Feature": MRNA_COLS, "SHAP": mean_feature_shap})
    feature_df = feature_df.sort_values("SHAP", ascending=False).reset_index(drop=True)

    per_chrom_df = pd.DataFrame(per_chrom_records)
    per_chrom_df.to_csv(os.path.join(task_interp_dir, "per_chrom_summary.csv"), index=False, encoding="utf-8-sig")
    seq_df.to_csv(os.path.join(task_interp_dir, f"{task_name}_sequence_importance.csv"), index=False, encoding="utf-8-sig")
    feature_df.to_csv(os.path.join(task_interp_dir, f"{task_name}_mrna_feature_importance.csv"), index=False, encoding="utf-8-sig")
    position_stability.to_csv(os.path.join(task_interp_dir, f"{task_name}_position_stability.csv"), index=False, encoding="utf-8-sig")

    save_sequence_importance_plot(
        seq_df,
        os.path.join(task_interp_dir, f"{task_name}_DNA_Importance.png"),
        os.path.join(task_interp_dir, f"{task_name}_DNA_Importance.pdf"),
        f"{task_name} DNA Sequence Importance Landscape",
    )
    save_feature_barplot(
        feature_df,
        os.path.join(task_interp_dir, f"{task_name}_mRNA_Feature_Importance.png"),
        os.path.join(task_interp_dir, f"{task_name}_mRNA_Feature_Importance.pdf"),
        f"{task_name} mRNA Feature Importance Ranking",
    )
    save_cv_plot(
        position_stability,
        os.path.join(task_interp_dir, f"{task_name}_CV_Trend.png"),
        os.path.join(task_interp_dir, f"{task_name}_CV_Trend.pdf"),
        f"{task_name} Coefficient of Variation Across Positions",
    )
    if task_name == "Zmay_leaf":
        save_nucleotide_plot(
            seq_df,
            os.path.join(task_interp_dir, f"{task_name}_Nucleotide_Importance.png"),
            os.path.join(task_interp_dir, f"{task_name}_Nucleotide_Importance.pdf"),
            f"{task_name} Nucleotide Specific Importance Profiles",
        )

    return {
        "task": task_name,
        "species": species_key,
        "tissue": tissue,
        "task_interp_dir": task_interp_dir,
        "seq_df": seq_df,
        "feature_df": feature_df,
        "position_stability": position_stability,
        "per_chrom_df": per_chrom_df,
    }

# =========================
# 6. 汇总面板（类似 Fig.3）
# =========================
def build_species_tissue_panels(task_results):
    figure3_dir = os.path.join(INTERP_ROOT, "figure3_like_panels")
    ensure_dir(figure3_dir)

    # 按 tissue / species 组织
    by_tissue = defaultdict(dict)
    by_species = defaultdict(list)
    for res in task_results:
        by_tissue[res["tissue"]][res["species"]] = res
        by_species[res["species"]].append(res)

    # a/b: 叶与根的特征重要性多物种图
    for tissue in ["leaf", "root"]:
        fmap = {sp: by_tissue[tissue][sp]["feature_df"] for sp in by_tissue[tissue]}
        if fmap:
            save_multispecies_feature_plot(
                fmap,
                os.path.join(figure3_dir, f"panel_{'a' if tissue == 'leaf' else 'b'}_{tissue}_feature_importance.png"),
                os.path.join(figure3_dir, f"panel_{'a' if tissue == 'leaf' else 'b'}_{tissue}_feature_importance.pdf"),
                f"{tissue.capitalize()} mRNA Feature Importance Across Species",
            )

    # c/d: 叶与根的序列位置重要性多物种图
    for tissue in ["leaf", "root"]:
        pmap = {}
        for sp, res in by_tissue[tissue].items():
            tmp = res["seq_df"][["Position", "Total_SHAP"]].copy()
            pmap[sp] = tmp
        if pmap:
            save_multispecies_lineplot(
                pmap,
                os.path.join(figure3_dir, f"panel_{'c' if tissue == 'leaf' else 'd'}_{tissue}_sequence_importance.png"),
                os.path.join(figure3_dir, f"panel_{'c' if tissue == 'leaf' else 'd'}_{tissue}_sequence_importance.pdf"),
                f"{tissue.capitalize()} Sequence Position Importance Across Species",
                ylabel="Mean |SHAP|",
                region_spans=True,
            )

    # e: 玉米叶 A/T/C/G
    if "Zmay" in by_tissue["leaf"]:
        zres = by_tissue["leaf"]["Zmay"]
        save_nucleotide_plot(
            zres["seq_df"],
            os.path.join(figure3_dir, "panel_e_Zmay_leaf_nucleotide_importance.png"),
            os.path.join(figure3_dir, "panel_e_Zmay_leaf_nucleotide_importance.pdf"),
            "Zmay_leaf Nucleotide Specific Importance Profiles",
        )

    # f: 各物种总体 CV 情况（leaf/root 合并均值）
    plt.figure(figsize=(18, 4.5))
    for species_key, res_list in by_species.items():
        cv_merged = pd.concat([r["position_stability"][["Position", "CV"]].copy() for r in res_list], ignore_index=True)
        cv_mean = cv_merged.groupby("Position", as_index=False)["CV"].mean()
        sns.lineplot(data=cv_mean, x="Position", y="CV", label=species_key, errorbar=None)
        cv_mean.to_csv(os.path.join(figure3_dir, f"{species_key}_overall_cv.csv"), index=False, encoding="utf-8-sig")
    plt.title("Overall CV Profiles Across Species")
    plt.xlabel("Position")
    plt.ylabel("CV")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig(os.path.join(figure3_dir, "panel_f_overall_cv_across_species.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(figure3_dir, "panel_f_overall_cv_across_species.pdf"), bbox_inches="tight")
    plt.close()

    # g/h: 额外给出两类相似性热图，便于你后续挑选
    for tissue in ["leaf", "root"]:
        feature_vectors = {sp: by_tissue[tissue][sp]["feature_df"].set_index("Feature").loc[MRNA_COLS, "SHAP"].values for sp in by_tissue[tissue]}
        seq_vectors = {sp: by_tissue[tissue][sp]["seq_df"]["Total_SHAP"].values for sp in by_tissue[tissue]}
        if len(feature_vectors) >= 2:
            fheat = corr_heatmap_from_vectors(
                feature_vectors,
                os.path.join(figure3_dir, f"{tissue}_feature_similarity_heatmap.png"),
                os.path.join(figure3_dir, f"{tissue}_feature_similarity_heatmap.pdf"),
                f"{tissue.capitalize()} Feature Importance Similarity",
                method="pearson",
            )
            fheat.to_csv(os.path.join(figure3_dir, f"{tissue}_feature_similarity_heatmap.csv"), encoding="utf-8-sig")
        if len(seq_vectors) >= 2:
            sheat = corr_heatmap_from_vectors(
                seq_vectors,
                os.path.join(figure3_dir, f"{tissue}_sequence_similarity_heatmap.png"),
                os.path.join(figure3_dir, f"{tissue}_sequence_similarity_heatmap.pdf"),
                f"{tissue.capitalize()} Sequence Importance Similarity",
                method="pearson",
            )
            sheat.to_csv(os.path.join(figure3_dir, f"{tissue}_sequence_similarity_heatmap.csv"), encoding="utf-8-sig")

    # 各物种总序列位置重要性（leaf/root 平均）
    species_profile_dir = os.path.join(figure3_dir, "species_overall_profiles")
    ensure_dir(species_profile_dir)
    for species_key, res_list in by_species.items():
        merged = []
        for res in res_list:
            tmp = res["seq_df"][["Position", "Total_SHAP"]].copy()
            tmp["task"] = res["task"]
            merged.append(tmp)
        merged = pd.concat(merged, ignore_index=True)
        overall = merged.groupby("Position", as_index=False)["Total_SHAP"].mean()
        overall.to_csv(os.path.join(species_profile_dir, f"{species_key}_overall_sequence_importance.csv"), index=False, encoding="utf-8-sig")
        save_sequence_importance_plot(
            overall.rename(columns={"Total_SHAP": "Total_SHAP"}),
            os.path.join(species_profile_dir, f"{species_key}_overall_sequence_importance.png"),
            os.path.join(species_profile_dir, f"{species_key}_overall_sequence_importance.pdf"),
            f"{species_key} Overall Sequence Position Importance",
        )

# =========================
# 7. 主流程
# =========================
def save_run_config():
    cfg = {
        "BASE_PATH": BASE_PATH,
        "DATA_DIR": DATA_DIR,
        "MODEL_OUTPUT_ROOT": MODEL_OUTPUT_ROOT,
        "INTERP_ROOT": INTERP_ROOT,
        "TASK_ORDER": TASK_ORDER,
        "BACKGROUND_PER_CHROM": BACKGROUND_PER_CHROM,
        "EXPLAIN_PER_CHROM": EXPLAIN_PER_CHROM,
        "EXPLAIN_BATCH_SIZE": EXPLAIN_BATCH_SIZE,
        "ONLY_CORRECT_PREDICTIONS": ONLY_CORRECT_PREDICTIONS,
        "USE_ABS_SHAP": USE_ABS_SHAP,
        "SAVE_PER_CHROM_RAW": SAVE_PER_CHROM_RAW,
        "MRNA_COLS": MRNA_COLS,
        "SPECIES_CONFIG": SPECIES_CONFIG,
        "VALIDATION_KEY_ALIASES": VALIDATION_KEY_ALIASES,
    }
    ensure_dir(INTERP_ROOT)
    with open(os.path.join(INTERP_ROOT, "deepshap_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def main():
    t0 = time.time()
    ensure_dir(INTERP_ROOT)
    save_run_config()

    species_run_order = []
    for task_name in TASK_ORDER:
        species_key, _ = resolve_task_parts(task_name)
        if species_key not in species_run_order:
            species_run_order.append(species_key)

    species_base_cache = {}
    task_results = []
    failures = []

    for species_key in species_run_order:
        try:
            val_gene_set, _, _ = load_validation_gene_whitelist(species_key)
            species_base_cache[species_key] = load_species_feature_table(species_key, val_gene_set)
        except Exception as e:
            failures.append({"species": species_key, "task": None, "error": repr(e)})
            print(f"[ERROR] 物种基础数据准备失败: {species_key} -> {e}")

    for task_name in TASK_ORDER:
        species_key, _ = resolve_task_parts(task_name)
        if species_key not in species_base_cache:
            continue
        try:
            print("\n" + "#" * 90)
            print(f"开始 DeepSHAP 分析: {task_name}")
            print("#" * 90)
            result = analyze_single_task(task_name, species_base_cache[species_key])
            task_results.append(result)
        except Exception as e:
            failures.append({"species": species_key, "task": task_name, "error": repr(e)})
            print(f"[ERROR] 任务失败: {task_name} -> {e}")
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    if task_results:
        summary_rows = []
        for res in task_results:
            summary_rows.append({
                "task": res["task"],
                "species": res["species"],
                "tissue": res["tissue"],
                "n_chrom_runs": len(res["per_chrom_df"]),
                "task_interp_dir": res["task_interp_dir"],
            })
        pd.DataFrame(summary_rows).to_csv(os.path.join(INTERP_ROOT, "task_interpretability_summary.csv"), index=False, encoding="utf-8-sig")
        build_species_tissue_panels(task_results)

    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(INTERP_ROOT, "deepshap_failures.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"DeepSHAP 分析完成，总耗时 {time.time() - t0:.2f} 秒")
    print(f"结果目录: {INTERP_ROOT}")
    print("=" * 90)


if __name__ == "__main__":
    main()


import os
import gc
import glob
import json
import random
import warnings
import traceback
import re
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pyranges as pr
from pyfaidx import Fasta

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D,
    MaxPool1D,
    Flatten,
    Dense,
    Dropout,
    Concatenate,
    Input,
    LSTM,
    Bidirectional,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

warnings.filterwarnings("ignore")

# =========================================================
# 0. 全局配置
# =========================================================
DATA_DIR = "/home/data"
VALIDATION_PICKLE_CANDIDATES = [
    os.path.join(DATA_DIR, "validation_genes.pickle"),
    "/home/validation_genes.pickle",
    "/home/data/validation_genes.pickle",
]
OUTPUT_ROOT = "/home/output_DeepCRE_mRNA_chr"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

UPSTREAM = 1000
DOWNSTREAM = 500
GAP_LEN = 20
SEQ_TOTAL_LEN = 2 * (UPSTREAM + DOWNSTREAM) + GAP_LEN  # 3020

BILSTM_UNITS = 64
BILSTM_DROPOUT = 0.20

MRNA_COLS = [
    "UTR5LEN", "CDSLEN", "INTRONLEN", "UTR3LEN",
    "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY"
]

FAIL_FAST = False
SAVE_CHROM_MODELS = True
SCALE_MRNA_WITH_TRAIN_ONLY = True
STRIP_GENE_VERSION = False

USE_MRNA_BRANCH = True
USE_BILSTM_BRANCH = False
STRICT_MATCH_MRNA_SAMPLE_SET = False
MODEL_DISPLAY_NAME = "DeepCRE_mRNA (DNA + mRNA, CNN)"

PROMOTER_MASK_SLICE = slice(UPSTREAM, UPSTREAM + 3)
TERMINATOR_MASK_SLICE = slice(
    UPSTREAM + (DOWNSTREAM * 2) + 17,
    UPSTREAM + (DOWNSTREAM * 2) + 20,
)

TASK_ORDER = [
    "Atha_leaf", "Atha_root",
    "Sbic_leaf", "Sbic_root",
    "Slyc_leaf", "Slyc_root",
    "Zmay_leaf", "Zmay_root",
]

OUTPUT_PREFIX = "output_DeepCRE_mRNA_chr"

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

# =========================================================
# 1. 确定性与 TF/GPU 配置
# =========================================================
def set_global_determinism(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] GPU memory growth 设置失败，继续运行: {e}")


set_global_determinism(SEED)
configure_tensorflow()

# =========================================================
# 2. 通用工具函数
# =========================================================
def normalize_gene_id(gene_id):
    gene_id = str(gene_id).strip()
    if STRIP_GENE_VERSION and "." in gene_id:
        gene_id = gene_id.split(".")[0]
    return gene_id


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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

    raise ValueError(
        f"表达量文件中未找到可用列，候选列名={EXPRESSION_VALUE_COL_CANDIDATES}，实际列={list(df.columns)}"
    )


def resolve_unique_file(base_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"未找到匹配文件: {os.path.join(base_dir, pattern)}")
    if len(matches) > 1:
        print(f"[WARN] 模式 {pattern} 匹配到多个文件，默认使用第一个: {matches[0]}")
    return matches[0]


def resolve_task_parts(task_name: str):
    species_key, tissue = task_name.split("_", 1)
    if species_key not in SPECIES_CONFIG:
        raise ValueError(f"未知物种缩写: {species_key}")
    if tissue not in {"leaf", "root"}:
        raise ValueError(f"未知组织类型: {tissue}")
    return species_key, tissue


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
        if extras:
            print(
                f"[WARN] {species_key} 还检测到未在预设列表中的染色体，将附加到交叉验证末尾: {extras}"
            )
            resolved.extend(extras)
        return resolved

    print(
        f"[WARN] {species_key} 预设染色体键名 {preferred} 与实际染色体名未直接匹配，将改用数据中实际染色体顺序: {actual_sorted}"
    )
    return actual_sorted

# =========================================================
# 3. 文件解析
# =========================================================
def resolve_species_files(species_key: str) -> dict:
    cfg = SPECIES_CONFIG[species_key]
    fasta_path = resolve_unique_file(DATA_DIR, cfg["fasta_glob"])
    gtf_path = resolve_unique_file(DATA_DIR, cfg["gtf_glob"])
    mrna_path = os.path.join(DATA_DIR, cfg["mrna_file"])
    if not os.path.exists(mrna_path):
        raise FileNotFoundError(f"未找到 mRNA 特征文件: {mrna_path}")

    return {
        "fasta_path": fasta_path,
        "gtf_path": gtf_path,
        "mrna_path": mrna_path,
    }


def resolve_expression_file(task_name: str) -> str:
    expr_path = os.path.join(DATA_DIR, f"{task_name}_expression_label_common.csv")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"未找到表达量文件: {expr_path}")
    return expr_path


def resolve_validation_pickle_path() -> str:
    for path in VALIDATION_PICKLE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "未找到 validation_genes.pickle，请检查以下候选路径是否存在: "
        f"{VALIDATION_PICKLE_CANDIDATES}"
    )


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

    if not isinstance(payload, dict):
        raise ValueError(
            f"validation_genes.pickle 顶层对象应为 dict，实际为 {type(payload).__name__}"
        )

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
        raise KeyError(
            f"在 validation_genes.pickle 中未找到物种 {species_key} 对应的验证白名单键。"
            f"可尝试的别名={aliases}；pickle 顶层键示例={list(payload.keys())[:20]}"
        )

    print("\n===== 验证白名单信息 =====")
    print(f"物种: {species_key}")
    print(f"validation_genes.pickle: {pickle_path}")
    print(f"命中键: {matched_key}")
    print(f"白名单基因数: {len(gene_set)}")

    return gene_set, pickle_path, matched_key

# =========================================================
# 4. 读取表达量 / 物种基础特征
# =========================================================
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
    expr_df = expr_df[["gene_id", value_col, "label"]].copy()
    expr_df = expr_df.rename(columns={value_col: "expression_value"})

    print("\n===== 表达量标签信息 =====")
    print(f"任务: {task_name}")
    print(f"使用列: {value_col}")
    print(f"Q25={q25:.6f} | Q75={q75:.6f}")
    print(expr_df["label"].value_counts().sort_index())

    return expr_df


def load_species_feature_table(species_key: str, validation_gene_set: set) -> pd.DataFrame:
    cfg = SPECIES_CONFIG[species_key]
    paths = resolve_species_files(species_key)

    print("\n" + "=" * 80)
    print(f"开始构建物种基础特征: {species_key} | {cfg['display_name']}")
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    print("=" * 80)

    fasta = Fasta(paths["fasta_path"], as_raw=True, sequence_always_upper=True, read_ahead=10000)

    gene_models = pr.read_gtf(paths["gtf_path"], as_df=True)
    gene_models = gene_models[gene_models["Feature"] == "gene"].copy()

    biotype_col = find_first_matching_column(gene_models, ["gene_biotype", "gene_type", "biotype"])
    if biotype_col is not None:
        gene_models = gene_models[gene_models[biotype_col].astype(str) == "protein_coding"].copy()
    else:
        print(f"[WARN] {species_key} 的 GTF 未找到 gene_biotype/gene_type 列，跳过 protein_coding 过滤")

    required_gtf_cols = ["Chromosome", "Start", "End", "Strand", "gene_id"]
    missing_gtf_cols = [c for c in required_gtf_cols if c not in gene_models.columns]
    if missing_gtf_cols:
        raise ValueError(f"GTF 缺少必要列: {missing_gtf_cols}")

    gene_models = gene_models[required_gtf_cols].copy()
    gene_models["gene_id"] = gene_models["gene_id"].astype(str).map(normalize_gene_id)
    gene_models["Chromosome"] = gene_models["Chromosome"].astype(str)

    valid_chroms = set(cfg.get("valid_chroms") or [])
    if valid_chroms:
        filtered_gene_models = gene_models[gene_models["Chromosome"].isin(valid_chroms)].copy()
        if len(filtered_gene_models) == 0:
            print(
                f"[WARN] {species_key} 的染色体过滤后无数据，可能是染色体命名风格与预设不一致。"
                f"将保留全部染色体继续运行。GTF 中示例染色体: {sorted(gene_models['Chromosome'].astype(str).unique())[:10]}"
            )
        else:
            gene_models = filtered_gene_models

    dup_gene_ids = gene_models["gene_id"].value_counts()
    dup_gene_ids = dup_gene_ids[dup_gene_ids > 1].index.tolist()
    if dup_gene_ids:
        print(f"[WARN] {species_key} 的 GTF 中存在重复 gene_id，保留首条记录，数量={len(dup_gene_ids)}")
        gene_models = gene_models.drop_duplicates(subset=["gene_id"], keep="first").copy()

    if USE_MRNA_BRANCH or STRICT_MATCH_MRNA_SAMPLE_SET:
        mrna_df = pd.read_csv(paths["mrna_path"])
        mrna_df = standardize_gene_id_column(mrna_df)
        if USE_MRNA_BRANCH:
            missing_mrna_cols = [c for c in MRNA_COLS if c not in mrna_df.columns]
            if missing_mrna_cols:
                raise ValueError(f"mRNA 特征文件缺少列 {missing_mrna_cols}，文件: {paths['mrna_path']}")
            mrna_df = mrna_df[["gene_id"] + MRNA_COLS].copy()
            mrna_df = mrna_df.dropna(subset=MRNA_COLS).copy()
        else:
            mrna_df = mrna_df[["gene_id"]].copy()
        mrna_gene_set = set(mrna_df["gene_id"])
    else:
        mrna_df = None
        mrna_gene_set = None

    seq_data = []
    total_gene_models = len(gene_models)

    for idx, row in enumerate(gene_models.itertuples(index=False), start=1):
        chrom = str(row.Chromosome)
        start = int(row.Start)
        end = int(row.End)
        strand = str(row.Strand)
        gene_id = normalize_gene_id(row.gene_id)

        if mrna_gene_set is not None and gene_id not in mrna_gene_set:
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

        if idx % 5000 == 0:
            print(f"[{species_key}] 已扫描 {idx}/{total_gene_models} 个 gene models，当前有效样本 {len(seq_data)}")

    core_df = pd.DataFrame(seq_data)
    if core_df.empty:
        raise ValueError(f"{species_key} 未成功构建任何序列样本，请检查 fasta/gtf/chromosome 设置")

    if USE_MRNA_BRANCH:
        base_df = core_df.merge(mrna_df, on="gene_id", how="inner")
        base_df = base_df.dropna(subset=MRNA_COLS).copy()
    else:
        base_df = core_df.copy()

    base_df = base_df.reset_index(drop=True)

    print("\n===== 物种基础特征概览 =====")
    suffix = " + mRNA" if USE_MRNA_BRANCH else ""
    print(f"物种: {species_key} | {cfg['display_name']}")
    print(f"序列{suffix}融合后样本数: {len(base_df)}")
    print("按染色体样本数:")
    print(base_df["chromosome"].value_counts().sort_index())
    print(f"固定序列长度: {SEQ_TOTAL_LEN}")
    if USE_MRNA_BRANCH:
        print(f"mRNA 特征数: {len(MRNA_COLS)}")
    print(f"验证白名单命中样本数: {int(base_df['is_val_gene'].sum())}")
    if (not USE_MRNA_BRANCH) and STRICT_MATCH_MRNA_SAMPLE_SET:
        print("严格对照模式: 已限制为与 mRNA 版本一致的 gene 集合")

    return base_df

# =========================================================
# 5. 训练数据平衡 / 模型
# =========================================================
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

    y_balanced = np.concatenate([
        np.take(train_labels, selected_low, axis=0),
        np.take(train_labels, selected_high, axis=0),
    ], axis=0)

    if train_mrna is None:
        x_mrna_balanced = None
        x_seq_balanced, y_balanced = shuffle(x_seq_balanced, y_balanced, random_state=SEED)
    else:
        x_mrna_balanced = np.concatenate([
            np.take(train_mrna, selected_low, axis=0),
            np.take(train_mrna, selected_high, axis=0),
        ], axis=0)
        x_seq_balanced, x_mrna_balanced, y_balanced = shuffle(
            x_seq_balanced, x_mrna_balanced, y_balanced, random_state=SEED
        )
    return x_seq_balanced, x_mrna_balanced, y_balanced


def build_model(seq_len: int, mrna_dim: int = None):
    K.clear_session()

    seq_input = Input(shape=(seq_len, 4), name="seq_input")

    x = Conv1D(64, kernel_size=8, activation="relu", padding="same", name="conv1_1")(seq_input)
    x = Conv1D(64, kernel_size=8, activation="relu", padding="same", name="conv1_2")(x)
    x = MaxPool1D(8, padding="same", name="pool1")(x)
    x = Dropout(0.25, name="dropout1")(x)

    if USE_BILSTM_BRANCH:
        x = Bidirectional(
            LSTM(
                BILSTM_UNITS,
                return_sequences=True,
                dropout=BILSTM_DROPOUT,
                recurrent_dropout=0.0,
            ),
            merge_mode="concat",
            name="bilstm_after_block1",
        )(x)
        x = Dropout(0.25, name="dropout_bilstm")(x)

    x = Conv1D(128, kernel_size=8, activation="relu", padding="same", name="conv2_1")(x)
    x = Conv1D(128, kernel_size=8, activation="relu", padding="same", name="conv2_2")(x)
    x = MaxPool1D(8, padding="same", name="pool2")(x)
    x = Dropout(0.25, name="dropout2")(x)

    x = Conv1D(64, kernel_size=8, activation="relu", padding="same", name="conv3_1")(x)
    x = Conv1D(64, kernel_size=8, activation="relu", padding="same", name="conv3_2")(x)
    x = MaxPool1D(8, padding="same", name="pool3")(x)
    x = Dropout(0.25, name="dropout3")(x)

    x = Flatten(name="flatten_seq")(x)

    if USE_MRNA_BRANCH:
        if mrna_dim is None:
            raise ValueError("USE_MRNA_BRANCH=True 时 mrna_dim 不能为空")
        mrna_input = Input(shape=(mrna_dim,), name="mrna_input")
        y = Dense(128, activation="relu", name="mrna_dense1")(mrna_input)
        y = Dropout(0.25, name="mrna_dropout1")(y)
        y = Dense(64, activation="relu", name="mrna_dense2")(y)
        y = Dropout(0.25, name="mrna_dropout2")(y)

        combined = Concatenate(name="concat_seq_mrna")([x, y])
        z = Dense(128, activation="relu", name="fusion_dense1")(combined)
        z = Dropout(0.25, name="fusion_dropout")(z)
        z = Dense(64, activation="relu", name="fusion_dense2")(z)
        output = Dense(1, activation="sigmoid", name="output")(z)

        model = Model(inputs=[seq_input, mrna_input], outputs=output)
    else:
        z = Dense(128, activation="relu", name="fusion_dense1")(x)
        z = Dropout(0.25, name="fusion_dropout")(z)
        z = Dense(64, activation="relu", name="fusion_dense2")(z)
        output = Dense(1, activation="sigmoid", name="output")(z)
        model = Model(inputs=seq_input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# =========================================================
# 6. 单任务数据构造与染色体交叉验证
# =========================================================
def make_task_dataframe(base_df: pd.DataFrame, expr_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    label_df = expr_df.loc[expr_df["label"] != 2, ["gene_id", "expression_value", "label"]].copy()

    merged_df = base_df.merge(label_df, on="gene_id", how="inner")
    if USE_MRNA_BRANCH:
        merged_df = merged_df.dropna(subset=MRNA_COLS).copy()
    merged_df = merged_df.reset_index(drop=True)

    print("\n===== 任务数据概览 =====")
    print(f"任务: {task_name}")
    print(f"有效样本数: {len(merged_df)}")
    print("按染色体样本数:")
    print(merged_df["chromosome"].value_counts().sort_index())
    print("按标签样本数:")
    print(merged_df["label"].value_counts().sort_index())
    print(f"白名单验证基因样本数: {int(merged_df['is_val_gene'].sum())}")

    return merged_df


def make_chromosome_split(df: pd.DataFrame, valid_chrom: str):
    chrom_mask = df["chromosome"].astype(str) == str(valid_chrom)
    val_mask = chrom_mask & df["is_val_gene"].astype(bool)
    train_mask = ~chrom_mask
    return df.loc[train_mask].copy(), df.loc[val_mask].copy()


def run_single_task(task_name: str, base_df: pd.DataFrame) -> dict:
    species_key, tissue = resolve_task_parts(task_name)
    task_dir = os.path.join(OUTPUT_ROOT, task_name)
    ensure_dir(task_dir)

    expr_df = load_expression_labels(task_name)
    data_df = make_task_dataframe(base_df, expr_df, task_name)

    cv_chromosomes = resolve_cv_chromosomes(data_df, species_key)
    if len(cv_chromosomes) < 2:
        raise ValueError(
            f"任务 {task_name} 实际可用染色体不足 2 条，无法进行染色体交叉验证。当前染色体: {cv_chromosomes}"
        )

    X_seq = np.stack(data_df["encoded_seq"].values).astype(np.float32)
    X_mrna_raw = data_df[MRNA_COLS].values.astype(np.float32) if USE_MRNA_BRANCH else None
    y = data_df["label"].values.astype(np.int32)

    metrics_rows = []
    auc_curve_rows = []
    oof_rows = []
    roc_cache = []
    mean_fpr = np.linspace(0, 1, 200)

    plt.figure(figsize=(10, 8))

    for fold_round, valid_chrom in enumerate(cv_chromosomes, start=1):
        print(f"\n{'=' * 20} {task_name} | Chr CV Round {fold_round}/{len(cv_chromosomes)} | valid_chrom={valid_chrom} {'=' * 20}")

        train_df, val_df = make_chromosome_split(data_df, valid_chrom)

        if len(val_df) == 0:
            print(f"任务 {task_name}, chromosome={valid_chrom} 无验证样本，跳过")
            metrics_rows.append({
                "task": task_name,
                "species": species_key,
                "tissue": tissue,
                "fold": fold_round,
                "val_chromosome": valid_chrom,
                "train_size": len(train_df),
                "val_size": 0,
                "train_pos": int(train_df["label"].sum()),
                "train_neg": int((train_df["label"] == 0).sum()),
                "val_pos": 0,
                "val_neg": 0,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auc": np.nan,
            })
            continue

        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()

        X_seq_train = X_seq[train_idx].copy()
        X_seq_val = X_seq[val_idx].copy()
        X_mrna_train = X_mrna_raw[train_idx].copy() if USE_MRNA_BRANCH else None
        X_mrna_val = X_mrna_raw[val_idx].copy() if USE_MRNA_BRANCH else None
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_seq_train_bal, X_mrna_train_bal, y_train_bal = balance_train_data(
            X_seq_train, X_mrna_train, y_train
        )

        X_seq_train_bal[:, PROMOTER_MASK_SLICE, :] = 0
        X_seq_train_bal[:, TERMINATOR_MASK_SLICE, :] = 0
        X_seq_val[:, PROMOTER_MASK_SLICE, :] = 0
        X_seq_val[:, TERMINATOR_MASK_SLICE, :] = 0

        if USE_MRNA_BRANCH:
            if SCALE_MRNA_WITH_TRAIN_ONLY:
                scaler = StandardScaler()
                X_mrna_train_bal = scaler.fit_transform(X_mrna_train_bal).astype(np.float32)
                X_mrna_val = scaler.transform(X_mrna_val).astype(np.float32)
            else:
                X_mrna_train_bal = X_mrna_train_bal.astype(np.float32)
                X_mrna_val = X_mrna_val.astype(np.float32)

        print(f"训练集（平衡后）: {len(X_seq_train_bal)} | 验证集: {len(X_seq_val)}")
        print(f"训练集 label 分布 -> 0: {(y_train_bal == 0).sum()} | 1: {(y_train_bal == 1).sum()}")
        print(f"验证集 label 分布 -> 0: {(y_val == 0).sum()} | 1: {(y_val == 1).sum()}")

        if len(np.unique(y_train_bal)) < 2:
            print(f"任务 {task_name}, chromosome={valid_chrom} 训练集仅单一类别，跳过")
            metrics_rows.append({
                "task": task_name,
                "species": species_key,
                "tissue": tissue,
                "fold": fold_round,
                "val_chromosome": valid_chrom,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "train_pos": int(y_train.sum()),
                "train_neg": int((y_train == 0).sum()),
                "val_pos": int(y_val.sum()),
                "val_neg": int((y_val == 0).sum()),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auc": np.nan,
            })
            continue

        K.clear_session()
        gc.collect()
        set_global_determinism(SEED + fold_round)

        model = build_model(seq_len=SEQ_TOTAL_LEN, mrna_dim=(len(MRNA_COLS) if USE_MRNA_BRANCH else None))

        model_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_model_chr{valid_chrom}.keras")
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=5, factor=0.1, verbose=1),
        ]
        if SAVE_CHROM_MODELS:
            callbacks.insert(1, ModelCheckpoint(model_path, save_best_only=True, verbose=1))

        train_inputs = [X_seq_train_bal, X_mrna_train_bal] if USE_MRNA_BRANCH else X_seq_train_bal
        val_inputs = [X_seq_val, X_mrna_val] if USE_MRNA_BRANCH else X_seq_val

        model.fit(
            train_inputs,
            y_train_bal,
            validation_data=(val_inputs, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=callbacks,
        )

        y_prob = model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if len(np.unique(y_val)) == 2:
            auc = roc_auc_score(y_val, y_prob)
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            roc_cache.append((valid_chrom, auc, fpr, tpr))
            auc_curve_rows.append(pd.DataFrame({
                "task": task_name,
                "species": species_key,
                "tissue": tissue,
                "fold": fold_round,
                "val_chromosome": valid_chrom,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc,
            }))
        else:
            auc = np.nan

        metrics_rows.append({
            "task": task_name,
            "species": species_key,
            "tissue": tissue,
            "fold": fold_round,
            "val_chromosome": valid_chrom,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_pos": int(y_train.sum()),
            "train_neg": int((y_train == 0).sum()),
            "val_pos": int(y_val.sum()),
            "val_neg": int((y_val == 0).sum()),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        })

        fold_oof_df = val_df[["gene_id", "chromosome", "label", "expression_value", "is_val_gene"]].copy()
        fold_oof_df["task"] = task_name
        fold_oof_df["species"] = species_key
        fold_oof_df["tissue"] = tissue
        fold_oof_df["cv_round"] = fold_round
        fold_oof_df["val_chromosome"] = valid_chrom
        fold_oof_df["pred_prob"] = y_prob
        fold_oof_df["pred_label"] = y_pred
        oof_rows.append(fold_oof_df)

        print(
            f"{task_name} | chromosome={valid_chrom} -> "
            f"AUC={auc:.4f} | Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}"
        )

        del model, X_seq_train_bal, X_seq_val, y_train_bal, y_val, y_prob, y_pred
        if USE_MRNA_BRANCH:
            del X_mrna_train_bal, X_mrna_val
        K.clear_session()
        gc.collect()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame({
        "task": [task_name] * 5,
        "species": [species_key] * 5,
        "tissue": [tissue] * 5,
        "metric": ["accuracy", "precision", "recall", "f1", "auc"],
        "mean": [metrics_df[c].mean(skipna=True) for c in ["accuracy", "precision", "recall", "f1", "auc"]],
        "std": [metrics_df[c].std(skipna=True) for c in ["accuracy", "precision", "recall", "f1", "auc"]],
        "n_samples": [len(data_df)] * 5,
        "pos_samples": [int((data_df["label"] == 1).sum())] * 5,
        "neg_samples": [int((data_df["label"] == 0).sum())] * 5,
    })
    summary_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    auc_curve_path = None
    if auc_curve_rows:
        auc_curve_df = pd.concat(auc_curve_rows, ignore_index=True)
        auc_curve_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_auc_curves.csv")
        auc_curve_df.to_csv(auc_curve_path, index=False, encoding="utf-8-sig")

    oof_path = None
    if oof_rows:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_oof_predictions.csv")
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    for valid_chrom, auc, fpr, tpr in roc_cache:
        plt.plot(fpr, tpr, lw=2, label=f"{valid_chrom} (AUC={auc:.3f})")

    if roc_cache:
        interp_tprs = []
        aucs = []
        for _, auc, fpr, tpr in roc_cache:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            aucs.append(auc)
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = float(np.mean(aucs))
        plt.plot(mean_fpr, mean_tpr, linestyle="--", lw=3, color="black", label=f"Mean AUC={mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{task_name} chromosome-wise CV ROC ({MODEL_DISPLAY_NAME})")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_png_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_roc.png")
    roc_pdf_path = os.path.join(task_dir, f"{OUTPUT_PREFIX}_roc.pdf")
    plt.savefig(roc_png_path, dpi=300)
    plt.savefig(roc_pdf_path)
    plt.close()

    summary_wide = {
        "task": task_name,
        "species": species_key,
        "tissue": tissue,
        "n_samples": len(data_df),
        "pos_samples": int((data_df["label"] == 1).sum()),
        "neg_samples": int((data_df["label"] == 0).sum()),
        "n_chromosomes": len(cv_chromosomes),
        "chromosomes": "|".join(map(str, cv_chromosomes)),
        "n_validation_whitelist_samples": int(data_df["is_val_gene"].sum()),
        "use_mrna_branch": USE_MRNA_BRANCH,
        "use_bilstm_branch": USE_BILSTM_BRANCH,
        "strict_match_mrna_sample_set": STRICT_MATCH_MRNA_SAMPLE_SET,
        "accuracy_mean": metrics_df["accuracy"].mean(skipna=True),
        "accuracy_std": metrics_df["accuracy"].std(skipna=True),
        "precision_mean": metrics_df["precision"].mean(skipna=True),
        "precision_std": metrics_df["precision"].std(skipna=True),
        "recall_mean": metrics_df["recall"].mean(skipna=True),
        "recall_std": metrics_df["recall"].std(skipna=True),
        "f1_mean": metrics_df["f1"].mean(skipna=True),
        "f1_std": metrics_df["f1"].std(skipna=True),
        "auc_mean": metrics_df["auc"].mean(skipna=True),
        "auc_std": metrics_df["auc"].std(skipna=True),
        "task_dir": task_dir,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "auc_curve_path": auc_curve_path,
        "oof_path": oof_path,
        "roc_png_path": roc_png_path,
        "roc_pdf_path": roc_pdf_path,
    }

    print("\n===== 单任务结果完成 =====")
    print(json.dumps(summary_wide, ensure_ascii=False, indent=2, default=str))

    return {
        "summary_wide": summary_wide,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
    }

# =========================================================
# 7. 总控流程：按物种加载一次基础特征，分别跑 leaf/root
# =========================================================
def save_run_config():
    cfg = {
        "DATA_DIR": DATA_DIR,
        "VALIDATION_PICKLE_CANDIDATES": VALIDATION_PICKLE_CANDIDATES,
        "VALIDATION_KEY_ALIASES": VALIDATION_KEY_ALIASES,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "OUTPUT_PREFIX": OUTPUT_PREFIX,
        "MODEL_DISPLAY_NAME": MODEL_DISPLAY_NAME,
        "USE_MRNA_BRANCH": USE_MRNA_BRANCH,
        "USE_BILSTM_BRANCH": USE_BILSTM_BRANCH,
        "STRICT_MATCH_MRNA_SAMPLE_SET": STRICT_MATCH_MRNA_SAMPLE_SET,
        "TASK_ORDER": TASK_ORDER,
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "UPSTREAM": UPSTREAM,
        "DOWNSTREAM": DOWNSTREAM,
        "GAP_LEN": GAP_LEN,
        "SEQ_TOTAL_LEN": SEQ_TOTAL_LEN,
        "MRNA_COLS": MRNA_COLS,
        "SAVE_CHROM_MODELS": SAVE_CHROM_MODELS,
        "SCALE_MRNA_WITH_TRAIN_ONLY": SCALE_MRNA_WITH_TRAIN_ONLY,
        "STRIP_GENE_VERSION": STRIP_GENE_VERSION,
        "SPECIES_CONFIG": SPECIES_CONFIG,
    }
    config_path = os.path.join(OUTPUT_ROOT, f"{OUTPUT_PREFIX}_run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def main():
    ensure_dir(OUTPUT_ROOT)
    save_run_config()

    print("\n" + "#" * 100)
    print("开始运行：4 个物种 × 2 个组织，共 8 个任务")
    print("当前版本已由 5-fold 改为按物种染色体交叉验证")
    print("验证集 = 当前验证染色体 ∩ validation_genes.pickle 白名单；训练集 = 其余染色体")
    print(f"当前模型: {MODEL_DISPLAY_NAME}")
    print(f"USE_MRNA_BRANCH={USE_MRNA_BRANCH} | USE_BILSTM_BRANCH={USE_BILSTM_BRANCH} | STRICT_MATCH_MRNA_SAMPLE_SET={STRICT_MATCH_MRNA_SAMPLE_SET}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("任务顺序:", TASK_ORDER)
    print("#" * 100)

    species_to_tasks = defaultdict(list)
    for task_name in TASK_ORDER:
        species_key, _ = resolve_task_parts(task_name)
        species_to_tasks[species_key].append(task_name)

    all_summary_rows = []
    all_metrics_list = []
    failures = []

    species_run_order = []
    for task_name in TASK_ORDER:
        species_key, _ = resolve_task_parts(task_name)
        if species_key not in species_run_order:
            species_run_order.append(species_key)

    for species_key in species_run_order:
        if species_key not in species_to_tasks:
            continue
        if len(species_to_tasks[species_key]) == 0:
            continue

        try:
            validation_gene_set, _, _ = load_validation_gene_whitelist(species_key)
            base_df = load_species_feature_table(species_key, validation_gene_set)
        except Exception as e:
            err = {
                "species": species_key,
                "task": None,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            failures.append(err)
            print(f"[ERROR] 物种基础特征构建失败: {species_key}\n{err['traceback']}")
            if FAIL_FAST:
                raise
            continue

        for task_name in species_to_tasks[species_key]:
            try:
                result = run_single_task(task_name, base_df)
                all_summary_rows.append(result["summary_wide"])
                all_metrics_list.append(result["metrics_df"])
            except Exception as e:
                err = {
                    "species": species_key,
                    "task": task_name,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                failures.append(err)
                print(f"[ERROR] 任务失败: {task_name}\n{err['traceback']}")
                if FAIL_FAST:
                    raise

        del base_df
        gc.collect()
        K.clear_session()
        species_to_tasks[species_key] = []

    if all_summary_rows:
        all_summary_df = pd.DataFrame(all_summary_rows)
        all_summary_df["task"] = pd.Categorical(all_summary_df["task"], categories=TASK_ORDER, ordered=True)
        all_summary_df = all_summary_df.sort_values("task").reset_index(drop=True)
        all_summary_path = os.path.join(OUTPUT_ROOT, f"{OUTPUT_PREFIX}_all_tasks_summary_ordered.csv")
        all_summary_df.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
    else:
        all_summary_df = pd.DataFrame()
        all_summary_path = None

    if all_metrics_list:
        all_metrics_df = pd.concat(all_metrics_list, ignore_index=True)
        all_metrics_df["task"] = pd.Categorical(all_metrics_df["task"], categories=TASK_ORDER, ordered=True)
        all_metrics_df = all_metrics_df.sort_values(["task", "fold"]).reset_index(drop=True)
        all_metrics_path = os.path.join(OUTPUT_ROOT, f"{OUTPUT_PREFIX}_all_tasks_chrom_metrics_ordered.csv")
        all_metrics_df.to_csv(all_metrics_path, index=False, encoding="utf-8-sig")
    else:
        all_metrics_df = pd.DataFrame()
        all_metrics_path = None

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_path = os.path.join(OUTPUT_ROOT, f"{OUTPUT_PREFIX}_run_failures.csv")
        failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")
    else:
        failures_path = None

    print("\n" + "=" * 100)
    print("全部任务运行结束")
    if all_summary_path:
        print(f"汇总结果: {all_summary_path}")
    if all_metrics_path:
        print(f"逐染色体结果: {all_metrics_path}")
    if failures_path:
        print(f"失败日志: {failures_path}")

    if not all_summary_df.empty:
        print("\n===== 总体汇总（按任务顺序） =====")
        display_cols = [
            "task", "n_samples", "pos_samples", "neg_samples", "n_chromosomes",
            "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"
        ]
        print(all_summary_df[display_cols])

    print("=" * 100)


if __name__ == "__main__":
    main()

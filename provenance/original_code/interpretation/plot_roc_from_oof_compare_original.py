from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, roc_auc_score

DEFAULT_BASE_DIR = Path("/home")
DEFAULT_OUTPUT_DIR = Path("/home/ROC_oof_compare_original")

GROUP_PREFIX_ORDER = [
    "Zmay_leaf", "Sbic_leaf", "Slyc_leaf", "Atha_leaf",
    "Zmay_root", "Sbic_root", "Slyc_root", "Atha_root",
]

ALL_PREFIX_ORDER = [
    "Atha_leaf", "Atha_root",
    "Sbic_leaf", "Sbic_root",
    "Slyc_leaf", "Slyc_root",
    "Zmay_leaf", "Zmay_root",
]

MODEL_SPECS = {
    "M1": {"folder": "output_DeepCRE_chr", "legend": "M1", "color": "#1f77b4"},
    "M2": {"folder": "output_DeepCRE_BiLSTM_chr", "legend": "M2", "color": "#ff7f0e"},
    "M3": {"folder": "output_DeepCRE_mRNA_chr", "legend": "M3", "color": "#2ca02c"},
    "M4": {"folder": "output_DeepCRE_mRNA_BiLSTM_chr", "legend": "M4", "color": "#d62728"},
}
MODEL_ORDER = ["M1", "M2", "M3", "M4"]

@dataclass
class OofLocation:
    prefix: str
    model_key: str
    csv_path: Path

@dataclass
class RocFromOof:
    prefix: str
    model_key: str
    source_file: str
    n_samples: int
    n_pos: int
    n_neg: int
    auc: float
    fpr: np.ndarray
    tpr: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成原始风格 ROC 组图用于对比")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def find_oof_csv(model_dir: Path, prefix: str) -> Optional[Path]:
    prefix_dir = model_dir / prefix
    if not prefix_dir.exists():
        return None
    preferred = prefix_dir / f"{model_dir.name}_oof_predictions.csv"
    if preferred.exists():
        return preferred
    cands = sorted(prefix_dir.glob("*_oof_predictions.csv"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        for c in cands:
            if model_dir.name.lower() in c.name.lower():
                return c
        return cands[0]
    return None


def discover_files(base_dir: Path) -> Dict[str, Dict[str, OofLocation]]:
    grouped: Dict[str, Dict[str, OofLocation]] = {p: {} for p in ALL_PREFIX_ORDER}
    for prefix in ALL_PREFIX_ORDER:
        for model_key in MODEL_ORDER:
            model_dir = base_dir / MODEL_SPECS[model_key]["folder"]
            csv_path = find_oof_csv(model_dir, prefix)
            if csv_path is not None:
                grouped[prefix][model_key] = OofLocation(prefix, model_key, csv_path)
    return grouped


def read_oof_and_compute(loc: OofLocation) -> RocFromOof:
    df = pd.read_csv(loc.csv_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    label_col = cols.get("label")
    prob_col = cols.get("pred_prob")
    if label_col is None or prob_col is None:
        raise ValueError(f"{loc.csv_path} 缺少 label/pred_prob 列，实际列={list(df.columns)}")

    work = df[[label_col, prob_col]].copy()
    work.columns = ["label", "pred_prob"]
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work["pred_prob"] = pd.to_numeric(work["pred_prob"], errors="coerce")
    work = work.dropna()
    work = work[work["label"].isin([0, 1])].copy()

    if work.empty:
        raise ValueError(f"{loc.csv_path} 清理后没有有效 0/1 样本")
    if work["label"].nunique() < 2:
        raise ValueError(f"{loc.csv_path} 只有单一类别，无法计算 ROC")

    y_true = work["label"].astype(int).to_numpy()
    y_prob = work["pred_prob"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))

    return RocFromOof(
        prefix=loc.prefix,
        model_key=loc.model_key,
        source_file=str(loc.csv_path),
        n_samples=len(work),
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
        auc=auc,
        fpr=fpr,
        tpr=tpr,
    )


def style_axis(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.20)


def plot_group_2x4(group_results: Dict[str, Dict[str, RocFromOof]], out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, prefix in enumerate(GROUP_PREFIX_ORDER):
        ax = axes[i]
        results = group_results.get(prefix, {})

        for model_key in MODEL_ORDER:
            if model_key not in results:
                continue
            res = results[model_key]
            spec = MODEL_SPECS[model_key]
            ax.plot(
                res.fpr, res.tpr,
                color=spec["color"],
                linewidth=3.8,
                alpha=1.0,
                linestyle="-",
                solid_capstyle="round",
                label=f"{spec['legend']} (AUC = {res.auc:.3f})",
            )

        ax.plot(
            [0, 1], [0, 1],
            color="#7F7F7F",
            linestyle="--",
            linewidth=2.6,
            alpha=0.95,
            label="Random Guess",
        )

        style_axis(ax)
        ax.set_title(prefix, fontsize=12.2, fontweight="bold", pad=7)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)

        leg = ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.98,
            fontsize=8.6,
            borderpad=0.45,
            handlelength=2.2,
            labelspacing=0.28,
        )
        leg.get_frame().set_edgecolor("#C8C8C8")

    plt.subplots_adjust(wspace=0.22, hspace=0.26, left=0.06, right=0.985, bottom=0.08, top=0.95)
    fig.savefig(out_dir / "ROC_group_2x4_original_like.png", dpi=dpi)
    fig.savefig(out_dir / "ROC_group_2x4_original_like.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"根目录不存在: {base_dir}")

    setup_style()
    grouped = discover_files(base_dir)

    group_results: Dict[str, Dict[str, RocFromOof]] = {}
    for prefix in ALL_PREFIX_ORDER:
        prefix_results: Dict[str, RocFromOof] = {}
        for model_key in MODEL_ORDER:
            loc = grouped.get(prefix, {}).get(model_key)
            if loc is None:
                warnings.warn(f"{prefix} 缺少 {model_key} 的 OOF CSV")
                continue
            res = read_oof_and_compute(loc)
            prefix_results[model_key] = res
            print(f"[OK] {prefix} {model_key}: all OOF samples={res.n_samples}, AUC={res.auc:.4f}")
        if prefix_results:
            group_results[prefix] = prefix_results

    if not group_results:
        raise RuntimeError("没有任何可用结果")

    plot_group_2x4(group_results, out_dir, args.dpi)
    print(f"[DONE] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()

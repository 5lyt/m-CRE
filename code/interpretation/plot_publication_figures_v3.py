from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


BASE_DIR_DEFAULT = Path("/home")
OUTPUT_DIR_DEFAULT = BASE_DIR_DEFAULT / "publication_figures_v3"

TASK_ORDER = [
    "Zmay_leaf", "Sbic_leaf", "Slyc_leaf", "Atha_leaf",
    "Zmay_root", "Sbic_root", "Slyc_root", "Atha_root",
]
SPECIES_ORDER = ["Zmay", "Sbic", "Slyc", "Atha"]
TISSUE_ORDER = ["leaf", "root"]
SPECIES_SHORT = {"Zmay": "Zmay", "Sbic": "Sbic", "Slyc": "Slyc", "Atha": "Atha"}

MODEL_SPECS = {
    "M1": {"dir_name": "output_DeepCRE_chr", "label": "M1", "color": "#4169E1"},
    "M2": {"dir_name": "output_DeepCRE_BiLSTM_chr", "label": "M2", "color": "#D95F02"},
    "M3": {"dir_name": "output_DeepCRE_mRNA_chr", "label": "M3", "color": "#008000"},
    "M4": {"dir_name": "output_DeepCRE_mRNA_BiLSTM_chr", "label": "M4", "color": "#E41A1C"},
}
MODEL_ORDER = ["M1", "M2", "M3", "M4"]

SPECIES_COLORS = {
    "Atha": "#2E8B57",
    "Sbic": "#9370DB",
    "Slyc": "#CD5C5C",
    "Zmay": "#FF8C00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready Figure 1 and Figure 2.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.9,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.23, color="#A9A9A9")
    ax.tick_params(axis="both", length=3.4, width=0.8)


def parse_task(task: str) -> Tuple[str, str]:
    species, tissue = task.split("_", 1)
    return species, tissue


def find_metrics_csv(task_dir: Path, preferred_prefix: str) -> Optional[Path]:
    preferred = task_dir / f"{preferred_prefix}_metrics.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(task_dir.glob("*_metrics.csv"))
    if not candidates:
        return None

    for c in candidates:
        if preferred_prefix in c.name:
            return c
    return candidates[0]


def load_metrics_for_model(base_dir: Path, model_key: str) -> pd.DataFrame:
    model_dir = base_dir / MODEL_SPECS[model_key]["dir_name"]
    if not model_dir.exists():
        warnings.warn(f"Model directory missing: {model_dir}")
        return pd.DataFrame()

    rows = []
    preferred_prefix = model_dir.name

    for task in TASK_ORDER:
        task_dir = model_dir / task
        if not task_dir.exists():
            warnings.warn(f"{model_key}/{task}: task directory missing")
            continue

        metrics_path = find_metrics_csv(task_dir, preferred_prefix)
        if metrics_path is None:
            warnings.warn(f"{model_key}/{task}: *_metrics.csv missing")
            continue

        df = pd.read_csv(metrics_path)
        species, tissue = parse_task(task)

        if "task" not in df.columns:
            df["task"] = task
        if "species" not in df.columns:
            df["species"] = species
        if "tissue" not in df.columns:
            df["tissue"] = tissue

        df["model_key"] = model_key
        df["model_label"] = MODEL_SPECS[model_key]["label"]
        df["source_file"] = str(metrics_path)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_all_metrics(base_dir: Path) -> pd.DataFrame:
    frames = []
    for model_key in MODEL_ORDER:
        df = load_metrics_for_model(base_dir, model_key)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No metrics data found. Please check the model output folders.")

    return pd.concat(frames, ignore_index=True)


def build_auc_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["model_key", "model_label", "task", "species", "tissue"]
    for keys, sub in metrics_df.groupby(group_cols):
        model_key, model_label, task, species, tissue = keys
        vals = pd.to_numeric(sub["auc"], errors="coerce").dropna()

        rows.append({
            "model_key": model_key,
            "model_label": model_label,
            "task": task,
            "species": species,
            "tissue": tissue,
            "n_folds": int(len(vals)),
            "auc_mean": float(vals.mean()) if len(vals) else np.nan,
            "auc_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "auc_sem": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "auc_min": float(vals.min()) if len(vals) else np.nan,
            "auc_max": float(vals.max()) if len(vals) else np.nan,
        })

    summary = pd.DataFrame(rows)
    summary["species"] = pd.Categorical(summary["species"], categories=SPECIES_ORDER, ordered=True)
    summary["tissue"] = pd.Categorical(summary["tissue"], categories=TISSUE_ORDER, ordered=True)
    return summary.sort_values(["tissue", "species", "model_key"]).reset_index(drop=True)


def auto_ylim_auc(summary_df: pd.DataFrame) -> Tuple[float, float]:
    values = summary_df["auc_mean"].dropna().values
    if len(values) == 0:
        return 0.84, 1.00

    ymin = max(0.84, np.floor((values.min() - 0.012) * 50) / 50)
    ymax = min(1.005, max(0.96, np.ceil((values.max() + 0.008) * 50) / 50))

    if ymax - ymin < 0.08:
        ymin = max(0.84, ymax - 0.10)
    return ymin, ymax


def save_figure(fig, path_prefix: Path, dpi: int) -> Tuple[Path, Path]:
    png = path_prefix.with_suffix(".png")
    pdf = path_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def save_figure1_csvs(summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "Figure1_multimodel_AUROC_pointline"
    ensure_dir(fig_dir)

    summary_df.to_csv(fig_dir / "Figure1_all_panels_source.csv", index=False, encoding="utf-8-sig")
    for tissue in TISSUE_ORDER:
        sub = summary_df[summary_df["tissue"] == tissue].copy()
        sub.to_csv(fig_dir / f"Figure1_{tissue}_panel_source.csv", index=False, encoding="utf-8-sig")


def plot_figure1_panel(ax, tissue_df: pd.DataFrame, tissue: str, panel_label: str, y_lim: Tuple[float, float]) -> None:
    x = np.arange(len(SPECIES_ORDER))

    for model_key in MODEL_ORDER:
        sub = tissue_df[tissue_df["model_key"] == model_key].copy()
        if sub.empty:
            continue

        sub["species"] = pd.Categorical(sub["species"], categories=SPECIES_ORDER, ordered=True)
        sub = sub.sort_values("species")

        ax.plot(
            x,
            sub["auc_mean"].values,
            color=MODEL_SPECS[model_key]["color"],
            linewidth=1.45,
            marker="o",
            markersize=5.9,
            alpha=0.93,
            label=MODEL_SPECS[model_key]["label"],
        )

    clean_axis(ax, grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([SPECIES_SHORT[s] for s in SPECIES_ORDER])
    ax.set_ylim(*y_lim)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlabel("Species")
    ax.set_title(f"{panel_label}. {tissue.capitalize()} tissue", loc="left")


def save_single_panel_figure1(summary_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig_dir = out_dir / "Figure1_multimodel_AUROC_pointline"
    ensure_dir(fig_dir)
    y_lim = auto_ylim_auc(summary_df)

    for tissue, label in zip(TISSUE_ORDER, ["a", "b"]):
        sub = summary_df[summary_df["tissue"] == tissue].copy()

        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        plot_figure1_panel(ax, sub, tissue, label, y_lim)
        ax.set_ylabel("AUROC")
        ax.legend(loc="upper right", frameon=False, ncol=2)

        fig.tight_layout()
        save_figure(fig, fig_dir / f"Figure1_{tissue}_panel", dpi)


def plot_figure1_group(summary_df: pd.DataFrame, out_dir: Path, dpi: int) -> Tuple[Path, Path]:
    fig_dir = out_dir / "Figure1_multimodel_AUROC_pointline"
    ensure_dir(fig_dir)
    save_figure1_csvs(summary_df, out_dir)

    y_lim = auto_ylim_auc(summary_df)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)

    for ax, tissue, label in zip(axes, TISSUE_ORDER, ["a", "b"]):
        sub = summary_df[summary_df["tissue"] == tissue].copy()
        plot_figure1_panel(ax, sub, tissue, label, y_lim)

    axes[0].set_ylabel("AUROC")

    legend_handles = [
        Line2D(
            [0], [0],
            color=MODEL_SPECS[m]["color"],
            marker="o",
            linewidth=1.45,
            markersize=5.9,
            label=MODEL_SPECS[m]["label"],
        )
        for m in MODEL_ORDER
    ]

    fig.legend(
        legend_handles,
        [MODEL_SPECS[m]["label"] for m in MODEL_ORDER],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    png, pdf = save_figure(fig, fig_dir / "Figure1_group", dpi)

    save_single_panel_figure1(summary_df, out_dir, dpi)
    return png, pdf


def jitter(n: int, width: float = 0.055) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])[:n]
    return np.linspace(-width, width, n)


def save_figure2_csvs(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "Figure2_M4_violin_box_folds"
    ensure_dir(fig_dir)

    m4 = metrics_df[metrics_df["model_key"] == "M4"].copy()
    m4.to_csv(fig_dir / "Figure2_all_panels_source.csv", index=False, encoding="utf-8-sig")

    for tissue in TISSUE_ORDER:
        sub = m4[m4["tissue"] == tissue].copy()
        sub.to_csv(fig_dir / f"Figure2_{tissue}_panel_source.csv", index=False, encoding="utf-8-sig")


def get_m4_ylim(metrics_df: pd.DataFrame) -> Tuple[float, float]:
    m4 = metrics_df[metrics_df["model_key"] == "M4"].copy()
    vals = pd.to_numeric(m4["auc"], errors="coerce").dropna().values
    if len(vals) == 0:
        return 0.84, 1.00

    ymin = max(0.80, np.floor((vals.min() - 0.015) * 50) / 50)
    ymax = min(1.005, max(0.98, np.ceil((vals.max() + 0.010) * 50) / 50))
    return ymin, ymax


def plot_figure2_panel(ax, tissue_df: pd.DataFrame, tissue: str, panel_label: str, y_lim: Tuple[float, float]) -> None:
    positions = np.arange(1, len(SPECIES_ORDER) + 1)

    for pos, species in zip(positions, SPECIES_ORDER):
        vals = tissue_df[tissue_df["species"] == species]["auc"].dropna().astype(float).values
        if len(vals) == 0:
            continue

        violin = ax.violinplot(
            [vals],
            positions=[pos],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(SPECIES_COLORS[species])
            body.set_edgecolor(SPECIES_COLORS[species])
            body.set_alpha(0.22)

        ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#333333", linewidth=1.0),
            boxprops=dict(facecolor="white", edgecolor="#666666", linewidth=0.95),
            whiskerprops=dict(color="#666666", linewidth=0.9),
            capprops=dict(color="#666666", linewidth=0.9),
        )

        ax.scatter(
            np.full(len(vals), pos) + jitter(len(vals)),
            vals,
            s=22,
            color=SPECIES_COLORS[species],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

        mean_val = float(np.mean(vals))
        sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.scatter([pos], [mean_val], marker="D", s=36, color="#111111", zorder=4)

        ax.text(
            pos,
            min(y_lim[1] - 0.002, mean_val + max(0.006, sd_val + 0.003)),
            f"{mean_val:.3f}±{sd_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#333333",
        )

    clean_axis(ax, grid_axis="y")
    ax.set_xticks(positions)
    ax.set_xticklabels([SPECIES_SHORT[s] for s in SPECIES_ORDER])
    ax.set_xlim(0.5, len(SPECIES_ORDER) + 0.5)
    ax.set_ylim(*y_lim)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlabel("Species")
    ax.set_title(f"{panel_label}. {tissue.capitalize()} tissue (M4)", loc="left")


def figure2_legend_handles() -> list:
    return [
        Patch(facecolor="#B0B0B0", alpha=0.22, edgecolor="#B0B0B0", label="Violin"),
        Patch(facecolor="white", edgecolor="#666666", label="Box"),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#888888",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            label="Fold values",
        ),
        Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            color="#111111",
            markersize=6,
            label="Mean",
        ),
    ]


def save_single_panel_figure2(metrics_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig_dir = out_dir / "Figure2_M4_violin_box_folds"
    ensure_dir(fig_dir)

    m4 = metrics_df[metrics_df["model_key"] == "M4"].copy()
    y_lim = get_m4_ylim(metrics_df)

    for tissue, label in zip(TISSUE_ORDER, ["a", "b"]):
        sub = m4[m4["tissue"] == tissue].copy()

        fig, ax = plt.subplots(figsize=(4.9, 4.15))
        plot_figure2_panel(ax, sub, tissue, label, y_lim)
        ax.set_ylabel("AUROC across chromosome-wise folds")

        handles = figure2_legend_handles()
        ax.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=False, ncol=2)

        fig.tight_layout()
        save_figure(fig, fig_dir / f"Figure2_{tissue}_panel", dpi)


def plot_figure2_group(metrics_df: pd.DataFrame, out_dir: Path, dpi: int) -> Tuple[Path, Path]:
    fig_dir = out_dir / "Figure2_M4_violin_box_folds"
    ensure_dir(fig_dir)
    save_figure2_csvs(metrics_df, out_dir)

    m4 = metrics_df[metrics_df["model_key"] == "M4"].copy()
    if m4.empty:
        raise RuntimeError("No M4 data found. Cannot generate Figure 2.")

    y_lim = get_m4_ylim(metrics_df)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharey=True)

    for ax, tissue, label in zip(axes, TISSUE_ORDER, ["a", "b"]):
        sub = m4[m4["tissue"] == tissue].copy()
        plot_figure2_panel(ax, sub, tissue, label, y_lim)

    axes[0].set_ylabel("AUROC across chromosome-wise folds")

    handles = figure2_legend_handles()
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    png, pdf = save_figure(fig, fig_dir / "Figure2_group", dpi)

    save_single_panel_figure2(metrics_df, out_dir, dpi)
    return png, pdf


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    ensure_dir(out_dir)

    setup_style()

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    metrics_df = load_all_metrics(base_dir)
    metrics_csv = out_dir / "all_models_fold_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {metrics_csv}")

    auc_summary = build_auc_summary(metrics_df)
    auc_summary_csv = out_dir / "all_models_auc_summary.csv"
    auc_summary.to_csv(auc_summary_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {auc_summary_csv}")

    fig1_png, fig1_pdf = plot_figure1_group(auc_summary, out_dir, args.dpi)
    print(f"[SAVE] {fig1_png}")
    print(f"[SAVE] {fig1_pdf}")

    fig2_png, fig2_pdf = plot_figure2_group(metrics_df, out_dir, args.dpi)
    print(f"[SAVE] {fig2_png}")
    print(f"[SAVE] {fig2_pdf}")

    print("\n[DONE] Figure 1 and Figure 2 generated successfully.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

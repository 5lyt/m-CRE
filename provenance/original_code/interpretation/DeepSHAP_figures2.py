# -*- coding: utf-8 -*-
"""
DeepSHAP_figures_publication_v2.py

Publication-style plotting for DeepSHAP interpretability results.

Supported input:
INTERP_ROOT/{task}/{task}_sequence_importance.csv
INTERP_ROOT/{task}/{task}_mrna_feature_importance.csv
INTERP_ROOT/{task}/{task}_position_stability.csv

or
INTERP_ROOT/{model_name}/{task}/...

Updated 3022-bp coordinate definition from 221.py:
0-1000 promoter; 1000-1501 5'UTR; 1501-1523 N spacer;
1523-2023 coding; 2023-3022 3'UTR. If CSV Position starts at 1,
it is converted to 0-based Coord for plotting.
"""

import os
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

BASE_PATH = "/home"
INTERP_ROOT = os.path.join(BASE_PATH, "output_DeepCRE_mRNA_BiLSTM_chr", "interpretability_deepshap")
OUT_ROOT = os.path.join(INTERP_ROOT, "summary_publication_figures_v2")

TASK_ORDER = [
    "Atha_leaf", "Atha_root", "Sbic_leaf", "Sbic_root",
    "Slyc_leaf", "Slyc_root", "Zmay_leaf", "Zmay_root",
]
SPECIES_ORDER = ["Zmay", "Sbic", "Slyc", "Atha"]
TISSUE_ORDER = ["leaf", "root"]
FEATURE_ORDER = [
    "UTR3GC", "CDSGC", "ORFEXONDENSITY", "UTR5GC",
    "INTRONLEN", "UTR3LEN", "UTR5LEN", "CDSLEN",
]
SPECIES_LABELS = {
    "Atha": "A. thaliana", "Sbic": "S. bicolor",
    "Slyc": "S. lycopersicum", "Zmay": "Z. mays",
}
SPECIES_SHORT = {"Atha": "A. tha", "Sbic": "S. bic", "Slyc": "S. lyc", "Zmay": "Z. may"}
SPECIES_COLORS = {"Atha": "#2E8B57", "Sbic": "#9370DB", "Slyc": "#CD5C5C", "Zmay": "#FF8C00"}
NT_COLORS = {"A": "#008000", "C": "#4169E1", "G": "#E41A1C", "T": "#D95F02"}

REGIONS = [
    (0, 1000, "Promoter\nRegion", "#fff7bc"),
    (1000, 1501, "5' UTR", "#fdd49e"),
    (1501, 1523, "N\nSpacer", "#d9d9d9"),
    (1523, 2023, "Coding\nSequence", "#fdbb84"),
    (2023, 3022, "3' UTR", "#fc8d59"),
]
TSS, TTS, SEQ_LEN = 1000, 2023, 3022
PROM_END, UTR5_END, SPACER_END, CODING_START = 1000, 1501, 1523, 1523
SIGMA_SEQ, SIGMA_NT, SIGMA_CV = 4, 4, 3
DPI = 600

plt.rcParams.update({
    "font.family": "Arial",
    "axes.edgecolor": "#333F4B",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_name(name):
    return str(name).replace(" ", "_").replace("/", "_")


def task_parts(task):
    sp, tissue = task.split("_", 1)
    return sp, tissue


def required_files_exist(root, task):
    d = os.path.join(root, task)
    return (
        os.path.isdir(d)
        and os.path.exists(os.path.join(d, f"{task}_sequence_importance.csv"))
        and os.path.exists(os.path.join(d, f"{task}_mrna_feature_importance.csv"))
    )


def discover_model_roots(interp_root):
    if any(required_files_exist(interp_root, t) for t in TASK_ORDER):
        return [("DeepCRE_mRNA_BiLSTM_chr", interp_root)]
    roots = []
    for name in sorted(os.listdir(interp_root)):
        p = os.path.join(interp_root, name)
        if os.path.isdir(p) and any(required_files_exist(p, t) for t in TASK_ORDER):
            roots.append((name, p))
    if not roots:
        # Allow one additional nesting level.
        for name in sorted(os.listdir(interp_root)):
            p1 = os.path.join(interp_root, name)
            if not os.path.isdir(p1):
                continue
            for sub in sorted(os.listdir(p1)):
                p2 = os.path.join(p1, sub)
                if os.path.isdir(p2) and any(required_files_exist(p2, t) for t in TASK_ORDER):
                    roots.append((f"{name}_{sub}", p2))
    if not roots:
        raise FileNotFoundError(f"No valid task folders found under {interp_root}")
    return roots


def standardize_position(df):
    out = df.copy()
    if "Position" not in out.columns:
        raise ValueError(f"Missing Position column: {list(out.columns)}")
    out["Position"] = pd.to_numeric(out["Position"], errors="coerce")
    out = out.dropna(subset=["Position"]).copy()
    out["Position"] = out["Position"].astype(int)
    if int(out["Position"].min()) >= 1:
        out["Coord"] = out["Position"] - 1
        out.attrs["coord_mode"] = "1-based Position -> 0-based Coord"
    else:
        out["Coord"] = out["Position"]
        out.attrs["coord_mode"] = "0-based Position"
    return out.sort_values("Coord").reset_index(drop=True)


def load_task(root, task):
    d = os.path.join(root, task)
    seq = standardize_position(pd.read_csv(os.path.join(d, f"{task}_sequence_importance.csv")))
    feat = pd.read_csv(os.path.join(d, f"{task}_mrna_feature_importance.csv"))
    cv_path = os.path.join(d, f"{task}_position_stability.csv")
    cv = standardize_position(pd.read_csv(cv_path)) if os.path.exists(cv_path) else None
    return seq, feat, cv


def smooth(x, sigma):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or sigma <= 0:
        return x
    return gaussian_filter1d(x, sigma=sigma, mode="nearest")


def savefig(fig, prefix):
    png, pdf = prefix + ".png", prefix + ".pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def style_ax(ax, grid_axis="both"):
    ax.grid(True, linestyle="--", alpha=0.25, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, colors="#333F4B")


def add_full_background(ax, alpha=0.38):
    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1e-12)
    ytxt = ymin + yrange * 0.08
    for s, e, label, color in REGIONS:
        ax.axvspan(s, e, color=color, alpha=alpha, zorder=0)
        ax.text((s + e) / 2, ytxt, label, ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1.0))
    ax.axvline(TSS, color="#777777", linestyle="--", linewidth=1.1)
    ax.axvline(TTS, color="#777777", linestyle="--", linewidth=1.1)
    ax.text(TSS, ymin + yrange * 0.015, "TSS", ha="center", va="bottom", fontsize=8.5)
    ax.text(TTS, ymin + yrange * 0.015, "TTS", ha="center", va="bottom", fontsize=8.5)


def add_broken_background(axl, axr, alpha=0.33):
    axl.axvspan(-1000, 0, color="#fff7bc", alpha=alpha, zorder=0)
    axl.axvspan(0, 501, color="#fdd49e", alpha=alpha, zorder=0)
    axl.axvline(0, color="#777777", linestyle="--", linewidth=1.2)
    axl.text(-500, 0.92, "Promoter", transform=axl.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0))
    axl.text(250, 0.92, "5' UTR", transform=axl.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0))
    axl.text(0, 0.03, "TSS", transform=axl.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)

    axr.axvspan(-500, 0, color="#fdbb84", alpha=alpha, zorder=0)
    axr.axvspan(0, 1000, color="#fc8d59", alpha=alpha, zorder=0)
    axr.axvline(0, color="#777777", linestyle="--", linewidth=1.2)
    axr.text(-250, 0.92, "Coding sequence", transform=axr.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0))
    axr.text(500, 0.92, "3' UTR", transform=axr.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0))
    axr.text(0, 0.03, "TTS", transform=axr.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)


def split_tss_tts(df, col, sigma):
    x = df[["Coord", col]].copy()
    x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna().sort_values("Coord")
    x["Smoothed"] = smooth(x[col].values, sigma)
    tss = x[(x["Coord"] >= 0) & (x["Coord"] <= 1501)].copy()
    tss["Rel"] = tss["Coord"] - TSS
    tts = x[(x["Coord"] >= 1523) & (x["Coord"] <= 3022)].copy()
    tts["Rel"] = tts["Coord"] - TTS
    return tss, tts


def prep_feature_summary(feat_map, out_csv):
    rows = []
    for sp, df in feat_map.items():
        tmp = df.copy()
        tmp["Species"] = sp
        tmp["SHAP"] = pd.to_numeric(tmp["SHAP"], errors="coerce")
        rows.append(tmp[["Species", "Feature", "SHAP"]])
    out = pd.concat(rows, ignore_index=True)
    out["Feature"] = pd.Categorical(out["Feature"], categories=FEATURE_ORDER, ordered=True)
    out = out.dropna(subset=["Feature", "SHAP"]).sort_values(["Feature", "Species"])
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def plot_feature_soft(df, tissue, prefix, label=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(FEATURE_ORDER))
    max_x = max(float(df["SHAP"].max()), 1e-12)
    for sp in SPECIES_ORDER:
        sub = df[df["Species"] == sp].set_index("Feature").reindex(FEATURE_ORDER)
        if sub.empty:
            continue
        x = sub["SHAP"].astype(float).values
        ok = np.isfinite(x)
        if ok.sum() == 0:
            continue
        ax.plot(x[ok], y[ok], color=SPECIES_COLORS[sp], lw=2.1, marker="o", ms=4.8,
                label=SPECIES_SHORT[sp], alpha=0.95, zorder=3)
        ax.fill_betweenx(y[ok], 0, x[ok], color=SPECIES_COLORS[sp], alpha=0.10, zorder=1)
        im = int(np.nanargmax(x))
        ax.text(x[im] + max_x * 0.018, y[im], f"{x[im]:.3f}", fontsize=7.5,
                color=SPECIES_COLORS[sp], va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(FEATURE_ORDER, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max_x * 1.18)
    ax.set_xlabel("Mean |SHAP|", fontsize=11)
    ax.set_ylabel("mRNA feature", fontsize=11)
    ax.set_title(f"{tissue.capitalize()} mRNA feature importance", fontsize=12.5, fontweight="bold")
    style_ax(ax)
    ax.legend(title="Species", fontsize=8.5, title_fontsize=9, frameon=True, loc="lower right")
    if label:
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return savefig(fig, prefix)


def plot_single_feature_bar(feat_df, task, prefix):
    df = feat_df.copy()
    df["Feature"] = pd.Categorical(df["Feature"], categories=FEATURE_ORDER, ordered=True)
    df["SHAP"] = pd.to_numeric(df["SHAP"], errors="coerce")
    df = df.dropna(subset=["Feature", "SHAP"]).sort_values("Feature")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    y = np.arange(len(df))
    ax.barh(y, df["SHAP"].values, color="#7AA6C2", edgecolor="#333333", lw=0.5, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Feature"].astype(str), fontsize=9)
    ax.invert_yaxis()
    max_x = max(float(df["SHAP"].max()), 1e-12)
    ax.set_xlim(0, max_x * 1.18)
    for yi, val in zip(y, df["SHAP"].values):
        ax.text(val + max_x * 0.015, yi, f"{val:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel("Mean |SHAP|", fontsize=10.5)
    ax.set_title(f"{task}: mRNA feature importance", fontsize=12, fontweight="bold")
    style_ax(ax, grid_axis="x")
    fig.tight_layout()
    return savefig(fig, prefix)


def plot_sequence_full(seq_map, tissue, prefix, label=None):
    fig, ax = plt.subplots(figsize=(13.6, 4.3))
    all_y = []
    for sp in SPECIES_ORDER:
        if sp not in seq_map:
            continue
        df = seq_map[sp].sort_values("Coord")
        y = smooth(pd.to_numeric(df["Total_SHAP"], errors="coerce").values, SIGMA_SEQ)
        ax.plot(df["Coord"].values, y, color=SPECIES_COLORS[sp], lw=1.8, alpha=0.92, label=SPECIES_SHORT[sp])
        all_y.extend(y[np.isfinite(y)])
    if all_y:
        ymax = max(all_y); ymin = min(all_y)
        pad = max((ymax - ymin) * 0.12, ymax * 0.08, 1e-8)
        ax.set_ylim(max(0, ymin - pad), ymax + pad)
    add_full_background(ax)
    ax.set_xlim(0, SEQ_LEN)
    ax.set_xlabel("Genomic position in concatenated sequence (bp)", fontsize=11)
    ax.set_ylabel("Mean |SHAP|", fontsize=11)
    ax.set_title(f"{tissue.capitalize()} sequence positional importance", fontsize=12.5, fontweight="bold")
    style_ax(ax)
    ax.legend(title="Species", fontsize=8.5, title_fontsize=9, frameon=True, ncol=2, loc="upper right")
    if label:
        ax.text(-0.07, 1.04, label, transform=ax.transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return savefig(fig, prefix)


def plot_sequence_broken(seq_map, tissue, prefix, label=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.95), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.04})
    axl, axr = axes
    all_y = []
    for sp in SPECIES_ORDER:
        if sp not in seq_map:
            continue
        tss, tts = split_tss_tts(seq_map[sp], "Total_SHAP", SIGMA_SEQ)
        axl.plot(tss["Rel"], tss["Smoothed"], color=SPECIES_COLORS[sp], lw=1.9, alpha=0.94, label=SPECIES_SHORT[sp])
        axr.plot(tts["Rel"], tts["Smoothed"], color=SPECIES_COLORS[sp], lw=1.9, alpha=0.94)
        all_y.extend(tss["Smoothed"].dropna())
        all_y.extend(tts["Smoothed"].dropna())
    if all_y:
        ymax = max(all_y); ymin = min(all_y)
        pad = max((ymax - ymin) * 0.14, ymax * 0.08, 1e-8)
        axl.set_ylim(max(0, ymin - pad), ymax + pad)
    add_broken_background(axl, axr)
    axl.set_xlim(-1000, 500); axr.set_xlim(-500, 1000)
    axl.set_xticks([-1000, -500, 0, 500]); axr.set_xticks([-500, 0, 500, 1000])
    axl.set_title("TSS-flanking segment", fontsize=11.2, fontweight="bold")
    axr.set_title("TTS-flanking segment", fontsize=11.2, fontweight="bold")
    axl.set_xlabel("Position relative to TSS (bp)", fontsize=10.5)
    axr.set_xlabel("Position relative to TTS (bp)", fontsize=10.5)
    axl.set_ylabel("Mean |SHAP|", fontsize=10.5)
    for ax in axes:
        style_ax(ax)
    axl.legend(title="Species", fontsize=8.5, title_fontsize=9, frameon=True, loc="upper left")
    fig.suptitle(f"{tissue.capitalize()} sequence positional importance", fontsize=12.5, fontweight="bold", y=1.02)
    if label:
        axl.text(-0.16, 1.15, label, transform=axl.transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return savefig(fig, prefix)


def prep_nt(seq_df, out_csv):
    out = seq_df[["Position", "Coord", "A", "T", "C", "G"]].copy()
    for nt in ["A", "C", "G", "T"]:
        out[nt] = pd.to_numeric(out[nt], errors="coerce")
        out[f"{nt}_smoothed"] = smooth(out[nt].values, SIGMA_NT)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def plot_nt_broken(nt_df, task, prefix, label=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.9), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.04})
    axl, axr = axes
    all_y = []
    for nt in ["A", "C", "G", "T"]:
        tmp = nt_df[["Coord", f"{nt}_smoothed"]].rename(columns={f"{nt}_smoothed": "value"})
        tss, tts = split_tss_tts(tmp, "value", 0)
        axl.plot(tss["Rel"], tss["Smoothed"], color=NT_COLORS[nt], lw=1.55, alpha=0.90, label=nt)
        axr.plot(tts["Rel"], tts["Smoothed"], color=NT_COLORS[nt], lw=1.55, alpha=0.90)
        all_y.extend(tss["Smoothed"].dropna()); all_y.extend(tts["Smoothed"].dropna())
    if all_y:
        ymax = max(all_y); ymin = min(all_y)
        pad = max((ymax - ymin) * 0.14, ymax * 0.08, 1e-8)
        axl.set_ylim(max(0, ymin - pad), ymax + pad)
    add_broken_background(axl, axr, alpha=0.30)
    axl.set_xlim(-1000, 500); axr.set_xlim(-500, 1000)
    axl.set_xticks([-1000, -500, 0, 500]); axr.set_xticks([-500, 0, 500, 1000])
    axl.set_title("TSS-flanking segment", fontsize=11.2, fontweight="bold")
    axr.set_title("TTS-flanking segment", fontsize=11.2, fontweight="bold")
    axl.set_xlabel("Position relative to TSS (bp)", fontsize=10.5)
    axr.set_xlabel("Position relative to TTS (bp)", fontsize=10.5)
    axl.set_ylabel("Mean |SHAP|", fontsize=10.5)
    for ax in axes:
        style_ax(ax)
    axl.legend(title="Nucleotide", fontsize=8.5, title_fontsize=9, frameon=True, loc="upper left", ncol=2)
    fig.suptitle(f"{task}: nucleotide-specific importance", fontsize=12.5, fontweight="bold", y=1.02)
    if label:
        axl.text(-0.16, 1.15, label, transform=axl.transAxes, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return savefig(fig, prefix)


def plot_cv_broken(cv_map, tissue, prefix):
    if not cv_map:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.04})
    axl, axr = axes
    plotted = False
    for sp in SPECIES_ORDER:
        if sp not in cv_map or cv_map[sp] is None or "CV" not in cv_map[sp].columns:
            continue
        tss, tts = split_tss_tts(cv_map[sp], "CV", SIGMA_CV)
        axl.plot(tss["Rel"], tss["Smoothed"], color=SPECIES_COLORS[sp], lw=1.45, alpha=0.90, label=SPECIES_SHORT[sp])
        axr.plot(tts["Rel"], tts["Smoothed"], color=SPECIES_COLORS[sp], lw=1.45, alpha=0.90)
        plotted = True
    if not plotted:
        plt.close(fig); return []
    add_broken_background(axl, axr, alpha=0.24)
    axl.set_xlim(-1000, 500); axr.set_xlim(-500, 1000)
    axl.set_xlabel("Position relative to TSS (bp)", fontsize=10.5)
    axr.set_xlabel("Position relative to TTS (bp)", fontsize=10.5)
    axl.set_ylabel("CV", fontsize=10.5)
    axl.set_title("TSS-flanking segment", fontsize=11.2, fontweight="bold")
    axr.set_title("TTS-flanking segment", fontsize=11.2, fontweight="bold")
    for ax in axes:
        style_ax(ax)
    axl.legend(title="Species", fontsize=8.5, title_fontsize=9, frameon=True, loc="upper left")
    fig.suptitle(f"{tissue.capitalize()} positional stability (CV)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return savefig(fig, prefix)


def pearson_matrix(vectors):
    order = [sp for sp in SPECIES_ORDER if sp in vectors]
    mat = np.full((len(order), len(order)), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            x, y = np.asarray(vectors[a], float), np.asarray(vectors[b], float)
            n = min(len(x), len(y)); x, y = x[:n], y[:n]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                mat[i, j] = np.corrcoef(x[ok], y[ok])[0, 1]
    labels = [SPECIES_SHORT[sp] for sp in order]
    return pd.DataFrame(mat, index=labels, columns=labels)


def plot_heatmap(df, title, prefix):
    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    im = ax.imshow(df.values, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(df.shape[1])); ax.set_yticks(range(df.shape[0]))
    ax.set_xticklabels(df.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(df.index, fontsize=9)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.values[i, j]
            ax.text(j, i, "NA" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if np.isfinite(val) and val >= 0.75 else "#333333")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = savefig(fig, prefix)
    df.to_csv(prefix + ".csv", encoding="utf-8-sig")
    return out + [prefix + ".csv"]


def run_one_model(model_name, model_root):
    out_dir = os.path.join(OUT_ROOT, clean_name(model_name))
    core = os.path.join(out_dir, "core")
    per = os.path.join(out_dir, "per_task")
    heat = os.path.join(out_dir, "heatmaps")
    csvd = os.path.join(out_dir, "csv")
    for d in [out_dir, core, per, heat, csvd]:
        ensure_dir(d)

    tasks, failures = {}, []
    by_feat, by_seq, by_cv = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for task in TASK_ORDER:
        try:
            if not required_files_exist(model_root, task):
                failures.append({"task": task, "error": "missing required files"}); continue
            seq, feat, cv = load_task(model_root, task)
            sp, tissue = task_parts(task)
            tasks[task] = {"seq": seq, "feat": feat, "cv": cv}
            by_feat[tissue][sp] = feat
            by_seq[tissue][sp] = seq
            if cv is not None:
                by_cv[tissue][sp] = cv
        except Exception as e:
            failures.append({"task": task, "error": repr(e)})

    outputs = []
    for tissue in TISSUE_ORDER:
        if not by_feat[tissue] or not by_seq[tissue]:
            continue
        feat_df = prep_feature_summary(by_feat[tissue], os.path.join(csvd, f"{tissue}_feature_importance_summary.csv"))
        outputs += plot_feature_soft(feat_df, tissue, os.path.join(core, f"{tissue}_feature_importance_soft"), "a" if tissue == "leaf" else "b")
        outputs += plot_sequence_broken(by_seq[tissue], tissue, os.path.join(core, f"{tissue}_sequence_importance_broken_axis"), "c" if tissue == "leaf" else "d")
        outputs += plot_sequence_full(by_seq[tissue], tissue, os.path.join(core, f"{tissue}_sequence_importance_full_context"))
        outputs += plot_cv_broken(by_cv[tissue], tissue, os.path.join(core, f"{tissue}_position_stability_cv_broken_axis"))

        try:
            fv = {sp: df.set_index("Feature").reindex(FEATURE_ORDER)["SHAP"].astype(float).values for sp, df in by_feat[tissue].items()}
            outputs += plot_heatmap(pearson_matrix(fv), f"{tissue.capitalize()} feature similarity", os.path.join(heat, f"{tissue}_feature_similarity_heatmap"))
        except Exception as e:
            failures.append({"task": f"{tissue}_feature_heatmap", "error": repr(e)})
        try:
            sv = {sp: smooth(df.sort_values("Coord")["Total_SHAP"].astype(float).values, SIGMA_SEQ) for sp, df in by_seq[tissue].items()}
            outputs += plot_heatmap(pearson_matrix(sv), f"{tissue.capitalize()} sequence similarity", os.path.join(heat, f"{tissue}_sequence_similarity_heatmap"))
        except Exception as e:
            failures.append({"task": f"{tissue}_sequence_heatmap", "error": repr(e)})

    if "Zmay_leaf" in tasks:
        nt = prep_nt(tasks["Zmay_leaf"]["seq"], os.path.join(csvd, "Zmay_leaf_nucleotide_importance_summary.csv"))
        outputs += plot_nt_broken(nt, "Zmay_leaf", os.path.join(core, "Zmay_leaf_nucleotide_importance_broken_axis"), "e")

    for task, dct in tasks.items():
        task_out = os.path.join(per, task); ensure_dir(task_out)
        outputs += plot_single_feature_bar(dct["feat"], task, os.path.join(task_out, f"{task}_feature_importance_bar"))
        sp, _ = task_parts(task)
        outputs += plot_sequence_broken({sp: dct["seq"]}, task, os.path.join(task_out, f"{task}_sequence_importance_broken_axis"))
        outputs += plot_sequence_full({sp: dct["seq"]}, task, os.path.join(task_out, f"{task}_sequence_importance_full_context"))

    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(out_dir, "figure_generation_failures.csv"), index=False, encoding="utf-8-sig")
    manifest = {
        "model_name": model_name,
        "model_root": model_root,
        "output_dir": out_dir,
        "coordinate_definition": "0-1000, 1000-1501, 1501-1523, 1523-2023, 2023-3022; CSV Position>=1 is converted to Coord=Position-1",
        "outputs": [os.path.relpath(x, out_dir) for x in outputs],
        "failures": failures,
    }
    with open(os.path.join(out_dir, "generation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main():
    ensure_dir(OUT_ROOT)
    roots = discover_model_roots(INTERP_ROOT)
    print("Discovered model roots:")
    for name, root in roots:
        print(f"  - {name}: {root}")
    manifests = []
    for name, root in roots:
        print("\n" + "=" * 80)
        print(f"Processing {name}")
        try:
            manifests.append(run_one_model(name, root))
        except Exception as e:
            manifests.append({"model_name": name, "model_root": root, "error": repr(e)})
            print(f"[ERROR] {name} failed: {e}")
    with open(os.path.join(OUT_ROOT, "all_models_generation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()

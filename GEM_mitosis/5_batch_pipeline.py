#!/usr/bin/env python3
"""
5_batch_pipeline.py

Batch pipeline for mitosis GEM analysis across 4 experiments.

- set1_3em_001: copies existing diffusion_per_cell.csv / cell_classification.csv
- sets 002, 003, 004: runs the full step 1-3 pipeline logic
- All 4: pooled analysis saved to pooled_results_v2/
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import tifffile
from pathlib import Path
from skimage import measure
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

DT                   = 0.1      # seconds per frame
PIXEL_TO_MICRON      = 0.094    # µm / pixel
MIN_TRAJ_LENGTH      = 10       # minimum frames per trajectory
MIN_TRAJ_PER_CELL    = 5        # minimum valid trajectories per cell
CELL_AREA_PERCENTILE = 10       # drop cells below this area percentile
MSD_FIT_FRACTION     = 0.8      # fraction of trajectory length used for MSD lags
MAX_POINTS_FOR_FIT   = 11       # cap on number of MSD points used in fit
CIRCULARITY_THRESHOLD = 0.70    # mitotic = circularity >= threshold

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "label":    "set1_3em_001",
        "traj_csv": SCRIPT_DIR / "Traj_set1_3em_001_crop.tif.csv",
        "mem_tif":  SCRIPT_DIR / "set1_3em_001.nd2membrane image.tif",
        "mask_png": SCRIPT_DIR / "set1_3em_001.nd2membrane image_cp_masks.png",
        "precomputed": True,   # copy existing outputs, skip processing
    },
    {
        "label":    "set1_3em_002",
        "traj_csv": SCRIPT_DIR / "Traj_set1_3em_002_crop.csv",
        "mem_tif":  SCRIPT_DIR / "set1_3em_002.nd2membrane image.tif",
        "mask_png": SCRIPT_DIR / "set1_3em_002.nd2membrane image_cp_masks.png",
        "precomputed": False,
    },
    {
        "label":    "set1_3em_003",
        "traj_csv": SCRIPT_DIR / "Traj_set1_3em_003_crop.csv",
        "mem_tif":  SCRIPT_DIR / "set1_3em_003.nd2membrane image.tif",
        "mask_png": SCRIPT_DIR / "set1_3em_003.nd2membrane image_cp_masks.png",
        "precomputed": False,
    },
    {
        "label":    "set1_3em_004",
        "traj_csv": SCRIPT_DIR / "Traj_set1_3em_004_crop.csv",
        "mem_tif":  SCRIPT_DIR / "set1_3em_004.nd2membrane image.tif",
        "mask_png": SCRIPT_DIR / "set1_3em_004.nd2membrane image_cp_masks.png",
        "precomputed": False,
    },
]

EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# MSD / diffusion helpers
# ---------------------------------------------------------------------------

def compute_msd(x_um, y_um, max_lag):
    """Return (time_lags_s, msd_um2) arrays for lags 1..max_lag."""
    lags, msd = [], []
    for lag in range(1, max_lag + 1):
        dx = x_um[lag:] - x_um[:-lag]
        dy = y_um[lag:] - y_um[:-lag]
        msd.append(np.mean(dx**2 + dy**2))
        lags.append(lag * DT)
    return np.array(lags), np.array(msd)


def linear_msd(t, D, offset):
    return 4.0 * D * t + offset


def fit_msd(lags, msd):
    n_pts = min(int(len(lags) * MSD_FIT_FRACTION), MAX_POINTS_FOR_FIT, len(lags))
    n_pts = max(n_pts, 3)
    t_fit   = lags[:n_pts]
    msd_fit = msd[:n_pts]
    valid   = ~np.isnan(msd_fit)
    t_fit, msd_fit = t_fit[valid], msd_fit[valid]
    nan_result = dict(D=np.nan, offset=np.nan, D_err=np.nan, r_squared=np.nan)
    if len(t_fit) < 3:
        return nan_result
    try:
        popt, pcov = curve_fit(linear_msd, t_fit, msd_fit,
                               p0=[0.01, 0.0], maxfev=2000)
        D, offset = popt
        fit_vals  = linear_msd(t_fit, D, offset)
        ss_res = np.sum((msd_fit - fit_vals)**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        D_err = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.nan
        return dict(D=D, offset=offset, D_err=D_err, r_squared=r2)
    except Exception:
        return nan_result


# ---------------------------------------------------------------------------
# Step 1: classify trajectories to cells
# ---------------------------------------------------------------------------

def run_step1(exp, out_dir):
    """Load trajectories + mask, assign trajectories to cells, compute shape metrics."""
    print(f"  [Step 1] Loading data for {exp['label']} ...")

    df_raw = pd.read_csv(exp["traj_csv"])
    df_raw.columns = df_raw.columns.str.strip()
    print(f"    {len(df_raw):,} localisation rows, {df_raw['Trajectory'].nunique():,} trajectories")

    membrane = tifffile.imread(str(exp["mem_tif"])).astype(float)
    membrane = (membrane - membrane.min()) / (membrane.max() - membrane.min() + 1e-12)

    mask = np.array(Image.open(str(exp["mask_png"])))  # uint16, shape (H, W)
    H, W = mask.shape
    n_cells = int(mask.max())
    print(f"    Image size: {W} x {H} px,  {n_cells} labelled cells")

    # Classify each localisation by mask lookup
    x_px = np.clip(df_raw["x"].values, 0, W - 1)
    y_px = np.clip(df_raw["y"].values, 0, H - 1)
    xi = np.round(x_px).astype(int)
    yi = np.round(y_px).astype(int)
    df_raw["cell_label"] = mask[yi, xi]

    # Majority-vote per trajectory
    traj_cell = (
        df_raw[df_raw["cell_label"] > 0]
        .groupby("Trajectory")["cell_label"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("cell_label")
        .reset_index()
    )

    df = df_raw.merge(traj_cell, on="Trajectory", suffixes=("_px", ""))
    df = df[df["cell_label_px"] > 0].copy()

    n_assigned = df["Trajectory"].nunique()
    print(f"    {n_assigned:,} trajectories assigned to a cell")

    # Per-cell shape metrics
    props = measure.regionprops(mask)
    shape_records = []
    for p in props:
        area_px = p.area
        perim_px = p.perimeter
        circularity = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else np.nan
        cy, cx = p.centroid
        shape_records.append({
            "cell_label":    p.label,
            "area_px":       area_px,
            "perimeter_px":  perim_px,
            "circularity":   circularity,
            "centroid_x":    cx,
            "centroid_y":    cy,
        })
    df_shape = pd.DataFrame(shape_records)

    # Build per-cell trajectory summary
    traj_counts = (
        df.groupby("cell_label")["Trajectory"].nunique()
        .reset_index()
        .rename(columns={"Trajectory": "n_trajectories"})
    )
    point_counts = (
        df.groupby("cell_label").size()
        .reset_index()
        .rename(columns={0: "n_points"})
    )
    summary = (
        df_shape
        .merge(traj_counts, on="cell_label", how="left")
        .merge(point_counts, on="cell_label", how="left")
        .fillna({"n_trajectories": 0, "n_points": 0})
    )
    summary["n_trajectories"] = summary["n_trajectories"].astype(int)
    summary["n_points"]       = summary["n_points"].astype(int)
    summary = summary.sort_values("cell_label").reset_index(drop=True)

    # Save outputs
    summary.to_csv(out_dir / "cell_trajectory_summary.csv", index=False)

    classified_data = {
        "trajectories_df": df,
        "traj_cell_map":   traj_cell,
        "cell_summary":    summary,
        "mask":            mask,
        "membrane":        membrane,
    }
    with open(out_dir / "cell_trajectories.pkl", "wb") as f:
        pickle.dump(classified_data, f)

    print(f"    Saved cell_trajectory_summary.csv and cell_trajectories.pkl")
    return classified_data


# ---------------------------------------------------------------------------
# Step 2: diffusion per cell
# ---------------------------------------------------------------------------

def run_step2(classified_data, out_dir):
    """Compute per-trajectory and per-cell diffusion coefficients."""
    print(f"  [Step 2] Fitting MSD ...")

    df_all       = classified_data["trajectories_df"]
    cell_summary = classified_data["cell_summary"].copy()

    records = []
    grouped = df_all.sort_values("Frame").groupby("Trajectory")

    for traj_id, grp in grouped:
        grp = grp.sort_values("Frame")
        cell_lbl = grp["cell_label"].iloc[0]
        n_frames = len(grp)

        if n_frames < MIN_TRAJ_LENGTH:
            records.append(dict(Trajectory=traj_id, cell_label=cell_lbl,
                                n_frames=n_frames, D=np.nan, r_squared=np.nan,
                                filtered="short"))
            continue

        x_um = grp["x"].values * PIXEL_TO_MICRON
        y_um = grp["y"].values * PIXEL_TO_MICRON

        max_lag = max(2, int(n_frames * MSD_FIT_FRACTION))
        lags, msd = compute_msd(x_um, y_um, max_lag)
        fit = fit_msd(lags, msd)

        reason = "ok"
        if np.isnan(fit["D"]):
            reason = "fit_failed"
        elif fit["D"] <= 0:
            reason = "negative_D"

        records.append(dict(
            Trajectory = traj_id,
            cell_label = cell_lbl,
            n_frames   = n_frames,
            D          = fit["D"] if reason == "ok" else np.nan,
            D_err      = fit.get("D_err", np.nan),
            r_squared  = fit["r_squared"],
            filtered   = reason,
        ))

    df_traj  = pd.DataFrame(records)
    df_valid = df_traj[df_traj["filtered"] == "ok"].copy()
    n_valid  = len(df_valid)
    n_total  = len(df_traj)
    print(f"    {n_valid:,} / {n_total:,} trajectories pass filters")

    # Per-cell aggregation
    def iqr(x):
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    cell_stats = (
        df_valid.groupby("cell_label")["D"]
        .agg(
            n_valid_traj="count",
            D_median=lambda x: np.nanmedian(x),
            D_mean=np.nanmean,
            D_std=np.nanstd,
            D_iqr=iqr,
            D_p25=lambda x: np.nanpercentile(x, 25),
            D_p75=lambda x: np.nanpercentile(x, 75),
        )
        .reset_index()
    )

    df_cells = cell_summary.merge(cell_stats, on="cell_label", how="left")
    df_cells = df_cells.fillna({"n_valid_traj": 0})

    # Filters
    mask_traj = df_cells["n_valid_traj"] >= MIN_TRAJ_PER_CELL
    area_ref  = df_cells.loc[mask_traj, "area_px"]
    if len(area_ref) == 0:
        area_threshold = 0
    else:
        area_threshold = np.percentile(area_ref, CELL_AREA_PERCENTILE)
    mask_area = df_cells["area_px"] >= area_threshold
    mask_both = mask_traj & mask_area
    df_filtered = df_cells[mask_both].copy()

    print(f"    All cells: {len(df_cells)}  |  Both filters: {mask_both.sum()} retained")

    df_filtered.to_csv(out_dir / "diffusion_per_cell.csv", index=False)
    df_traj.to_csv(out_dir / "diffusion_per_traj.csv", index=False)
    print(f"    Saved diffusion_per_cell.csv ({len(df_filtered)} rows)")

    return df_filtered, df_valid


# ---------------------------------------------------------------------------
# Step 3: classify mitotic vs interphase
# ---------------------------------------------------------------------------

def run_step3(df_filtered, out_dir):
    """Classify cells by circularity threshold and save cell_classification.csv."""
    print(f"  [Step 3] Classifying cells ...")

    df = df_filtered.dropna(subset=["D_median", "circularity"]).copy()
    df["cell_type"] = np.where(df["circularity"] >= CIRCULARITY_THRESHOLD,
                               "mitotic", "interphase")

    n_mit = (df["cell_type"] == "mitotic").sum()
    n_int = (df["cell_type"] == "interphase").sum()
    print(f"    Mitotic: {n_mit}  |  Interphase: {n_int}")

    df.to_csv(out_dir / "cell_classification.csv", index=False)
    print(f"    Saved cell_classification.csv")

    return df


# ---------------------------------------------------------------------------
# Main per-experiment processing loop
# ---------------------------------------------------------------------------

exp_subdirs = {}

for exp in EXPERIMENTS:
    label   = exp["label"]
    out_dir = EXPERIMENTS_DIR / label
    out_dir.mkdir(exist_ok=True)
    exp_subdirs[label] = out_dir

    print(f"\n{'='*60}")
    print(f"Processing experiment: {label}")
    print(f"  Output dir: {out_dir}")

    if exp["precomputed"]:
        # Copy existing outputs from the root directory
        src_diff  = SCRIPT_DIR / "diffusion_per_cell.csv"
        src_class = SCRIPT_DIR / "cell_classification.csv"
        shutil.copy2(src_diff,  out_dir / "diffusion_per_cell.csv")
        shutil.copy2(src_class, out_dir / "cell_classification.csv")
        print(f"  [set1_3em_001] Copied existing diffusion_per_cell.csv and cell_classification.csv")
    else:
        classified_data = run_step1(exp, out_dir)
        df_filtered, df_valid = run_step2(classified_data, out_dir)
        run_step3(df_filtered, out_dir)

print(f"\n{'='*60}")
print("All experiments processed. Running pooled analysis (Step 4)...")


# ---------------------------------------------------------------------------
# Step 4: pooled analysis
# ---------------------------------------------------------------------------

OUTPUT_DIR = SCRIPT_DIR / "pooled_results_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

colours     = {"mitotic": "#e64b35", "interphase": "#4dbbd5"}
exp_labels  = sorted(exp_subdirs.keys())
exp_colours = {e: plt.cm.Set2(i / max(len(exp_labels) - 1, 1))
               for i, e in enumerate(exp_labels)}

# Load all experiments
frames = []
for label in exp_labels:
    sub_dir  = exp_subdirs[label]
    csv_diff = sub_dir / "diffusion_per_cell.csv"
    csv_cls  = sub_dir / "cell_classification.csv"

    df_exp = pd.read_csv(csv_diff)

    if csv_cls.exists():
        df_cls = pd.read_csv(csv_cls)[["cell_label", "cell_type"]]
        df_exp = df_exp.merge(df_cls, on="cell_label", how="left")
        print(f"  [{label}] {len(df_exp)} cells (with Step 3 labels)")
    else:
        print(f"  [{label}] {len(df_exp)} cells (no Step 3 labels — will classify)")

    df_exp["experiment"] = label
    df_exp["cell_uid"]   = label + "_" + df_exp["cell_label"].astype(str)
    frames.append(df_exp)

df_pool = pd.concat(frames, ignore_index=True)
df_pool = df_pool[df_pool["n_valid_traj"] >= MIN_TRAJ_PER_CELL].copy()
print(f"\nPooled dataset: {len(df_pool)} cells from {df_pool['experiment'].nunique()} experiment(s)")

# Classify any cells without cell_type
need_classif = df_pool["cell_type"].isna() if "cell_type" in df_pool.columns \
               else pd.Series(True, index=df_pool.index)

if need_classif.any() or "cell_type" not in df_pool.columns:
    df_pool["cell_type"] = np.where(
        df_pool["circularity"] >= CIRCULARITY_THRESHOLD, "mitotic", "interphase"
    )

n_mit = (df_pool["cell_type"] == "mitotic").sum()
n_int = (df_pool["cell_type"] == "interphase").sum()
print(f"Classification:  {n_mit} mitotic,  {n_int} interphase")

df_pool.to_csv(OUTPUT_DIR / "pooled_cells.csv", index=False)
print(f"Saved pooled_cells.csv")


def compare_groups(a, b, label_a="A", label_b="B"):
    a, b = np.array(a)[~np.isnan(np.array(a))], np.array(b)[~np.isnan(np.array(b))]
    u, p_mw  = stats.mannwhitneyu(a, b, alternative="two-sided")
    ks, p_ks = stats.ks_2samp(a, b)
    return dict(
        label_a=label_a, n_a=len(a),
        median_a=np.median(a), iqr_a=np.percentile(a, 75) - np.percentile(a, 25),
        label_b=label_b, n_b=len(b),
        median_b=np.median(b), iqr_b=np.percentile(b, 75) - np.percentile(b, 25),
        U=u, p_mw=p_mw, KS=ks, p_ks=p_ks,
        fold_change=np.median(b) / np.median(a) if np.median(a) != 0 else np.nan,
    )


# ---------------------------------------------------------------------------
# Figure 1: Pooled correlation
# ---------------------------------------------------------------------------
print("\nGenerating pooled correlation figure ...")

df_valid_pool = df_pool.dropna(subset=["D_median", "circularity"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=130)

ax = axes[0]
for ct in ["mitotic", "interphase"]:
    sub = df_valid_pool[df_valid_pool["cell_type"] == ct]
    for exp, grp in sub.groupby("experiment"):
        ax.scatter(grp["circularity"], grp["D_median"],
                   color=colours[ct], marker="o" if ct == "mitotic" else "s",
                   s=50, alpha=0.8, edgecolors=exp_colours[exp][:3],
                   linewidths=1.2, label=f"{ct} ({exp})", zorder=3)

x_c = df_valid_pool["circularity"].values
y_d = df_valid_pool["D_median"].values
slope, intercept, r, p, _ = stats.linregress(x_c, y_d)
x_line = np.array([x_c.min(), x_c.max()])
ax.plot(x_line, slope * x_line + intercept, "k-", lw=1.5,
        label=f"r = {r:.2f},  p = {p:.3f}", zorder=2)
ax.axvline(CIRCULARITY_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.7,
           label=f"threshold = {CIRCULARITY_THRESHOLD:.2f}")
ax.set_xlabel("Circularity", fontsize=10)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Diffusion vs Circularity (pooled)", fontsize=10)
handles, lbls = ax.get_legend_handles_labels()
by_label = dict(zip(lbls, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2)

ax = axes[1]
for ct in ["mitotic", "interphase"]:
    sub = df_valid_pool[df_valid_pool["cell_type"] == ct]
    for exp, grp in sub.groupby("experiment"):
        ax.scatter(grp["area_px"], grp["D_median"],
                   color=colours[ct], marker="o" if ct == "mitotic" else "s",
                   s=50, alpha=0.8, edgecolors=exp_colours[exp][:3],
                   linewidths=1.2, zorder=3)

x_a = df_valid_pool["area_px"].values
slope2, intercept2, r2, p2, _ = stats.linregress(x_a, y_d)
ax.plot(np.array([x_a.min(), x_a.max()]),
        slope2 * np.array([x_a.min(), x_a.max()]) + intercept2,
        "k-", lw=1.5, label=f"r = {r2:.2f},  p = {p2:.3f}", zorder=2)
ax.set_xlabel("Cell area  (px²)", fontsize=10)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Diffusion vs Cell Area (pooled)", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle(f"Pooled correlations  [{len(df_pool)} cells, "
             f"{df_pool['experiment'].nunique()} experiment(s)]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pooled_correlation_diffusion.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved pooled_correlation_diffusion.png")


# ---------------------------------------------------------------------------
# Figure 2: Mitotic vs Interphase (pooled)
# ---------------------------------------------------------------------------
print("Generating pooled mitotic vs interphase figure ...")

mit_D = df_pool.loc[df_pool["cell_type"] == "mitotic",    "D_median"].dropna().values
int_D = df_pool.loc[df_pool["cell_type"] == "interphase", "D_median"].dropna().values
comp  = compare_groups(mit_D, int_D, "mitotic", "interphase")

p_val = comp["p_mw"]
sig   = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
         ("*" if p_val < 0.05 else "ns"))

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), dpi=130)

ax = axes2[0]
parts = ax.violinplot([mit_D, int_D], positions=[1, 2],
                      showmedians=True, showextrema=False, widths=0.65)
for pc, col in zip(parts["bodies"], [colours["mitotic"], colours["interphase"]]):
    pc.set_facecolor(col); pc.set_alpha(0.75)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

rng = np.random.default_rng(7)
for pos, vals, col in [(1, mit_D, colours["mitotic"]),
                        (2, int_D, colours["interphase"])]:
    jit = rng.uniform(-0.09, 0.09, len(vals))
    ax.scatter(pos + jit, vals, s=35, color=col, alpha=0.9,
               edgecolors="white", linewidths=0.4, zorder=3)

y_max = max(np.nanmax(mit_D), np.nanmax(int_D)) * 1.06
ax.plot([1, 2], [y_max, y_max], "k-", lw=1)
ax.text(1.5, y_max * 1.02, sig, ha="center", va="bottom", fontsize=13)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Mitotic\n(n={len(mit_D)})", f"Interphase\n(n={len(int_D)})"])
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title(f"Mitotic vs Interphase\nMW p = {p_val:.4f}", fontsize=10)

ax = axes2[1]
x_pos = 0
xtick_pos, xtick_lbl = [], []
for exp in exp_labels:
    for i, ct in enumerate(["mitotic", "interphase"]):
        sub_vals = df_pool.loc[(df_pool["experiment"] == exp) &
                                (df_pool["cell_type"] == ct), "D_median"].dropna().values
        if len(sub_vals) == 0:
            continue
        x_pos += 1
        jit = rng.uniform(-0.1, 0.1, len(sub_vals))
        ax.scatter(x_pos + jit, sub_vals, s=25, color=colours[ct],
                   alpha=0.8, edgecolors="white", linewidths=0.3)
        ax.plot([x_pos - 0.25, x_pos + 0.25],
                [np.median(sub_vals)] * 2, color="black", lw=2)
        xtick_pos.append(x_pos)
        xtick_lbl.append(f"{exp[:8]}\n{ct[:3]}")
    x_pos += 0.5

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=7)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Per-experiment breakdown", fontsize=10)

ax = axes2[2]
for vals, col, lbl in [(mit_D, colours["mitotic"],    f"Mitotic (n={len(mit_D)})"),
                        (int_D, colours["interphase"], f"Interphase (n={len(int_D)})")]:
    sv = np.sort(vals)
    ax.plot(sv, np.arange(1, len(sv) + 1) / len(sv), color=col, lw=2, label=lbl)
ax.set_xlabel("Median D  (µm²/s)", fontsize=10)
ax.set_ylabel("Cumulative fraction", fontsize=10)
ax.set_title("CDF — per-cell median D", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle(f"Pooled mitotic vs interphase GEM diffusion  [{df_pool['experiment'].nunique()} experiments]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "pooled_mitotic_vs_interphase.png", dpi=130, bbox_inches="tight")
plt.close(fig2)
print("  Saved pooled_mitotic_vs_interphase.png")


# ---------------------------------------------------------------------------
# Figure 3: Between-mitotic-cell comparison (pooled)
# ---------------------------------------------------------------------------
print("Generating pooled mitotic cell comparison figure ...")

df_mit = df_pool[(df_pool["cell_type"] == "mitotic") &
                 (df_pool["n_valid_traj"] >= MIN_TRAJ_PER_CELL)].copy()
df_mit = df_mit.sort_values("D_median").reset_index(drop=True)
n_mit_cells = len(df_mit)
print(f"  {n_mit_cells} mitotic cells with >= {MIN_TRAJ_PER_CELL} trajectories")

if n_mit_cells >= 2:
    fig3, ax3 = plt.subplots(figsize=(max(10, n_mit_cells * 0.55), 5), dpi=130)

    positions  = np.arange(1, n_mit_cells + 1)
    cmap_vals  = plt.cm.Reds(np.linspace(0.35, 0.9, n_mit_cells))

    ax3.bar(positions, df_mit["D_median"], color=cmap_vals, alpha=0.8,
            edgecolor="white", width=0.65, zorder=2)
    ax3.errorbar(positions, df_mit["D_median"],
                 yerr=[df_mit["D_median"] - df_mit["D_p25"],
                       df_mit["D_p75"] - df_mit["D_median"]],
                 fmt="none", color="black", capsize=3, linewidth=1, zorder=3)

    for pos, (_, row) in zip(positions, df_mit.iterrows()):
        exp_col = exp_colours[row["experiment"]]
        ax3.bar(pos, row["D_median"], color="none",
                edgecolor=exp_col, linewidth=2.5, width=0.65, zorder=4)

    ax3.set_xticks(positions)
    xticklabels = [
        f"C{int(row['cell_label'])}\n{row['experiment'][:6]}\ncirc={row['circularity']:.2f}"
        for _, row in df_mit.iterrows()
    ]
    ax3.set_xticklabels(xticklabels, fontsize=6, rotation=45, ha="right")
    ax3.set_ylabel("Median D  (µm²/s)", fontsize=10)
    ax3.set_title(
        f"Individual mitotic cell diffusion  [{n_mit_cells} cells, "
        f"{df_mit['experiment'].nunique()} experiment(s)]\n"
        f"Bar = median, error bars = IQR, border colour = experiment",
        fontsize=10,
    )

    if n_mit_cells >= 3:
        kw_stat, kw_p = stats.kruskal(*[
            [row["D_median"]] for _, row in df_mit.iterrows()
        ])
        ax3.text(0.98, 0.97,
                 f"Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f}",
                 transform=ax3.transAxes, fontsize=8,
                 ha="right", va="top", color="darkred",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=exp_colours[e], label=e)
                      for e in exp_labels]
    ax3.legend(handles=legend_handles, fontsize=8, title="Experiment",
               loc="upper left", framealpha=0.8)

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "pooled_mitotic_cell_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved pooled_mitotic_cell_comparison.png")


# ---------------------------------------------------------------------------
# Stats text report
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "pooled_stats.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("Step 4 — Pooled Experiment Analysis Report (v2)\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Experiments included ({len(EXPERIMENTS)}):\n")
    for exp in EXPERIMENTS:
        f.write(f"  {exp['label']}  (precomputed={exp['precomputed']})\n")
    f.write(f"\nTotal cells after filtering: {len(df_pool)}\n")
    f.write(f"  Mitotic (circularity >= {CIRCULARITY_THRESHOLD}): {n_mit} cells\n")
    f.write(f"  Interphase: {n_int} cells\n\n")
    f.write(f"Circularity threshold: {CIRCULARITY_THRESHOLD}\n\n")

    f.write("Correlation — D_median vs circularity:\n")
    f.write(f"  Pearson r = {r:.4f},  p = {p:.4e}\n\n")
    f.write("Correlation — D_median vs area:\n")
    f.write(f"  Pearson r = {r2:.4f},  p = {p2:.4e}\n\n")

    f.write("Mitotic vs Interphase (cell-level median D):\n")
    f.write(f"  Mitotic    median D = {comp['median_a']:.5f} µm²/s  (IQR {comp['iqr_a']:.5f})\n")
    f.write(f"  Interphase median D = {comp['median_b']:.5f} µm²/s  (IQR {comp['iqr_b']:.5f})\n")
    f.write(f"  Fold change (int/mit) = {comp['fold_change']:.3f}x\n")
    f.write(f"  Mann-Whitney U={comp['U']:.1f},  p={comp['p_mw']:.4e}  [{sig}]\n")
    f.write(f"  KS stat={comp['KS']:.4f},  p={comp['p_ks']:.4e}\n\n")

    f.write("Per-experiment breakdown:\n")
    for exp_lbl in exp_labels:
        m = df_pool.loc[(df_pool["experiment"] == exp_lbl) & (df_pool["cell_type"] == "mitotic"), "D_median"]
        i = df_pool.loc[(df_pool["experiment"] == exp_lbl) & (df_pool["cell_type"] == "interphase"), "D_median"]
        m_med = np.median(m.values) if len(m) > 0 else float("nan")
        i_med = np.median(i.values) if len(i) > 0 else float("nan")
        f.write(f"  {exp_lbl}: mitotic n={len(m)}, median={m_med:.5f} | "
                f"interphase n={len(i)}, median={i_med:.5f}\n")

print(f"Saved pooled_stats.txt")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 4 POOLED RESULTS SUMMARY")
print("=" * 55)
print(f"Experiments: {len(EXPERIMENTS)},  Total cells: {len(df_pool)}")
print(f"Mitotic: {n_mit},  Interphase: {n_int}")
print()
print(f"D vs circularity:  r = {r:.3f},  p = {p:.4f}")
print(f"D vs area:         r = {r2:.3f},  p = {p2:.4f}")
print()
print(f"Mitotic    median D = {comp['median_a']:.5f} um2/s")
print(f"Interphase median D = {comp['median_b']:.5f} um2/s")
print(f"Fold change (int/mit) = {comp['fold_change']:.2f}x")
print(f"Mann-Whitney p = {comp['p_mw']:.4e}  [{sig}]")
print()
print(f"Results saved to: {OUTPUT_DIR}/")
print("\nStep 4 complete.")

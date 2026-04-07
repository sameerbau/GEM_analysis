#!/usr/bin/env python3
"""
2_diffusion_per_cell.py

Step 2 of the mitosis GEM analysis pipeline.

For each cell (from Step 1), computes per-trajectory diffusion coefficients via
linear MSD fitting, then aggregates to per-cell statistics.  Applies two
relative filters:
  - Trajectory length  ≥ MIN_TRAJ_LENGTH frames
  - Valid trajectories per cell ≥ MIN_TRAJ_PER_CELL
  - Cell area ≥ CELL_AREA_PERCENTILE of the *occupied-cell* area distribution

Outputs:
  diffusion_per_cell.csv     – per-cell median D, IQR, n_valid, area, circularity
  diffusion_per_traj.csv     – per-trajectory D, R², cell_label
  diffusion_per_cell.pkl     – same data as PKL for downstream scripts
  diffusion_qc.png           – filter summary: how many cells/trajectories survive each cut
  diffusion_per_cell.png     – per-cell violin plots (D vs cell, sorted by median D)

Requires: cell_trajectories.pkl (output of 1_cell_trajectory_classifier.py)

Usage:
  python 2_diffusion_per_cell.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

IN_PKL  = SCRIPT_DIR / "cell_trajectories.pkl"

# Physics / acquisition
DT               = 0.1       # seconds per frame
PIXEL_TO_MICRON  = 0.094     # µm / pixel

# Trajectory quality filters
MIN_TRAJ_LENGTH  = 10        # minimum frames per trajectory
MIN_R2           = 0.0       # minimum R² for MSD fit (0 = keep all fits that converge)

# Cell-level filters
MIN_TRAJ_PER_CELL     = 5    # minimum valid trajectories per cell
CELL_AREA_PERCENTILE  = 10   # drop cells below this percentile of the area distribution
                              # (computed across cells that have ≥1 valid trajectory)

# MSD fitting window (mirrors 2diffusion_analyzer.py)
MAX_POINTS_FOR_FIT = 11
MSD_FIT_FRACTION   = 0.8     # use first 80% of trajectory length for MSD lags

# Output files
OUT_CSV_CELL  = SCRIPT_DIR / "diffusion_per_cell.csv"
OUT_CSV_TRAJ  = SCRIPT_DIR / "diffusion_per_traj.csv"
OUT_PKL       = SCRIPT_DIR / "diffusion_per_cell.pkl"
OUT_FIG_QC    = SCRIPT_DIR / "diffusion_qc.png"
OUT_FIG_MAIN  = SCRIPT_DIR / "diffusion_per_cell.png"

# ---------------------------------------------------------------------------
# MSD / diffusion helpers  (adapted from 2diffusion_analyzer.py)
# ---------------------------------------------------------------------------

def compute_msd(x_um, y_um, max_lag):
    """Return (time_lags_s, msd_um2) arrays for lags 1..max_lag."""
    n = len(x_um)
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
    """
    Fit MSD = 4*D*t + offset.  Returns dict with D (µm²/s), r_squared, etc.
    Returns NaN dict if fitting fails.
    """
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
# Load Step 1 data
# ---------------------------------------------------------------------------
print("Loading classified trajectory data …")
with open(IN_PKL, "rb") as f:
    data = pickle.load(f)

df_all     = data["trajectories_df"]   # localisation table with 'Trajectory', 'cell_label', 'x', 'y', 'Frame'
cell_summary = data["cell_summary"].copy()

print(f"  {df_all['Trajectory'].nunique():,} trajectories across "
      f"{df_all['cell_label'].nunique()} cells")

# ---------------------------------------------------------------------------
# Per-trajectory diffusion fitting
# ---------------------------------------------------------------------------
print("Fitting MSD for each trajectory …")

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
    elif fit["r_squared"] < MIN_R2:
        reason = "low_r2"
    elif fit["D"] <= 0:
        reason = "negative_D"

    records.append(dict(
        Trajectory  = traj_id,
        cell_label  = cell_lbl,
        n_frames    = n_frames,
        D           = fit["D"] if reason == "ok" else np.nan,
        D_err       = fit.get("D_err", np.nan),
        r_squared   = fit["r_squared"],
        filtered    = reason,
    ))

df_traj = pd.DataFrame(records)
n_total  = len(df_traj)
n_valid  = (df_traj["filtered"] == "ok").sum()
print(f"  {n_valid:,} / {n_total:,} trajectories pass quality filters")
print(f"  (short: {(df_traj['filtered']=='short').sum()}, "
      f"fit_failed: {(df_traj['filtered']=='fit_failed').sum()}, "
      f"negative_D: {(df_traj['filtered']=='negative_D').sum()})")

df_valid = df_traj[df_traj["filtered"] == "ok"].copy()

# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------
print("Aggregating per-cell statistics …")

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

# ---------------------------------------------------------------------------
# Relative filters
# ---------------------------------------------------------------------------
# 1. Cells that have at least MIN_TRAJ_PER_CELL valid trajectories
mask_traj = df_cells["n_valid_traj"] >= MIN_TRAJ_PER_CELL

# 2. Cell area ≥ CELL_AREA_PERCENTILE of area distribution
#    (computed only over cells with enough trajectories so tiny edge-cells
#     don't drag the threshold down)
area_ref = df_cells.loc[mask_traj, "area_px"]
area_threshold = np.percentile(area_ref, CELL_AREA_PERCENTILE)
mask_area = df_cells["area_px"] >= area_threshold

mask_both = mask_traj & mask_area
df_filtered = df_cells[mask_both].copy()

print(f"\nFilter summary:")
print(f"  All cells:              {len(df_cells)}")
print(f"  ≥{MIN_TRAJ_PER_CELL} valid trajectories:  {mask_traj.sum()}")
print(f"  Area ≥ {CELL_AREA_PERCENTILE}th pctile ({area_threshold:.0f} px²): {mask_area.sum()}")
print(f"  Both filters:           {mask_both.sum()} cells retained")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
df_filtered.to_csv(OUT_CSV_CELL, index=False)
df_traj.to_csv(OUT_CSV_TRAJ, index=False)
print(f"\nSaved {OUT_CSV_CELL.name}  ({len(df_filtered)} rows)")
print(f"Saved {OUT_CSV_TRAJ.name}  ({len(df_traj)} rows)")

with open(OUT_PKL, "wb") as f:
    pickle.dump({
        "cell_stats":   df_filtered,
        "traj_stats":   df_traj,
        "valid_traj":   df_valid,
        "area_threshold": area_threshold,
        "params": {
            "DT": DT,
            "PIXEL_TO_MICRON": PIXEL_TO_MICRON,
            "MIN_TRAJ_LENGTH": MIN_TRAJ_LENGTH,
            "MIN_TRAJ_PER_CELL": MIN_TRAJ_PER_CELL,
            "CELL_AREA_PERCENTILE": CELL_AREA_PERCENTILE,
        },
    }, f)
print(f"Saved {OUT_PKL.name}")

# ---------------------------------------------------------------------------
# QC figure  –  how many cells/trajectories survive each filter
# ---------------------------------------------------------------------------
print("\nGenerating QC figure …")

fig_qc, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)

# A. Distribution of valid trajectories per cell, with threshold line
ax = axes[0]
ax.hist(df_cells["n_valid_traj"], bins=40, color="steelblue", edgecolor="white")
ax.axvline(MIN_TRAJ_PER_CELL, color="red", lw=1.5, ls="--",
           label=f"min = {MIN_TRAJ_PER_CELL}")
ax.set_xlabel("Valid trajectories per cell")
ax.set_ylabel("# cells")
ax.set_title("Trajectory count per cell")
ax.legend()

# B. Distribution of cell areas with threshold line
ax = axes[1]
ax.hist(df_cells["area_px"], bins=40, color="darkorange", edgecolor="white")
ax.axvline(area_threshold, color="red", lw=1.5, ls="--",
           label=f"{CELL_AREA_PERCENTILE}th pctile = {area_threshold:.0f} px²")
ax.set_xlabel("Cell area (px²)")
ax.set_ylabel("# cells")
ax.set_title("Cell area distribution")
ax.legend()

# C. Pie chart: filter outcome
n_pass  = mask_both.sum()
n_small = (~mask_area & mask_traj).sum()
n_few   = (mask_area & ~mask_traj).sum()
n_both_fail = (~mask_area & ~mask_traj).sum()
ax = axes[2]
sizes  = [n_pass, n_few, n_small, n_both_fail]
labels = [f"Pass ({n_pass})", f"Too few traj ({n_few})",
          f"Too small ({n_small})", f"Both fail ({n_both_fail})"]
colours = ["mediumseagreen", "salmon", "gold", "lightgray"]
ax.pie([s for s in sizes if s > 0],
       labels=[l for s, l in zip(sizes, labels) if s > 0],
       colors=[c for s, c in zip(sizes, colours) if s > 0],
       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
ax.set_title("Cell filter outcome")

plt.suptitle("Step 2 — Quality filters", fontsize=13, fontweight="bold")
plt.tight_layout()
fig_qc.savefig(OUT_FIG_QC, dpi=120, bbox_inches="tight")
plt.close(fig_qc)
print(f"  Saved {OUT_FIG_QC.name}")

# ---------------------------------------------------------------------------
# Main figure  –  per-cell D distributions
# ---------------------------------------------------------------------------
print("Generating per-cell diffusion figure …")

# Sort cells by median D
df_plot = df_filtered.dropna(subset=["D_median"]).sort_values("D_median")
cell_order = df_plot["cell_label"].tolist()
n_cells_plot = len(cell_order)

# Collect per-cell D arrays for violin / box plots
cell_D = {}
for lbl in cell_order:
    d_vals = df_valid.loc[df_valid["cell_label"] == lbl, "D"].dropna().values
    cell_D[lbl] = d_vals

# --- figure layout ---
fig = plt.figure(figsize=(max(14, n_cells_plot * 0.25), 10), dpi=130)
gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1],
                         hspace=0.4, wspace=0.35)

ax_violin = fig.add_subplot(gs[0, :])   # full-width violin plot
ax_circ   = fig.add_subplot(gs[1, 0])   # circularity vs D_median
ax_area   = fig.add_subplot(gs[1, 1])   # area vs D_median

# ---- violin plot ----
positions = np.arange(1, n_cells_plot + 1)
parts = ax_violin.violinplot(
    [cell_D[lbl] for lbl in cell_order],
    positions=positions,
    showmedians=True, showextrema=False, widths=0.7,
)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.7)
parts["cmedians"].set_color("red")
parts["cmedians"].set_linewidth(1.5)

ax_violin.set_xticks(positions)
ax_violin.set_xticklabels(cell_order, rotation=90, fontsize=5)
ax_violin.set_xlabel("Cell label (sorted by median D)", fontsize=10)
ax_violin.set_ylabel("Diffusion coefficient  D  (µm²/s)", fontsize=10)
ax_violin.set_title(
    f"Per-cell GEM diffusion coefficients  "
    f"[{n_cells_plot} cells,  DT = {DT} s,  {PIXEL_TO_MICRON} µm/px]",
    fontsize=11, fontweight="bold",
)
ax_violin.set_xlim(0, n_cells_plot + 1)

# ---- circularity vs D_median ----
ax_circ.scatter(df_plot["circularity"], df_plot["D_median"],
                s=25, alpha=0.7, color="steelblue", edgecolors="none")
ax_circ.set_xlabel("Circularity", fontsize=10)
ax_circ.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax_circ.set_title("Circularity vs Diffusion", fontsize=10)

# quick Pearson r
from scipy.stats import pearsonr
valid_mask = df_plot["circularity"].notna() & df_plot["D_median"].notna()
if valid_mask.sum() > 3:
    r, p = pearsonr(df_plot.loc[valid_mask, "circularity"],
                    df_plot.loc[valid_mask, "D_median"])
    ax_circ.text(0.05, 0.93, f"r = {r:.2f},  p = {p:.3f}",
                 transform=ax_circ.transAxes, fontsize=8,
                 va="top", color="darkred")

# ---- area vs D_median ----
ax_area.scatter(df_plot["area_px"], df_plot["D_median"],
                s=25, alpha=0.7, color="darkorange", edgecolors="none")
ax_area.set_xlabel("Cell area  (px²)", fontsize=10)
ax_area.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax_area.set_title("Cell area vs Diffusion", fontsize=10)

if valid_mask.sum() > 3:
    r2, p2 = pearsonr(df_plot.loc[valid_mask, "area_px"],
                      df_plot.loc[valid_mask, "D_median"])
    ax_area.text(0.05, 0.93, f"r = {r2:.2f},  p = {p2:.3f}",
                 transform=ax_area.transAxes, fontsize=8,
                 va="top", color="darkred")

fig.savefig(OUT_FIG_MAIN, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_FIG_MAIN.name}")

# ---------------------------------------------------------------------------
# Quick text summary
# ---------------------------------------------------------------------------
print("\n=== Per-cell diffusion summary (top & bottom 10 by median D) ===")
cols = ["cell_label", "n_valid_traj", "area_px", "circularity",
        "D_median", "D_iqr"]
print("--- Lowest D (10) ---")
print(df_plot.head(10)[cols].to_string(index=False))
print("--- Highest D (10) ---")
print(df_plot.tail(10)[cols].to_string(index=False))

print("\nStep 2 complete.")
print("  Next step: run 3_circularity_diffusion_correlation.py")

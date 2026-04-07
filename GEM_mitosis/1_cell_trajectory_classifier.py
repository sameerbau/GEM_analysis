#!/usr/bin/env python3
"""
1_cell_trajectory_classifier.py

Step 1 of the mitosis GEM analysis pipeline.

Loads the cellpose label image (or ImageJ ROI zip) and classifies each GEM
trajectory into the cell it belongs to, based on the trajectory's mean (x, y)
position.  Produces:
  - A summary table (cell_trajectory_summary.csv) with per-cell trajectory counts
    and basic shape metrics (area, perimeter, circularity).
  - A diagnostic figure (cell_trajectory_classification.png) showing the label
    image with all classified trajectories overlaid, coloured by cell ID.

Input files (expected in the same directory as this script):
  Traj_set1_3em_001_crop.tif.csv          – TrackMate trajectory table
  set1_3em_001.nd2membrane image.tif      – membrane channel (background)
  set1_3em_001.nd2membrane image_cp_masks.png  – cellpose label image
  set1_3em_001.nd2membrane image_rois.zip – ImageJ ROI file (optional)

Usage:
  python 1_cell_trajectory_classifier.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image
import tifffile
from pathlib import Path
from skimage import measure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

TRAJ_CSV      = SCRIPT_DIR / "Traj_set1_3em_001_crop.tif.csv"
MEMBRANE_TIF  = SCRIPT_DIR / "set1_3em_001.nd2membrane image.tif"
MASK_PNG      = SCRIPT_DIR / "set1_3em_001.nd2membrane image_cp_masks.png"

# Minimum number of trajectory points (not trajectories) a cell must contain
# to appear in the output.  Set to 0 to keep all cells.
MIN_TRAJ_POINTS_PER_CELL = 0

# When plotting trajectories, subsample long ones to keep the figure readable
MAX_TRAJ_POINTS_PLOT = 50_000

# Output files
OUT_CSV = SCRIPT_DIR / "cell_trajectory_summary.csv"
OUT_FIG = SCRIPT_DIR / "cell_trajectory_classification.png"
OUT_PKL = SCRIPT_DIR / "cell_trajectories.pkl"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading trajectory data …")
df_raw = pd.read_csv(TRAJ_CSV)
# Normalise column names (strip whitespace)
df_raw.columns = df_raw.columns.str.strip()
print(f"  {len(df_raw):,} localisation rows, {df_raw['Trajectory'].nunique():,} trajectories")

print("Loading membrane image …")
membrane = tifffile.imread(str(MEMBRANE_TIF)).astype(float)
# Normalise for display
membrane = (membrane - membrane.min()) / (membrane.max() - membrane.min() + 1e-12)

print("Loading cellpose label image …")
mask = np.array(Image.open(str(MASK_PNG)))   # uint16, shape (H, W)
H, W = mask.shape
n_cells = int(mask.max())
print(f"  Image size: {W} × {H} px,  {n_cells} labelled cells")

# ---------------------------------------------------------------------------
# Classify each localisation to a cell by looking up mask[y, x]
# ---------------------------------------------------------------------------
print("Classifying localisations to cells …")

x_px = np.clip(df_raw["x"].values, 0, W - 1)
y_px = np.clip(df_raw["y"].values, 0, H - 1)
xi = np.round(x_px).astype(int)
yi = np.round(y_px).astype(int)

df_raw["cell_label"] = mask[yi, xi]          # 0 = background

# Classification based on majority vote per trajectory
traj_cell = (
    df_raw[df_raw["cell_label"] > 0]         # exclude background points
    .groupby("Trajectory")["cell_label"]
    .agg(lambda s: s.mode().iloc[0])          # most common cell label
    .rename("cell_label")
    .reset_index()
)

df = df_raw.merge(traj_cell, on="Trajectory", suffixes=("_px", ""))
# Keep only trajectories assigned to a cell
df = df[df["cell_label_px"] > 0].copy()      # drop background localisations
# For rows where the per-row label disagrees with the trajectory assignment, use
# the trajectory-level assignment (traj_cell) for consistency.
df["cell_label"] = df["cell_label"]           # already set by merge

n_assigned = df["Trajectory"].nunique()
print(f"  {n_assigned:,} trajectories assigned to a cell")
print(f"  {df_raw['Trajectory'].nunique() - n_assigned:,} trajectories in background (dropped)")

# ---------------------------------------------------------------------------
# Compute per-cell shape metrics from the mask
# ---------------------------------------------------------------------------
print("Computing cell shape metrics …")
props = measure.regionprops(mask)

shape_records = []
for p in props:
    area_px = p.area
    perim_px = p.perimeter
    circularity = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else np.nan
    cy, cx = p.centroid          # (row, col) = (y, x)
    shape_records.append({
        "cell_label": p.label,
        "area_px":    area_px,
        "perimeter_px": perim_px,
        "circularity": circularity,
        "centroid_x": cx,
        "centroid_y": cy,
    })

df_shape = pd.DataFrame(shape_records)

# ---------------------------------------------------------------------------
# Build per-cell trajectory summary
# ---------------------------------------------------------------------------
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

summary.to_csv(OUT_CSV, index=False)
print(f"  Saved summary table → {OUT_CSV.name}")

# Print a quick overview
print("\n=== Cell summary (first 20) ===")
print(summary[["cell_label","area_px","circularity","n_trajectories"]].head(20).to_string(index=False))

# ---------------------------------------------------------------------------
# Save classified trajectory dataframe as pickle for downstream scripts
# ---------------------------------------------------------------------------
import pickle

classified_data = {
    "trajectories_df": df,      # all localisations with 'Trajectory' and 'cell_label'
    "traj_cell_map": traj_cell, # Trajectory → cell_label mapping
    "cell_summary": summary,    # per-cell shape + counts
    "mask": mask,
    "membrane": membrane,
}
with open(OUT_PKL, "wb") as f:
    pickle.dump(classified_data, f)
print(f"  Saved classified data → {OUT_PKL.name}")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
print("\nGenerating classification figure …")

# Build a colourmap: one distinct colour per cell label
rng = np.random.default_rng(42)
cell_labels = np.arange(1, n_cells + 1)
colours = plt.cm.tab20(np.linspace(0, 1, 20))
cell_colours = {lbl: colours[i % 20] for i, lbl in enumerate(cell_labels)}
cell_colours[0] = (0, 0, 0, 0)   # background = transparent

fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=150)

# ---- Left panel: label image overview ----
ax = axes[0]
ax.imshow(membrane, cmap="gray", interpolation="nearest")

# Colour-fill each cell region
label_rgb = np.zeros((*mask.shape, 4), dtype=float)
for lbl, col in cell_colours.items():
    if lbl == 0:
        continue
    label_rgb[mask == lbl] = col

ax.imshow(label_rgb, alpha=0.45, interpolation="nearest")

# Label each cell with its ID
for _, row in summary.iterrows():
    if row["n_trajectories"] > 0:
        ax.text(row["centroid_x"], row["centroid_y"], str(int(row["cell_label"])),
                fontsize=4, ha="center", va="center", color="white",
                fontweight="bold")

ax.set_title(f"Cellpose label image\n{n_cells} cells  |  {W}×{H} px", fontsize=11)
ax.axis("off")

# ---- Right panel: trajectories coloured by cell ----
ax = axes[1]
ax.imshow(membrane, cmap="gray", interpolation="nearest")
ax.imshow(label_rgb, alpha=0.20, interpolation="nearest")

# Subsample to keep plot fast
df_plot = df.copy()
if len(df_plot) > MAX_TRAJ_POINTS_PLOT:
    df_plot = df_plot.sample(MAX_TRAJ_POINTS_PLOT, random_state=42)

for lbl in df_plot["cell_label"].unique():
    sub = df_plot[df_plot["cell_label"] == lbl]
    col = cell_colours.get(lbl, (0.5, 0.5, 0.5, 1))
    ax.scatter(sub["x"], sub["y"], s=0.3, color=col[:3], alpha=0.5, linewidths=0)

# Annotate cells with >0 trajectories
for _, row in summary[summary["n_trajectories"] > 0].iterrows():
    ax.text(row["centroid_x"], row["centroid_y"], str(int(row["n_trajectories"])),
            fontsize=4, ha="center", va="center", color="yellow",
            fontweight="bold")

ax.set_title(
    f"GEM trajectories classified by cell\n"
    f"{n_assigned:,} trajectories in {(summary['n_trajectories'] > 0).sum()} cells",
    fontsize=11,
)
ax.axis("off")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved figure → {OUT_FIG.name}")

# ---------------------------------------------------------------------------
# Distribution overview
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), dpi=120)

axes2[0].hist(summary["n_trajectories"], bins=30, color="steelblue", edgecolor="white")
axes2[0].set_xlabel("Trajectories per cell")
axes2[0].set_ylabel("# cells")
axes2[0].set_title("Trajectory count distribution")

axes2[1].hist(summary["area_px"], bins=30, color="darkorange", edgecolor="white")
axes2[1].set_xlabel("Cell area (px²)")
axes2[1].set_ylabel("# cells")
axes2[1].set_title("Cell area distribution")

axes2[2].hist(summary["circularity"].dropna(), bins=30, color="mediumseagreen", edgecolor="white")
axes2[2].set_xlabel("Circularity  (1 = perfect circle)")
axes2[2].set_ylabel("# cells")
axes2[2].set_title("Cell circularity distribution")

plt.suptitle("Per-cell overview", fontsize=13, fontweight="bold")
plt.tight_layout()
overview_fig = SCRIPT_DIR / "cell_overview_distributions.png"
fig2.savefig(overview_fig, dpi=120, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved distribution overview → {overview_fig.name}")

print("\nStep 1 complete.")
print(f"  Output files:")
print(f"    {OUT_CSV.name}")
print(f"    {OUT_PKL.name}")
print(f"    {OUT_FIG.name}")
print(f"    {overview_fig.name}")
print("\nNext step: run 2_diffusion_per_cell.py")

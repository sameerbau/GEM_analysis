#!/usr/bin/env python3
"""
1_cell_trajectory_classifier.py

Step 1 of the mitosis GEM analysis pipeline — BATCH FOLDER MODE.

Prompts the user for a folder that contains all input files for one or
more embryos, auto-detects the file sets, and processes every embryo found.

Expected files in the input folder (per embryo):
  *_cp_masks.png  or  *_cp_masks.tif   – Cellpose label image (REQUIRED)
  *.tif / *.tiff                         – membrane channel image (REQUIRED)
  Traj_*.csv                             – TrackMate trajectory table (REQUIRED)
  *_rois.zip                             – ImageJ ROI file (optional)

File matching:
  The mask file is used as the anchor.  From its name everything before
  "_cp_masks" is treated as the "base name".  The membrane TIF is the file
  with that same base name plus ".tif/.tiff".  The trajectory CSV is the
  file whose name contains the embryo key (the part before the first "."
  in the base name, e.g. "Em001" from "Em001.nd2membrane image").

Outputs (written to  <output_dir>/<embryo_name>/):
  cell_trajectory_summary.csv        – per-cell trajectory counts + shape metrics
  cell_trajectories.pkl              – classified data for Step 2
  cell_trajectory_classification.png – overlay figure
  cell_overview_distributions.png    – area / circularity / trajectory-count histograms

Usage:
  python 1_cell_trajectory_classifier.py
  (then enter paths when prompted)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from pathlib import Path
from skimage import measure

# ---------------------------------------------------------------------------
# Parameters (edit if needed, but usually leave as-is)
# ---------------------------------------------------------------------------
# Minimum trajectory *points* for a cell to appear in the summary
MIN_TRAJ_POINTS_PER_CELL = 0

# Maximum scatter points in the classification figure (keeps it fast)
MAX_TRAJ_POINTS_PLOT = 50_000

# Recognised mask file suffixes
MASK_SUFFIXES = ["_cp_masks.png", "_cp_masks.tif", "_cp_masks.tiff",
                 "_masks.png",    "_masks.tif"]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _strip_mask_suffix(name: str) -> str | None:
    """Return base name (without mask suffix) or None if not a mask file."""
    for sfx in MASK_SUFFIXES:
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return None


def find_embryo_sets(folder: Path) -> list[dict]:
    """
    Scan *folder* and group files into per-embryo dicts.

    Returns a list of dicts, each with keys:
      embryo_name  – short label used for the output subfolder
      mask         – Path to the Cellpose label image
      membrane     – Path to the membrane TIF (or None if not found)
      traj_csv     – Path to the TrackMate trajectory CSV (or None)
      roi_zip      – Path to the ImageJ ROI zip (or None)
    """
    all_files = sorted(folder.iterdir())

    # ---- Identify mask files → one entry per embryo ----
    embryos = []
    for f in all_files:
        base = _strip_mask_suffix(f.name)
        if base is None:
            continue

        # Embryo key: part before the first "." (e.g. "Em001" from
        # "Em001.nd2membrane image") — used to match traj CSV
        key = base.split(".")[0]

        # Membrane TIF: same base name + .tif / .tiff
        membrane = None
        for ext in [".tif", ".tiff"]:
            candidate = folder / (base + ext)
            if candidate.exists():
                membrane = candidate
                break

        # Trajectory CSV: Traj_*.csv files that contain the embryo key
        traj_candidates = sorted(folder.glob("Traj_*.csv"))
        traj_match = None
        for tc in traj_candidates:
            if key in tc.name:
                traj_match = tc
                break
        # Fallback: first Traj CSV in folder if only one exists
        if traj_match is None and len(traj_candidates) == 1:
            traj_match = traj_candidates[0]

        # ROI zip (optional)
        roi_candidates = sorted(folder.glob(f"{key}*_rois.zip"))
        roi_zip = roi_candidates[0] if roi_candidates else None

        embryos.append({
            "embryo_name": key,
            "mask":        f,
            "membrane":    membrane,
            "traj_csv":    traj_match,
            "roi_zip":     roi_zip,
        })

    return embryos


def summarise_found(embryos: list[dict]) -> None:
    """Print a table of what was auto-detected."""
    print(f"\n  Found {len(embryos)} embryo(s):\n")
    print(f"  {'Embryo':<20}  {'Mask':^5}  {'Membrane':^8}  {'Traj CSV':^8}  {'ROI':^5}")
    print("  " + "-" * 60)
    for em in embryos:
        print(f"  {em['embryo_name']:<20}  "
              f"{'OK':^5}  "
              f"{'OK' if em['membrane'] else 'MISS':^8}  "
              f"{'OK' if em['traj_csv'] else 'MISS':^8}  "
              f"{'OK' if em['roi_zip'] else '–':^5}")
    print()


# ---------------------------------------------------------------------------
# Core analysis for one embryo
# ---------------------------------------------------------------------------

def process_embryo(embryo_name: str,
                   traj_csv:   Path,
                   mask_path:  Path,
                   membrane_path: Path | None,
                   output_dir: Path) -> bool:
    """
    Run Step 1 for a single embryo.  Saves all outputs to *output_dir*.
    Returns True on success, False if an error occurred.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Processing: {embryo_name}")
    print(f"  Mask:       {mask_path.name}")
    print(f"  Trajectory: {traj_csv.name}")
    print(f"  Membrane:   {membrane_path.name if membrane_path else '(not found – using mask)'}")
    print(f"  Output dir: {output_dir}")

    try:
        # ---- Load trajectory data ----
        print("  Loading trajectory data …")
        df_raw = pd.read_csv(traj_csv)
        df_raw.columns = df_raw.columns.str.strip()
        if "Trajectory" not in df_raw.columns:
            raise ValueError("'Trajectory' column not found in CSV – is this a TrackMate export?")
        print(f"    {len(df_raw):,} localisation rows, "
              f"{df_raw['Trajectory'].nunique():,} trajectories")

        # ---- Load membrane image ----
        if membrane_path and membrane_path.exists():
            print("  Loading membrane image …")
            membrane = tifffile.imread(str(membrane_path)).astype(float)
        else:
            print("  Loading mask as stand-in for membrane image …")
            membrane = np.array(Image.open(str(mask_path))).astype(float)

        # Normalise for display (handle 2-D and multi-channel)
        if membrane.ndim == 3:
            membrane = membrane[..., :3].mean(axis=-1)
        vmin, vmax = membrane.min(), membrane.max()
        membrane = (membrane - vmin) / (vmax - vmin + 1e-12)

        # ---- Load Cellpose mask ----
        print("  Loading Cellpose label image …")
        mask_img = np.array(Image.open(str(mask_path)))
        # If RGB/RGBA, take first channel (label images are usually grey)
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        H, W = mask_img.shape
        n_cells = int(mask_img.max())
        print(f"    Image size: {W} × {H} px,  {n_cells} labelled cells")

        # ---- Classify localisations → cells ----
        print("  Classifying localisations to cells …")
        x_px = np.clip(df_raw["x"].values, 0, W - 1)
        y_px = np.clip(df_raw["y"].values, 0, H - 1)
        xi   = np.round(x_px).astype(int)
        yi   = np.round(y_px).astype(int)
        df_raw["cell_label"] = mask_img[yi, xi]

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
        n_total    = df_raw["Trajectory"].nunique()
        print(f"    {n_assigned:,} / {n_total:,} trajectories assigned to a cell")

        # ---- Per-cell shape metrics ----
        print("  Computing cell shape metrics …")
        props = measure.regionprops(mask_img)
        shape_records = []
        for p in props:
            area_px  = p.area
            perim_px = p.perimeter
            circ = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else np.nan
            cy, cx = p.centroid
            shape_records.append({
                "cell_label":   p.label,
                "area_px":      area_px,
                "perimeter_px": perim_px,
                "circularity":  circ,
                "centroid_x":   cx,
                "centroid_y":   cy,
            })
        df_shape = pd.DataFrame(shape_records)

        # ---- Build per-cell summary ----
        traj_counts = (df.groupby("cell_label")["Trajectory"].nunique()
                       .reset_index().rename(columns={"Trajectory": "n_trajectories"}))
        point_counts = (df.groupby("cell_label").size()
                        .reset_index().rename(columns={0: "n_points"}))

        summary = (
            df_shape
            .merge(traj_counts, on="cell_label", how="left")
            .merge(point_counts, on="cell_label", how="left")
            .fillna({"n_trajectories": 0, "n_points": 0})
        )
        summary["n_trajectories"] = summary["n_trajectories"].astype(int)
        summary["n_points"]       = summary["n_points"].astype(int)

        if MIN_TRAJ_POINTS_PER_CELL > 0:
            summary = summary[summary["n_points"] >= MIN_TRAJ_POINTS_PER_CELL]
        summary = summary.sort_values("cell_label").reset_index(drop=True)

        out_csv = output_dir / "cell_trajectory_summary.csv"
        summary.to_csv(out_csv, index=False)
        print(f"    Saved {out_csv.name}")

        # ---- Save pickle ----
        classified_data = {
            "trajectories_df": df,
            "traj_cell_map":   traj_cell,
            "cell_summary":    summary,
            "mask":            mask_img,
            "membrane":        membrane,
        }
        out_pkl = output_dir / "cell_trajectories.pkl"
        with open(out_pkl, "wb") as fh:
            pickle.dump(classified_data, fh)
        print(f"    Saved {out_pkl.name}")

        # ---- Classification figure ----
        print("  Generating figures …")
        cell_labels = np.arange(1, n_cells + 1)
        colours     = plt.cm.tab20(np.linspace(0, 1, 20))
        cell_colours = {lbl: colours[i % 20] for i, lbl in enumerate(cell_labels)}
        cell_colours[0] = (0, 0, 0, 0)

        label_rgb = np.zeros((*mask_img.shape, 4), dtype=float)
        for lbl, col in cell_colours.items():
            if lbl > 0:
                label_rgb[mask_img == lbl] = col

        fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=150)

        # Left: label overview
        ax = axes[0]
        ax.imshow(membrane, cmap="gray", interpolation="nearest")
        ax.imshow(label_rgb, alpha=0.45, interpolation="nearest")
        for _, row in summary.iterrows():
            if row["n_trajectories"] > 0:
                ax.text(row["centroid_x"], row["centroid_y"],
                        str(int(row["cell_label"])),
                        fontsize=4, ha="center", va="center",
                        color="white", fontweight="bold")
        ax.set_title(f"Cellpose labels — {n_cells} cells  |  {W}×{H} px\n"
                     f"[{embryo_name}]", fontsize=11)
        ax.axis("off")

        # Right: trajectories coloured by cell
        ax = axes[1]
        ax.imshow(membrane, cmap="gray", interpolation="nearest")
        ax.imshow(label_rgb, alpha=0.20, interpolation="nearest")
        df_plot = df.sample(min(MAX_TRAJ_POINTS_PLOT, len(df)),
                            random_state=42) if len(df) > MAX_TRAJ_POINTS_PLOT else df
        for lbl in df_plot["cell_label"].unique():
            sub = df_plot[df_plot["cell_label"] == lbl]
            col = cell_colours.get(lbl, (0.5, 0.5, 0.5, 1))
            ax.scatter(sub["x"], sub["y"], s=0.3, color=col[:3],
                       alpha=0.5, linewidths=0)
        for _, row in summary[summary["n_trajectories"] > 0].iterrows():
            ax.text(row["centroid_x"], row["centroid_y"],
                    str(int(row["n_trajectories"])),
                    fontsize=4, ha="center", va="center",
                    color="yellow", fontweight="bold")
        ax.set_title(
            f"GEM trajectories by cell\n"
            f"{n_assigned:,} trajectories in "
            f"{(summary['n_trajectories'] > 0).sum()} cells  [{embryo_name}]",
            fontsize=11,
        )
        ax.axis("off")

        plt.tight_layout()
        out_fig = output_dir / "cell_trajectory_classification.png"
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_fig.name}")

        # ---- Distribution overview figure ----
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
        axes2[0].hist(summary["n_trajectories"], bins=30,
                      color="steelblue", edgecolor="white")
        axes2[0].set(xlabel="Trajectories per cell", ylabel="# cells",
                     title="Trajectory count")
        axes2[1].hist(summary["area_px"], bins=30,
                      color="darkorange", edgecolor="white")
        axes2[1].set(xlabel="Cell area (px²)", ylabel="# cells", title="Cell area")
        axes2[2].hist(summary["circularity"].dropna(), bins=30,
                      color="mediumseagreen", edgecolor="white")
        axes2[2].set(xlabel="Circularity", ylabel="# cells", title="Circularity")
        plt.suptitle(f"Per-cell overview — {embryo_name}", fontsize=13,
                     fontweight="bold")
        plt.tight_layout()
        out_fig2 = output_dir / "cell_overview_distributions.png"
        fig2.savefig(out_fig2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"    Saved {out_fig2.name}")

        print(f"  [{embryo_name}] DONE  "
              f"({n_assigned} trajectories → {(summary['n_trajectories']>0).sum()} cells)")
        return True

    except Exception as exc:
        print(f"\n  [ERROR] {embryo_name}: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main — prompt user, discover files, run batch
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1 — Cell Trajectory Classifier  (batch folder mode)")
print("=" * 60)
print()
print("Enter the path to the folder containing your embryo files.")
print("(Tip: you can drag-and-drop the folder into this terminal.)")
print()

# ---- Prompt for input folder ----
while True:
    raw = input("  Input folder path: ").strip().strip("'\"")
    input_folder = Path(raw).expanduser().resolve()
    if input_folder.is_dir():
        break
    print(f"  Not found or not a directory: '{raw}'")
    print("  Please try again.\n")

# ---- Prompt for output folder (optional) ----
print()
print("  Output folder (subfolders will be created here for each embryo).")
print(f"  Press Enter to use the same folder: {input_folder}")
raw_out = input("  Output folder path [Enter = same]: ").strip().strip("'\"")
output_base = Path(raw_out).expanduser().resolve() if raw_out else input_folder
output_base.mkdir(parents=True, exist_ok=True)

print()
print(f"  Scanning: {input_folder}")
embryos = find_embryo_sets(input_folder)

if not embryos:
    print("\n  No Cellpose mask files found in that folder.")
    print("  Expected files named like:  *_cp_masks.png  or  *_cp_masks.tif")
    sys.exit(1)

summarise_found(embryos)

# ---- Check for missing required files ----
missing_required = [em for em in embryos
                    if em["traj_csv"] is None or em["membrane"] is None]
if missing_required:
    print("  WARNING: some embryos are missing required files:")
    for em in missing_required:
        if em["traj_csv"] is None:
            print(f"    [{em['embryo_name']}] no trajectory CSV found")
        if em["membrane"] is None:
            print(f"    [{em['embryo_name']}] no membrane TIF found "
                  "(will use mask for display)")
    print()

# ---- Confirm before processing ----
resp = input("  Proceed with all embryos above? [Y/n]: ").strip().lower()
if resp not in ("", "y", "yes"):
    print("  Aborted.")
    sys.exit(0)

# ---- Process each embryo ----
results = []
for em in embryos:
    if em["traj_csv"] is None:
        print(f"\n  Skipping {em['embryo_name']} — no trajectory CSV.")
        results.append((em["embryo_name"], "SKIPPED"))
        continue

    out_dir = output_base / em["embryo_name"]
    ok = process_embryo(
        embryo_name   = em["embryo_name"],
        traj_csv      = em["traj_csv"],
        mask_path     = em["mask"],
        membrane_path = em["membrane"],
        output_dir    = out_dir,
    )
    results.append((em["embryo_name"], "OK" if ok else "ERROR"))

# ---- Final summary ----
print()
print("=" * 60)
print("Step 1 — Batch complete")
print("=" * 60)
for name, status in results:
    print(f"  {name:<30}  {status}")

print()
print(f"  Output written to: {output_base}")
print()
print("  Next step: for each embryo subfolder, run:")
print("    python GEM_mitosis/2_diffusion_per_cell.py")
print("  (or use the batch pipeline: python GEM_mitosis/5_batch_pipeline.py)")

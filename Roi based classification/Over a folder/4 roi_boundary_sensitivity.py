#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 roi_boundary_sensitivity.py

Tests how sensitive enrichment and diffusion results are to the exact placement
of ROI boundaries by systematically eroding and dilating each ROI.

The key question: if you had drawn the ROI boundary a few pixels inward or
outward, would your conclusions change?

For each dilation level (negative = erosion, positive = dilation):
  - The ROI polygon is rasterized to a binary mask
  - The mask is eroded or dilated by that number of pixels
  - All trajectories are reclassified inside/outside the modified ROI
  - Enrichment ratio and median D are recomputed

A result is considered robust if both metrics remain stable across a range of
dilation levels (e.g., ±10 pixels).

Input files matched by sample ID (e.g. 'Em1.nd2'):
  {sample_id}_membrane_.tif           membrane image (for embryo mask)
  {sample_id}_membrane_.tif_rois.zip  ImageJ ROI zip
  roi_trajectory_data_{sample_id}.pkl ROI-assigned pkl (from script 1)

Output per sample → {sample_id}_boundary_sensitivity/:
  sensitivity_curves.png              enrichment + D vs. dilation level
  sensitivity_results.csv             full numerical results

Cross-sample summary → boundary_sensitivity_summary_{timestamp}/:
  cross_sample_sensitivity.png
  cross_sample_summary.csv

Usage:
  python "4 roi_boundary_sensitivity.py"

Notes:
  - ROIs that become empty after erosion are excluded at that dilation level.
  - The dilation is applied to the rasterized mask, not the polygon vertices.
  - Pixel-to-micron conversion: 0.094 µm/pixel (from coordinate_transform in pkl).
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import read_roi
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_erosion, binary_dilation, disk
from skimage.draw import polygon as draw_polygon

try:
    from scipy.stats import binomtest
    def _binomtest(k, n, p):
        if n == 0 or p <= 0 or p >= 1:
            return 1.0
        result = binomtest(k, n, p, alternative='two-sided')
        return result.pvalue
except ImportError:
    from scipy.stats import binom_test
    def _binomtest(k, n, p):
        if n == 0 or p <= 0 or p >= 1:
            return 1.0
        return binom_test(k, n, p, alternative='two-sided')

# ============================================================
# Parameters
# ============================================================
DILATION_STEPS       = [-20, -15, -10, -5, 0, 5, 10, 15, 20]  # pixels
MIN_ROI_AREA_PIXELS  = 50
MIN_TRAJECTORIES     = 5
CLOSING_DISK_RADIUS  = 5
DEFAULT_PX_TO_UM     = 0.094


# ─────────────────────────────────────────────────────────────
# File discovery  (same convention as script 5)
# ─────────────────────────────────────────────────────────────
def find_sample_files(folder):
    membrane_files = sorted(glob.glob(os.path.join(folder, '*_membrane_.tif')))
    if not membrane_files:
        print(f"No *_membrane_.tif files found in {folder}")
        return []
    samples = []
    for mem_tif in membrane_files:
        basename  = os.path.basename(mem_tif)
        sample_id = basename.replace('_membrane_.tif', '')
        roi_zip   = mem_tif + '_rois.zip'
        pkl_path  = os.path.join(folder, f'roi_trajectory_data_{sample_id}.pkl')
        missing = []
        if not os.path.exists(roi_zip):
            missing.append(os.path.basename(roi_zip))
        if not os.path.exists(pkl_path):
            missing.append(os.path.basename(pkl_path))
        if missing:
            print(f"[{sample_id}] Skipping — missing: {', '.join(missing)}")
            continue
        samples.append((sample_id, mem_tif, roi_zip, pkl_path))
        print(f"[{sample_id}] All files found")
    return samples


# ─────────────────────────────────────────────────────────────
# Image / mask helpers
# ─────────────────────────────────────────────────────────────
def load_membrane_image(tif_path):
    img = tifffile.imread(tif_path).astype(np.float32)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[0] if img.shape[0] <= img.shape[-1] else img.max(axis=-1)
    if img.ndim == 4:
        return img[0].max(axis=0)
    raise ValueError(f"Unexpected image dimensions: {img.shape}")


def compute_embryo_mask(image):
    threshold = threshold_otsu(image)
    return binary_closing(image > threshold, disk(CLOSING_DISK_RADIUS))


def rasterize_roi(roi_x, roi_y, image_shape):
    """Rasterize an ImageJ polygon ROI to a boolean mask."""
    mask = np.zeros(image_shape, dtype=bool)
    rr, cc = draw_polygon(roi_y, roi_x, image_shape)
    mask[rr, cc] = True
    return mask


def apply_dilation(mask, delta_pixels):
    """Apply erosion (delta<0) or dilation (delta>0) to a binary mask."""
    if delta_pixels == 0:
        return mask.copy()
    selem = disk(abs(delta_pixels))
    if delta_pixels > 0:
        return binary_dilation(mask, selem)
    else:
        return binary_erosion(mask, selem)


# ─────────────────────────────────────────────────────────────
# ROI loading
# ─────────────────────────────────────────────────────────────
def load_roi_polygons(roi_zip_path):
    rois_raw = read_roi.read_roi_zip(roi_zip_path)
    rois = {}
    for roi_id, roi in rois_raw.items():
        if 'x' not in roi or 'y' not in roi:
            continue
        rois[roi_id] = {
            'x': np.array(roi['x'], dtype=float),
            'y': np.array(roi['y'], dtype=float),
        }
    return rois


# ─────────────────────────────────────────────────────────────
# Trajectory extraction
# ─────────────────────────────────────────────────────────────
def get_all_trajectories(pkl_data):
    """
    Extract all trajectories (from all ROI groups including unassigned)
    as a list of dicts with pixel positions and D values.
    Only trajectories with valid D are included.
    """
    ct     = pkl_data.get('coordinate_transform', {})
    scale  = ct.get('scale_factor', 1.0 / DEFAULT_PX_TO_UM)
    x_off  = ct.get('x_offset', 0.0)
    y_off  = ct.get('y_offset', 0.0)

    trajs = []
    for roi_id, roi_trajs in pkl_data.get('roi_trajectories', {}).items():
        for traj in roi_trajs:
            D = traj.get('D', np.nan)
            if np.isnan(D):
                continue
            if 'x' not in traj or 'y' not in traj:
                continue
            x_px = float(np.mean(traj['x'])) * scale + x_off
            y_px = float(np.mean(traj['y'])) * scale + y_off
            trajs.append({'D': D, 'x_px': x_px, 'y_px': y_px})
    return trajs


def classify_by_mask(trajs, roi_mask, embryo_mask):
    """
    Classify trajectories as inside or outside the ROI mask.
    Only trajectories within the embryo mask are considered.
    Returns (inside_D, outside_D) as numpy arrays.
    """
    h, w = roi_mask.shape
    inside_D  = []
    outside_D = []
    for t in trajs:
        xi = int(round(t['x_px']))
        yi = int(round(t['y_px']))
        if not (0 <= yi < h and 0 <= xi < w):
            continue
        if not embryo_mask[yi, xi]:
            continue
        if roi_mask[yi, xi]:
            inside_D.append(t['D'])
        else:
            outside_D.append(t['D'])
    return np.array(inside_D), np.array(outside_D)


# ─────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────
def compute_metrics(inside_D, outside_D, roi_area, embryo_area):
    """
    Returns a dict with enrichment ratio, D metrics, and p-values.
    """
    n_inside  = len(inside_D)
    n_outside = len(outside_D)
    n_total   = n_inside + n_outside

    expected_frac = roi_area / embryo_area if embryo_area > 0 else np.nan

    if n_total >= MIN_TRAJECTORIES and expected_frac > 0:
        observed_frac    = n_inside / n_total
        enrichment_ratio = observed_frac / expected_frac
        p_enrichment     = _binomtest(n_inside, n_total, expected_frac)
    else:
        observed_frac = enrichment_ratio = p_enrichment = np.nan

    median_D_in  = float(np.median(inside_D))  if len(inside_D)  >= 3 else np.nan
    median_D_out = float(np.median(outside_D)) if len(outside_D) >= 3 else np.nan
    D_diff       = median_D_in - median_D_out   if not (np.isnan(median_D_in) or np.isnan(median_D_out)) else np.nan

    return {
        'n_inside':        n_inside,
        'n_outside':       n_outside,
        'n_total':         n_total,
        'roi_area':        roi_area,
        'embryo_area':     embryo_area,
        'expected_frac':   expected_frac,
        'observed_frac':   observed_frac,
        'enrichment_ratio':enrichment_ratio,
        'p_enrichment':    p_enrichment,
        'median_D_in':     median_D_in,
        'median_D_out':    median_D_out,
        'D_diff':          D_diff,
    }


# ─────────────────────────────────────────────────────────────
# Plotting — per sample
# ─────────────────────────────────────────────────────────────
def plot_sensitivity_curves(roi_sensitivity, sample_id, output_path):
    """
    For each ROI, plot enrichment ratio and D metrics vs. dilation level.
    """
    n_rois = len(roi_sensitivity)
    if n_rois == 0:
        return

    fig, axes = plt.subplots(2, n_rois,
                              figsize=(max(6, n_rois * 4), 8),
                              squeeze=False)

    for col, (roi_id, records) in enumerate(roi_sensitivity.items()):
        deltas = [r['delta'] for r in records]
        enrich = [r['enrichment_ratio'] for r in records]
        D_diff = [r['D_diff'] for r in records]
        n_in   = [r['n_inside'] for r in records]

        short_id = roi_id.split('-')[0][:12]

        # ── Top: enrichment ratio ──────────────────────────────
        ax = axes[0][col]
        valid = [(d, e) for d, e in zip(deltas, enrich) if not np.isnan(e)]
        if valid:
            ds, es = zip(*valid)
            ax.plot(ds, es, 'o-', color='steelblue', linewidth=2, markersize=6)
            # Highlight the original ROI (delta=0) in orange
            if 0 in ds:
                idx0 = ds.index(0)
                ax.scatter([ds[idx0]], [es[idx0]], color='darkorange',
                           s=80, zorder=5, label='Original ROI')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Dilation (pixels)\nnegative = erosion', fontsize=9)
        ax.set_ylabel('Enrichment ratio', fontsize=9)
        ax.set_title(f'{short_id}\nEnrichment vs. boundary shift', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate n_inside at each point
        for d, e, n in zip(deltas, enrich, n_in):
            if not np.isnan(e):
                ax.annotate(f'n={n}', (d, e), textcoords='offset points',
                            xytext=(0, 6), ha='center', fontsize=6, color='gray')

        # ── Bottom: D difference ───────────────────────────────
        ax2 = axes[1][col]
        valid2 = [(d, dd) for d, dd in zip(deltas, D_diff) if not np.isnan(dd)]
        if valid2:
            ds2, dds = zip(*valid2)
            ax2.plot(ds2, dds, 'o-', color='firebrick', linewidth=2, markersize=6)
            if 0 in ds2:
                idx0 = ds2.index(0)
                ax2.scatter([ds2[idx0]], [dds[idx0]], color='darkorange',
                            s=80, zorder=5, label='Original ROI')
        ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('Dilation (pixels)\nnegative = erosion', fontsize=9)
        ax2.set_ylabel('Median D_in − D_out (µm²/s)', fontsize=9)
        ax2.set_title(f'{short_id}\nD difference vs. boundary shift', fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    plt.suptitle(f'ROI Boundary Sensitivity — {sample_id}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def save_sensitivity_csv(roi_sensitivity, sample_id, output_path):
    rows = []
    for roi_id, records in roi_sensitivity.items():
        for r in records:
            rows.append({
                'sample_id':        sample_id,
                'roi_id':           roi_id,
                'delta_pixels':     r['delta'],
                'roi_area_pixels2': round(r['roi_area'], 1),
                'n_inside':         r['n_inside'],
                'n_outside':        r['n_outside'],
                'n_total':          r['n_total'],
                'enrichment_ratio': round(r['enrichment_ratio'], 4) if not np.isnan(r['enrichment_ratio']) else np.nan,
                'p_enrichment':     r['p_enrichment'],
                'median_D_in':      round(r['median_D_in'],  4) if not np.isnan(r['median_D_in'])  else np.nan,
                'median_D_out':     round(r['median_D_out'], 4) if not np.isnan(r['median_D_out']) else np.nan,
                'D_diff':           round(r['D_diff'], 4)       if not np.isnan(r['D_diff'])       else np.nan,
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Plotting — cross-sample
# ─────────────────────────────────────────────────────────────
def plot_cross_sample(all_sample_results, output_dir):
    """
    Overlay the δ=0 sensitivity curve from all samples on one figure.
    Each curve shows enrichment ratio vs dilation for pooled-ROI level.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_sample_results), 1)))

    for i, (sample_id, sample_data) in enumerate(all_sample_results.items()):
        # Average across all ROIs at each dilation level
        all_records_by_delta = {}
        for roi_id, records in sample_data.items():
            for r in records:
                d = r['delta']
                if d not in all_records_by_delta:
                    all_records_by_delta[d] = []
                all_records_by_delta[d].append(r)

        deltas  = sorted(all_records_by_delta.keys())
        enrich_mean = []
        D_diff_mean = []
        for d in deltas:
            recs = all_records_by_delta[d]
            ens  = [r['enrichment_ratio'] for r in recs if not np.isnan(r['enrichment_ratio'])]
            dds  = [r['D_diff']           for r in recs if not np.isnan(r['D_diff'])]
            enrich_mean.append(np.mean(ens)  if ens else np.nan)
            D_diff_mean.append(np.mean(dds)  if dds else np.nan)

        ax1.plot(deltas, enrich_mean, 'o-', color=colors[i], linewidth=1.5,
                 markersize=5, label=sample_id[:20])
        ax2.plot(deltas, D_diff_mean, 'o-', color=colors[i], linewidth=1.5,
                 markersize=5, label=sample_id[:20])

    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax1.axvline(0,   color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Dilation (pixels)');  ax1.set_ylabel('Mean enrichment ratio')
    ax1.set_title('Enrichment ratio vs. boundary shift\n(average across ROIs per sample)')
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(0,   color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Dilation (pixels)');  ax2.set_ylabel('Mean ΔD_median (µm²/s)')
    ax2.set_title('D difference vs. boundary shift\n(average across ROIs per sample)')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    plt.suptitle('Cross-sample ROI boundary sensitivity', fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cross_sample_sensitivity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out)}")


def save_cross_sample_csv(all_sample_results, output_path):
    rows = []
    for sample_id, sample_data in all_sample_results.items():
        for roi_id, records in sample_data.items():
            for r in records:
                rows.append({
                    'sample_id':        sample_id,
                    'roi_id':           roi_id,
                    'delta_pixels':     r['delta'],
                    'enrichment_ratio': round(r['enrichment_ratio'], 4) if not np.isnan(r['enrichment_ratio']) else np.nan,
                    'D_diff':           round(r['D_diff'], 4)           if not np.isnan(r['D_diff'])           else np.nan,
                    'n_inside':         r['n_inside'],
                    'n_total':          r['n_total'],
                })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Per-sample pipeline
# ─────────────────────────────────────────────────────────────
def process_sample(sample_id, membrane_tif, roi_zip, pkl_path, folder):
    print(f"\n{'='*60}")
    print(f"  {sample_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(folder, f'{sample_id}_boundary_sensitivity')
    os.makedirs(output_dir, exist_ok=True)

    # Embryo mask
    print("  Computing embryo mask...")
    try:
        image = tifffile.imread(membrane_tif).astype(np.float32)
        if image.ndim == 3:
            image = image[0] if image.shape[0] <= image.shape[-1] else image.max(axis=-1)
        elif image.ndim == 4:
            image = image[0].max(axis=0)
    except Exception as e:
        print(f"  ERROR loading image: {e}"); return None
    embryo_mask = binary_closing(image > threshold_otsu(image), disk(CLOSING_DISK_RADIUS))
    embryo_area = int(embryo_mask.sum())
    print(f"  Embryo area: {embryo_area:,} px²")

    # ROI polygons
    print("  Loading ROI polygons...")
    try:
        rois = load_roi_polygons(roi_zip)
    except Exception as e:
        print(f"  ERROR loading ROI zip: {e}"); return None
    if not rois:
        print("  No valid ROIs found — skipping."); return None

    # Trajectory data
    print("  Loading trajectory pkl...")
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
    except Exception as e:
        print(f"  ERROR loading pkl: {e}"); return None

    all_trajs = get_all_trajectories(pkl_data)
    if len(all_trajs) < MIN_TRAJECTORIES:
        print(f"  Only {len(all_trajs)} trajectories with valid D — skipping."); return None
    print(f"  {len(all_trajs)} trajectories with valid D")

    # ── Main loop: per ROI, per dilation level ──────────────
    roi_sensitivity = {}
    for roi_id, roi in rois.items():
        print(f"\n  ROI: {roi_id}")
        base_mask = rasterize_roi(roi['x'], roi['y'], image.shape)
        records   = []
        for delta in DILATION_STEPS:
            mod_mask  = apply_dilation(base_mask, delta)
            # Intersect with embryo (ROI can't extend outside embryo)
            mod_mask  = mod_mask & embryo_mask
            roi_area  = int(mod_mask.sum())
            if roi_area < MIN_ROI_AREA_PIXELS:
                print(f"    delta={delta:+d}: ROI too small ({roi_area} px²) — skipped")
                records.append({'delta': delta, 'roi_area': roi_area,
                                'n_inside': 0, 'n_outside': 0, 'n_total': 0,
                                'enrichment_ratio': np.nan, 'p_enrichment': np.nan,
                                'median_D_in': np.nan, 'median_D_out': np.nan,
                                'D_diff': np.nan})
                continue
            inside_D, outside_D = classify_by_mask(all_trajs, mod_mask, embryo_mask)
            metrics = compute_metrics(inside_D, outside_D, roi_area, embryo_area)
            metrics['delta'] = delta
            records.append(metrics)
            print(f"    delta={delta:+d}: n_in={metrics['n_inside']:3d}  "
                  f"enrich={metrics['enrichment_ratio']:.3f}  "
                  f"ΔD={metrics['D_diff']:.4f} µm²/s")
        roi_sensitivity[roi_id] = records

    # Plots + CSV
    print("\n  Generating plots...")
    plot_sensitivity_curves(roi_sensitivity, sample_id,
                            os.path.join(output_dir, 'sensitivity_curves.png'))
    save_sensitivity_csv(roi_sensitivity, sample_id,
                         os.path.join(output_dir, 'sensitivity_results.csv'))
    print(f"\n  Output: {output_dir}")
    return roi_sensitivity


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("ROI Boundary Sensitivity Analyzer")
    print("==================================\n")

    folder = input("Enter path to data folder: ").strip()
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}"); sys.exit(1)

    samples = find_sample_files(folder)
    if not samples:
        print("No complete sample sets found. Exiting."); sys.exit(1)

    print(f"\n{len(samples)} sample(s) to process.\n")
    all_results = {}
    for sample_id, membrane_tif, roi_zip, pkl_path in samples:
        result = process_sample(sample_id, membrane_tif, roi_zip, pkl_path, folder)
        if result is not None:
            all_results[sample_id] = result

    if not all_results:
        print("\nNo samples processed successfully."); sys.exit(1)

    # Cross-sample summary
    print(f"\n{'='*60}")
    print("Cross-sample summary")
    print(f"{'='*60}")
    summary_dir = os.path.join(
        folder,
        f'boundary_sensitivity_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(summary_dir, exist_ok=True)
    plot_cross_sample(all_results, summary_dir)
    save_cross_sample_csv(all_results, os.path.join(summary_dir, 'cross_sample_summary.csv'))
    print(f"\nSummary saved to: {summary_dir}")
    print("\nDone.")


if __name__ == '__main__':
    main()

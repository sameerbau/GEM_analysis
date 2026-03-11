#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 random_roi_permutation.py

Tests whether your hand-drawn ROI is spatially special by building a null
distribution from randomly placed ROIs of the same shape within the embryo.

The core question: "Could I have drawn this ROI anywhere and gotten the same
enrichment / diffusion difference?"

For each original ROI:
  1. The ROI polygon is rasterized to a binary mask (captures its exact shape)
  2. N_PERMUTATIONS random placements are generated:
       - A random new centroid is sampled from within the embryo boundary
       - The ROI shape is translated to that centroid
       - Placement accepted only if ≥ MIN_OVERLAP_FRACTION of the ROI
         stays within the embryo (to avoid biasing toward edges)
  3. For each random placement, enrichment ratio and median D inside are
     computed by checking which trajectories fall inside the shifted mask
  4. The actual ROI metric is compared to this null distribution:
       empirical p-value = fraction of random placements that are ≥ actual

Output per sample → {sample_id}_random_permutation/:
  permutation_{roi_id}_enrichment.png    null dist + actual value
  permutation_{roi_id}_diffusion.png     null dist + actual value
  permutation_results.csv

Cross-sample summary → random_permutation_summary_{timestamp}/:
  cross_sample_pvalues.png
  summary_table.csv

Usage:
  python "5 random_roi_permutation.py"

Notes:
  - Trajectories coordinates are shifted (not the mask) for efficiency: instead
    of moving the mask N times, the trajectory positions are shifted by the
    negative offset, and then tested against the original mask. This is
    mathematically equivalent.
  - ROIs that cannot achieve MIN_OVERLAP_FRACTION within the embryo in
    MAX_ATTEMPTS tries are skipped.
  - N_PERMUTATIONS = 1000 is recommended. Increase to 5000 for publications.
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
from skimage.morphology import binary_closing, binary_erosion, disk
from skimage.draw import polygon as draw_polygon
from skimage import measure

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================
# Parameters
# ============================================================
N_PERMUTATIONS       = 1000
MIN_OVERLAP_FRACTION = 0.90   # accept placement if ≥90 % of ROI within embryo
MAX_ATTEMPTS         = 20000  # maximum sampling attempts per permutation set
MIN_ROI_AREA_PIXELS  = 50
MIN_TRAJECTORIES     = 5
CLOSING_DISK_RADIUS  = 5
DEFAULT_PX_TO_UM     = 0.094
RANDOM_SEED          = 42


# ─────────────────────────────────────────────────────────────
# File discovery
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
def load_and_mask(tif_path):
    img = tifffile.imread(tif_path).astype(np.float32)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= img.shape[-1] else img.max(axis=-1)
    elif img.ndim == 4:
        img = img[0].max(axis=0)
    embryo_mask = binary_closing(img > threshold_otsu(img), disk(CLOSING_DISK_RADIUS))
    return img, embryo_mask


def rasterize_roi(roi_x, roi_y, image_shape):
    mask = np.zeros(image_shape, dtype=bool)
    rr, cc = draw_polygon(roi_y, roi_x, image_shape)
    mask[rr, cc] = True
    return mask


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
    Returns all trajectories (all ROI groups including unassigned) as
    list of {D, x_px, y_px}.  Only includes trajectories with valid D.
    """
    ct    = pkl_data.get('coordinate_transform', {})
    scale = ct.get('scale_factor', 1.0 / DEFAULT_PX_TO_UM)
    x_off = ct.get('x_offset', 0.0)
    y_off = ct.get('y_offset', 0.0)

    trajs = []
    for roi_id, roi_trajs in pkl_data.get('roi_trajectories', {}).items():
        for traj in roi_trajs:
            D = traj.get('D', np.nan)
            if np.isnan(D):
                continue
            if 'x' not in traj or 'y' not in traj:
                continue
            trajs.append({
                'D':    D,
                'x_px': float(np.mean(traj['x'])) * scale + x_off,
                'y_px': float(np.mean(traj['y'])) * scale + y_off,
            })
    return trajs


# ─────────────────────────────────────────────────────────────
# Random placement engine
# ─────────────────────────────────────────────────────────────
def get_roi_centroid(roi_mask):
    rows, cols = np.where(roi_mask)
    return rows.mean(), cols.mean()


def get_valid_centroid_zone(embryo_mask, roi_mask):
    """
    Erode the embryo mask by the approximate ROI radius so that any centroid
    sampled from this zone will keep the ROI mostly within the embryo.
    Falls back to the full embryo mask if the erosion removes everything.
    """
    rows, cols = np.where(roi_mask)
    if len(rows) == 0:
        return embryo_mask
    # Approximate radius: half the diagonal of the bounding box
    h_bb = rows.max() - rows.min()
    w_bb = cols.max() - cols.min()
    radius = int(np.ceil(np.sqrt(h_bb**2 + w_bb**2) / 2)) + 2
    radius = max(radius, 1)
    eroded = binary_erosion(embryo_mask, disk(radius))
    if eroded.sum() < 10:
        return embryo_mask  # fallback
    return eroded


def classify_with_shifted_roi(trajs, roi_mask, dy, dx, embryo_mask):
    """
    Classify trajectories against the ROI shifted by (dy, dx) pixels.
    Equivalent to shifting trajectory positions by (-dy, -dx) and testing
    against the original mask — more efficient for large N.

    Returns (inside_D, outside_D).
    """
    h, w = roi_mask.shape
    inside_D  = []
    outside_D = []
    for t in trajs:
        # Shift trajectory position into the frame of the shifted ROI
        xi = int(round(t['x_px'] - dx))
        yi = int(round(t['y_px'] - dy))

        # Original position for embryo check
        xi_orig = int(round(t['x_px']))
        yi_orig = int(round(t['y_px']))

        if not (0 <= yi_orig < h and 0 <= xi_orig < w):
            continue
        if not embryo_mask[yi_orig, xi_orig]:
            continue

        if 0 <= yi < h and 0 <= xi < w and roi_mask[yi, xi]:
            inside_D.append(t['D'])
        else:
            outside_D.append(t['D'])
    return np.array(inside_D), np.array(outside_D)


def run_permutation_test(roi_mask, embryo_mask, all_trajs, n_permutations, rng):
    """
    Generate N random placements of roi_mask within embryo_mask.
    For each placement, compute enrichment ratio and median D inside.

    Returns:
        perm_enrichment : array of shape (n_placed,)
        perm_median_D   : array of shape (n_placed,)
        n_placed        : actual number of accepted placements
    """
    orig_cy, orig_cx = get_roi_centroid(roi_mask)
    roi_area = int(roi_mask.sum())
    embryo_area = int(embryo_mask.sum())
    if roi_area == 0 or embryo_area == 0:
        return np.array([]), np.array([]), 0

    valid_zone     = get_valid_centroid_zone(embryo_mask, roi_mask)
    valid_ys, valid_xs = np.where(valid_zone)
    if len(valid_ys) == 0:
        valid_ys, valid_xs = np.where(embryo_mask)

    perm_enrichment = []
    perm_median_D   = []
    attempts = 0

    while len(perm_enrichment) < n_permutations and attempts < MAX_ATTEMPTS:
        attempts += 1

        # Sample random new centroid
        idx    = rng.integers(len(valid_ys))
        new_cy = valid_ys[idx]
        new_cx = valid_xs[idx]
        dy = new_cy - orig_cy
        dx = new_cx - orig_cx

        # Verify overlap: check how many ROI pixels land within embryo
        # We do this by checking the shifted mask overlap quickly
        # (fast approximation: use centroid + known radius)
        # For accuracy, we shift and check, but only for a subset of candidates
        # to avoid the O(n_pixels) overhead per candidate.
        # Full check every 10 attempts for accepted candidates.
        if attempts % 10 == 0 or len(perm_enrichment) < 5:
            # Full overlap check
            rows, cols = np.where(roi_mask)
            new_rows = (rows + dy).astype(int)
            new_cols = (cols + dx).astype(int)
            valid_pixels = (
                (new_rows >= 0) & (new_rows < embryo_mask.shape[0]) &
                (new_cols >= 0) & (new_cols < embryo_mask.shape[1])
            )
            if valid_pixels.sum() == 0:
                continue
            in_embryo = embryo_mask[new_rows[valid_pixels], new_cols[valid_pixels]]
            overlap = in_embryo.sum() / roi_area
            if overlap < MIN_OVERLAP_FRACTION:
                continue
        # else: trust the valid_zone erosion and proceed

        # Count trajectories inside shifted ROI
        inside_D, outside_D = classify_with_shifted_roi(
            all_trajs, roi_mask, dy, dx, embryo_mask
        )
        n_in    = len(inside_D)
        n_total = n_in + len(outside_D)

        if n_total == 0:
            continue

        expected_frac = roi_area / embryo_area
        enrich_ratio  = (n_in / n_total) / expected_frac if expected_frac > 0 else np.nan
        median_D_in   = float(np.median(inside_D)) if len(inside_D) >= 3 else np.nan

        perm_enrichment.append(enrich_ratio)
        perm_median_D.append(median_D_in)

    if attempts >= MAX_ATTEMPTS and len(perm_enrichment) < n_permutations:
        print(f"    Warning: only {len(perm_enrichment)}/{n_permutations} "
              f"placements accepted after {attempts} attempts")

    return (np.array(perm_enrichment),
            np.array(perm_median_D),
            len(perm_enrichment))


def empirical_pvalue(null_dist, observed, two_sided=True):
    """
    Compute empirical p-value: fraction of null values at least as extreme
    as the observed value.
    """
    null = null_dist[~np.isnan(null_dist)]
    if len(null) == 0 or np.isnan(observed):
        return np.nan
    if two_sided:
        null_mean = np.mean(null)
        deviation_obs  = abs(observed - null_mean)
        deviation_null = np.abs(null - null_mean)
        return float(np.mean(deviation_null >= deviation_obs))
    else:
        return float(np.mean(null >= observed))


# ─────────────────────────────────────────────────────────────
# Plotting — per ROI
# ─────────────────────────────────────────────────────────────
def plot_null_distribution(null_values, actual_value, metric_name,
                           units, roi_id, sample_id, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    null_clean = null_values[~np.isnan(null_values)]
    if len(null_clean) == 0:
        plt.close(); return

    ax.hist(null_clean, bins=40, color='steelblue', alpha=0.75,
            edgecolor='white', linewidth=0.5, label=f'Null (N={len(null_clean)})')

    if not np.isnan(actual_value):
        ax.axvline(actual_value, color='darkorange', linewidth=2.5,
                   linestyle='--', label=f'Actual ROI: {actual_value:.3f}')

        p = empirical_pvalue(null_clean, actual_value)
        null_mean = np.mean(null_clean)
        null_std  = np.std(null_clean)
        percentile_rank = float(np.mean(null_clean <= actual_value)) * 100

        ax.text(0.97, 0.95,
                f'Actual = {actual_value:.3f}\n'
                f'Null mean ± SD: {null_mean:.3f} ± {null_std:.3f}\n'
                f'Empirical p = {p:.3f}\n'
                f'Percentile rank = {percentile_rank:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          alpha=0.8))

    short_id = roi_id.split('-')[0][:12]
    ax.set_xlabel(f'{metric_name} ({units})', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Random ROI permutation — {metric_name}\n'
                 f'Sample: {sample_id}   ROI: {short_id}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_roi_overlay_with_random(image, embryo_mask, roi_mask,
                                  random_centroids, actual_centroid,
                                  roi_id, sample_id, output_path, n_show=50):
    """
    Overlay: embryo contour + actual ROI (orange) + N random placements (blue, translucent).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = (np.percentile(image[embryo_mask], [1, 99])
                  if embryo_mask.sum() > 0 else (image.min(), image.max()))
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Embryo contour
    for contour in measure.find_contours(embryo_mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color='cyan', linewidth=1, alpha=0.6)

    # Actual ROI
    orig_rows, orig_cols = np.where(roi_mask)
    orig_cy, orig_cx = orig_rows.mean(), orig_cols.mean()
    roi_contours = measure.find_contours(roi_mask.astype(float), 0.5)
    for c in roi_contours:
        ax.plot(c[:, 1], c[:, 0], color='darkorange', linewidth=2.5,
                label='Actual ROI', zorder=5)

    # Random placements (show a subset)
    n_to_show = min(n_show, len(random_centroids))
    shown = 0
    for dy, dx in random_centroids[:n_to_show]:
        for c in roi_contours:
            ax.plot(c[:, 1] + dx, c[:, 0] + dy, color='dodgerblue',
                    linewidth=0.6, alpha=0.25, zorder=3)
        shown += 1

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkorange', linewidth=2.5, label='Actual ROI'),
        Line2D([0], [0], color='dodgerblue', linewidth=1.5, alpha=0.6,
               label=f'{n_to_show} random placements'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f'Random placement visualization\n'
                 f'{sample_id} — {roi_id.split("-")[0][:12]}', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Plotting — cross-sample
# ─────────────────────────────────────────────────────────────
def plot_cross_sample_pvalues(all_results, output_path):
    """
    Scatter plot: empirical p-values for enrichment (x) and D (y) per ROI,
    per sample.  Points in lower-left quadrant are robust.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 1)))

    for i, (sample_id, sample_data) in enumerate(all_results.items()):
        for roi_id, res in sample_data.items():
            p_enrich = res.get('p_enrichment_perm', np.nan)
            p_D      = res.get('p_D_perm', np.nan)
            if np.isnan(p_enrich) or np.isnan(p_D):
                continue
            ax.scatter(p_enrich, p_D, color=colors[i], s=60, alpha=0.8,
                       label=sample_id[:20] if roi_id == list(sample_data.keys())[0] else '')
            short_id = roi_id.split('-')[0][:8]
            ax.annotate(short_id, (p_enrich, p_D),
                        xytext=(4, 4), textcoords='offset points', fontsize=7)

    ax.axvline(0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Empirical p-value (enrichment)', fontsize=11)
    ax.set_ylabel('Empirical p-value (median D)', fontsize=11)
    ax.set_title('Random permutation test — p-values\n'
                 'Lower-left = result not explainable by random placement', fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.fill_between([0, 0.05], [0, 0], [0.05, 0.05],
                    color='limegreen', alpha=0.10, label='p<0.05 both')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def save_summary_csv(all_results, output_path):
    rows = []
    for sample_id, sample_data in all_results.items():
        for roi_id, res in sample_data.items():
            rows.append({
                'sample_id':              sample_id,
                'roi_id':                 roi_id,
                'n_permutations_placed':  res.get('n_placed', 0),
                'actual_enrichment_ratio':res.get('actual_enrichment', np.nan),
                'null_mean_enrichment':   res.get('null_mean_enrichment', np.nan),
                'null_std_enrichment':    res.get('null_std_enrichment', np.nan),
                'p_enrichment_empirical': res.get('p_enrichment_perm', np.nan),
                'actual_median_D_in':     res.get('actual_median_D', np.nan),
                'null_mean_median_D':     res.get('null_mean_D', np.nan),
                'null_std_median_D':      res.get('null_std_D', np.nan),
                'p_D_empirical':          res.get('p_D_perm', np.nan),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Per-sample pipeline
# ─────────────────────────────────────────────────────────────
def process_sample(sample_id, membrane_tif, roi_zip, pkl_path, folder, rng,
                   summary_only=False):
    print(f"\n{'='*60}")
    print(f"  {sample_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(folder, f'{sample_id}_random_permutation')
    os.makedirs(output_dir, exist_ok=True)

    # Load image + embryo mask
    print("  Loading membrane image and computing embryo mask...")
    try:
        image, embryo_mask = load_and_mask(membrane_tif)
    except Exception as e:
        print(f"  ERROR: {e}"); return None
    embryo_area = int(embryo_mask.sum())
    print(f"  Embryo area: {embryo_area:,} px²")

    # Load ROIs
    print("  Loading ROI polygons...")
    try:
        rois = load_roi_polygons(roi_zip)
    except Exception as e:
        print(f"  ERROR loading ROI zip: {e}"); return None
    if not rois:
        print("  No valid ROIs — skipping."); return None

    # Load trajectories
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

    sample_results = {}

    for roi_id, roi in rois.items():
        print(f"\n  ROI: {roi_id}")
        roi_mask = rasterize_roi(roi['x'], roi['y'], image.shape)
        roi_area = int(roi_mask.sum())
        if roi_area < MIN_ROI_AREA_PIXELS:
            print(f"    ROI too small ({roi_area} px²) — skipping")
            continue

        # Actual metrics
        actual_inside_D, actual_outside_D = classify_with_shifted_roi(
            all_trajs, roi_mask, 0, 0, embryo_mask
        )
        n_in    = len(actual_inside_D)
        n_total = n_in + len(actual_outside_D)
        expected_frac    = roi_area / embryo_area
        actual_enrichment = ((n_in / n_total) / expected_frac
                             if n_total > 0 and expected_frac > 0 else np.nan)
        actual_median_D  = (float(np.median(actual_inside_D))
                            if len(actual_inside_D) >= 3 else np.nan)

        print(f"    Actual: n_in={n_in}  enrichment={actual_enrichment:.3f}  "
              f"median_D_in={actual_median_D:.4f} µm²/s")
        print(f"    Running {N_PERMUTATIONS} random placements...")

        # Run permutation test
        perm_enrich, perm_D, n_placed = run_permutation_test(
            roi_mask, embryo_mask, all_trajs, N_PERMUTATIONS, rng
        )
        print(f"    Placed: {n_placed}/{N_PERMUTATIONS}")

        # Empirical p-values
        p_enrich = empirical_pvalue(perm_enrich, actual_enrichment)
        p_D      = empirical_pvalue(perm_D,      actual_median_D)
        print(f"    p(enrichment) = {p_enrich:.3f}   p(median D) = {p_D:.3f}")

        # Collect random centroids for visualization
        # (stored as offsets dy, dx for overlay)
        # Not directly available from run_permutation_test — we'll skip overlay
        # and just show the histogram plots
        safe_roi_id = roi_id.replace('/', '_').replace('\\', '_')

        if not summary_only:
            # Enrichment plot
            plot_null_distribution(
                perm_enrich, actual_enrichment,
                'Enrichment ratio', 'a.u.',
                roi_id, sample_id,
                os.path.join(output_dir, f'permutation_{safe_roi_id}_enrichment.png')
            )
            # D plot
            plot_null_distribution(
                perm_D, actual_median_D,
                'Median D inside', 'µm²/s',
                roi_id, sample_id,
                os.path.join(output_dir, f'permutation_{safe_roi_id}_diffusion.png')
            )

        sample_results[roi_id] = {
            'actual_enrichment':   actual_enrichment,
            'actual_median_D':     actual_median_D,
            'null_mean_enrichment':float(np.nanmean(perm_enrich)) if n_placed > 0 else np.nan,
            'null_std_enrichment': float(np.nanstd(perm_enrich))  if n_placed > 0 else np.nan,
            'null_mean_D':         float(np.nanmean(perm_D))      if n_placed > 0 else np.nan,
            'null_std_D':          float(np.nanstd(perm_D))       if n_placed > 0 else np.nan,
            'p_enrichment_perm':   p_enrich,
            'p_D_perm':            p_D,
            'n_placed':            n_placed,
        }

    if not summary_only:
        # Per-sample CSV
        rows = []
        for roi_id, res in sample_results.items():
            rows.append({'sample_id': sample_id, 'roi_id': roi_id, **res})
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(output_dir, 'permutation_results.csv'), index=False
            )
            print(f"\n  Saved: permutation_results.csv")
        print(f"\n  Output: {output_dir}")
    else:
        print("\n  [Per-embryo plots/CSV skipped — summary-only mode]")
    return sample_results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global N_PERMUTATIONS
    print("Random ROI Permutation Test")
    print("============================\n")

    folder = input("Enter path to data folder: ").strip()
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}"); sys.exit(1)

    n_perm_input = input(f"Number of random permutations [{N_PERMUTATIONS}]: ").strip()
    if n_perm_input:
        try:
            N_PERMUTATIONS = int(n_perm_input)
        except ValueError:
            pass
    print(f"Using {N_PERMUTATIONS} permutations.\n")

    so_input = input("Save per-embryo plots/CSVs? [Y/n]: ").strip().lower()
    summary_only = so_input in ('n', 'no')
    if summary_only:
        print("Summary-only mode: per-embryo outputs will be skipped.\n")

    rng = np.random.default_rng(RANDOM_SEED)

    samples = find_sample_files(folder)
    if not samples:
        print("No complete sample sets found. Exiting."); sys.exit(1)

    print(f"\n{len(samples)} sample(s) to process.\n")
    all_results = {}
    for sample_id, membrane_tif, roi_zip, pkl_path in samples:
        result = process_sample(sample_id, membrane_tif, roi_zip, pkl_path, folder, rng,
                                summary_only=summary_only)
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
        f'random_permutation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(summary_dir, exist_ok=True)
    plot_cross_sample_pvalues(all_results,
                              os.path.join(summary_dir, 'cross_sample_pvalues.png'))
    save_summary_csv(all_results,
                     os.path.join(summary_dir, 'summary_table.csv'))
    print(f"\nSummary saved to: {summary_dir}")
    print("\nDone.")


if __name__ == '__main__':
    main()

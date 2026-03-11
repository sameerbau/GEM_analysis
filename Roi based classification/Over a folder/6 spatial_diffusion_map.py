#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6 spatial_diffusion_map.py

Creates a data-driven spatial map of diffusion coefficients across the embryo
using Gaussian kernel density estimation — no ROI required.

This is the most unbiased view of your data: it asks "where in the embryo is
D high or low?" without any prior hypothesis about which regions are special.
You can then compare your drawn ROIs to where the map shows genuine D gradients.

Method:
  For each point on a regular grid within the embryo:
    - Find all trajectories within KERNEL_BANDWIDTH_UM (Gaussian kernel)
    - Compute the weighted median D (Gaussian weights by distance)
    - Assign to that grid point

Additionally computes Moran's I spatial autocorrelation statistic:
  - Tests whether nearby trajectories have more similar D values than expected
    by chance (no ROI needed)
  - High Moran's I (>0, p<0.05) means D is spatially structured across embryo

If ROI zip is provided, the drawn ROIs are overlaid on the map.

Input per sample (matched by sample ID):
  {sample_id}_membrane_.tif           membrane image (optional, for overlay)
  {sample_id}_membrane_.tif_rois.zip  ImageJ ROI zip (optional, for overlay)
  roi_trajectory_data_{sample_id}.pkl ROI-assigned pkl (from script 1)

Output per sample → {sample_id}_spatial_diffusion_map/:
  spatial_D_map.png          heatmap of median D on a grid
  spatial_D_overlay.png      heatmap overlaid on membrane image
  spatial_D_with_rois.png    heatmap + your drawn ROIs (if zip available)
  trajectory_positions.png   scatter: trajectory positions coloured by D
  morans_I_result.txt        Moran's I statistic and p-value
  spatial_map_data.csv       D value at each grid point

Cross-sample → spatial_map_summary_{timestamp}/:
  cross_sample_morans_I.png

Usage:
  python "6 spatial_diffusion_map.py"

Notes:
  - KERNEL_BANDWIDTH_UM is the sigma of the Gaussian kernel (in µm).
    Set to ~2–5× the typical step size of your particles.
  - Grid points with fewer than MIN_TRAJS_IN_KERNEL trajectories contributing
    meaningful weight are masked (shown as NaN / gray).
  - Moran's I uses a distance-based weight matrix (inverse distance, neighbours
    within MORANS_NEIGHBOUR_RADIUS_UM).
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
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from datetime import datetime
import read_roi
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk
from skimage.draw import polygon as draw_polygon
from skimage import measure
from scipy.ndimage import gaussian_filter

# ============================================================
# Parameters
# ============================================================
KERNEL_BANDWIDTH_UM      = 3.0   # µm — Gaussian sigma for spatial smoothing
GRID_SPACING_UM          = 0.5   # µm — grid resolution
MIN_TRAJS_IN_KERNEL      = 3     # minimum effective trajectories at grid point
MIN_WEIGHT_THRESHOLD     = 0.05  # minimum Gaussian weight to count as "effective"
MORANS_NEIGHBOUR_UM      = 5.0   # µm — neighbourhood radius for Moran's I
MIN_TRAJECTORIES         = 10    # skip sample below this count
CLOSING_DISK_RADIUS      = 5
DEFAULT_PX_TO_UM         = 0.094
COLORMAP                 = 'RdYlBu_r'   # red = high D, blue = low D


# ─────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────
def find_sample_files(folder):
    """
    Finds pkl files. Membrane tif and roi zip are optional (for overlay).
    """
    pkl_files = sorted(glob.glob(os.path.join(folder, 'roi_trajectory_data_*.pkl')))
    if not pkl_files:
        print(f"No roi_trajectory_data_*.pkl files found in {folder}")
        return []
    samples = []
    for pkl_path in pkl_files:
        basename  = os.path.basename(pkl_path)
        sample_id = basename.replace('roi_trajectory_data_', '').replace('.pkl', '')
        mem_tif   = os.path.join(folder, f'{sample_id}_membrane_.tif')
        roi_zip   = mem_tif + '_rois.zip'
        has_tif   = os.path.exists(mem_tif)
        has_roi   = os.path.exists(roi_zip)
        print(f"[{sample_id}] pkl=✓  tif={'✓' if has_tif else '—'}  "
              f"roi_zip={'✓' if has_roi else '—'}")
        samples.append((
            sample_id, pkl_path,
            mem_tif   if has_tif else None,
            roi_zip   if has_roi else None,
        ))
    return samples


# ─────────────────────────────────────────────────────────────
# Image helpers
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
    return binary_closing(image > threshold_otsu(image), disk(CLOSING_DISK_RADIUS))


# ─────────────────────────────────────────────────────────────
# Trajectory extraction
# ─────────────────────────────────────────────────────────────
def get_all_trajectories_um(pkl_data):
    """
    Returns list of {D, x_um, y_um} for all trajectories with valid D.
    Positions in µm.
    """
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
                'x_um': float(np.mean(traj['x'])),
                'y_um': float(np.mean(traj['y'])),
            })
    return trajs


# ─────────────────────────────────────────────────────────────
# Spatial map computation
# ─────────────────────────────────────────────────────────────
def build_spatial_grid(trajs, bandwidth_um, grid_spacing_um):
    """
    Build a Gaussian-weighted median D map on a regular grid in µm.

    Returns:
        grid_x, grid_y  : 2D arrays of grid coordinates (µm)
        D_map           : 2D array of weighted median D (NaN where too sparse)
        weight_map      : 2D array of total Gaussian weight (proxy for density)
        n_eff_map       : 2D array of effective trajectory count at each grid point
    """
    if not trajs:
        return None, None, None, None, None

    xs = np.array([t['x_um'] for t in trajs])
    ys = np.array([t['y_um'] for t in trajs])
    Ds = np.array([t['D']    for t in trajs])

    # Grid extent with margin
    margin  = 2 * bandwidth_um
    x_range = np.arange(xs.min() - margin, xs.max() + margin, grid_spacing_um)
    y_range = np.arange(ys.min() - margin, ys.max() + margin, grid_spacing_um)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    gx_flat = grid_x.ravel()
    gy_flat = grid_y.ravel()

    D_map_flat     = np.full(len(gx_flat), np.nan)
    weight_flat    = np.zeros(len(gx_flat))
    n_eff_flat     = np.zeros(len(gx_flat))

    sig2 = bandwidth_um ** 2

    for i, (gx, gy) in enumerate(zip(gx_flat, gy_flat)):
        dist2   = (xs - gx)**2 + (ys - gy)**2
        weights = np.exp(-0.5 * dist2 / sig2)
        total_w = weights.sum()
        n_eff   = (weights >= MIN_WEIGHT_THRESHOLD * weights.max()
                   if weights.max() > 0 else weights > 0)
        n_eff_count = int(n_eff.sum())

        weight_flat[i] = total_w

        if n_eff_count < MIN_TRAJS_IN_KERNEL:
            continue

        # Weighted median: sort by D, find quantile where cumulative weight = 0.5
        sort_idx      = np.argsort(Ds)
        cum_w         = np.cumsum(weights[sort_idx])
        cum_w        /= cum_w[-1]
        median_idx    = np.searchsorted(cum_w, 0.5)
        D_map_flat[i] = Ds[sort_idx[median_idx]]
        n_eff_flat[i] = n_eff_count

    D_map     = D_map_flat.reshape(grid_x.shape)
    weight_map = weight_flat.reshape(grid_x.shape)
    n_eff_map  = n_eff_flat.reshape(grid_x.shape)

    return grid_x, grid_y, D_map, weight_map, n_eff_map


# ─────────────────────────────────────────────────────────────
# Moran's I
# ─────────────────────────────────────────────────────────────
def compute_morans_I(trajs, neighbour_radius_um, n_permutations=999):
    """
    Compute Moran's I spatial autocorrelation for D values.
    Uses inverse-distance weights for pairs within neighbour_radius_um.
    Permutation test for p-value.

    Returns dict: {'I': float, 'expected_I': float, 'p_value': float,
                   'z_score': float, 'n': int}
    """
    if len(trajs) < 10:
        return None

    xs = np.array([t['x_um'] for t in trajs])
    ys = np.array([t['y_um'] for t in trajs])
    Ds = np.array([t['D']    for t in trajs])
    n  = len(Ds)
    D_mean = Ds.mean()
    D_dev  = Ds - D_mean

    # Build weight matrix (inverse distance, within radius)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dx   = xs - xs[i]
        dy   = ys - ys[i]
        dist = np.sqrt(dx**2 + dy**2)
        mask = (dist > 0) & (dist <= neighbour_radius_um)
        W[i, mask] = 1.0 / dist[mask]

    W_sum = W.sum()
    if W_sum == 0:
        return None

    # Moran's I
    numerator   = n * np.sum(W * np.outer(D_dev, D_dev))
    denominator = W_sum * np.sum(D_dev**2)
    I_obs = numerator / denominator if denominator != 0 else np.nan

    # Expected value under H0
    E_I = -1.0 / (n - 1)

    # Permutation p-value
    rng = np.random.default_rng(0)
    I_perm = np.empty(n_permutations)
    for k in range(n_permutations):
        D_perm = rng.permutation(Ds)
        d_dev  = D_perm - D_perm.mean()
        num_p  = n * np.sum(W * np.outer(d_dev, d_dev))
        den_p  = W_sum * np.sum(d_dev**2)
        I_perm[k] = num_p / den_p if den_p != 0 else np.nan

    I_perm_clean = I_perm[~np.isnan(I_perm)]
    p_value = float(np.mean(np.abs(I_perm_clean - E_I) >= abs(I_obs - E_I)))
    z_score = (I_obs - E_I) / I_perm_clean.std() if I_perm_clean.std() > 0 else np.nan

    return {
        'I':          I_obs,
        'expected_I': E_I,
        'p_value':    p_value,
        'z_score':    z_score,
        'n':          n,
    }


# ─────────────────────────────────────────────────────────────
# ROI polygon loading
# ─────────────────────────────────────────────────────────────
def load_roi_polygons_um(roi_zip_path, px_to_um):
    """Load ROI polygons and convert coordinates from pixels to µm."""
    rois_raw = read_roi.read_roi_zip(roi_zip_path)
    rois = {}
    for roi_id, roi in rois_raw.items():
        if 'x' not in roi or 'y' not in roi:
            continue
        rois[roi_id] = {
            'x_um': np.array(roi['x'], dtype=float) * px_to_um,
            'y_um': np.array(roi['y'], dtype=float) * px_to_um,
        }
    return rois


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
def _common_D_limits(D_map):
    """Robust colour limits for D map."""
    valid = D_map[~np.isnan(D_map)]
    if len(valid) == 0:
        return 0, 1
    return np.percentile(valid, 2), np.percentile(valid, 98)


def plot_scatter_by_D(trajs, rois_um, sample_id, output_path):
    """Scatter plot of trajectory positions coloured by D."""
    xs = np.array([t['x_um'] for t in trajs])
    ys = np.array([t['y_um'] for t in trajs])
    Ds = np.array([t['D']    for t in trajs])

    vmin, vmax = np.percentile(Ds, [2, 98])

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xs, ys, c=Ds, cmap=COLORMAP, s=8, alpha=0.7,
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='D (µm²/s)', shrink=0.7)

    if rois_um:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(rois_um), 1)))
        for i, (roi_id, roi) in enumerate(rois_um.items()):
            poly = MplPolygon(list(zip(roi['x_um'], roi['y_um'])),
                              closed=True, fill=False,
                              edgecolor=colors[i % len(colors)], linewidth=2)
            ax.add_patch(poly)
            ax.text(roi['x_um'].mean(), roi['y_um'].mean(),
                    roi_id.split('-')[0][:8], ha='center', va='center',
                    fontsize=7, color=colors[i % len(colors)])

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
    ax.set_title(f'Trajectory positions coloured by D\n{sample_id}', fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_D_map(grid_x, grid_y, D_map, rois_um, sample_id, output_path,
               background_image=None, px_to_um=DEFAULT_PX_TO_UM):
    """
    Plot the spatial D map as a heatmap.
    If background_image is provided, overlay on it.
    """
    vmin, vmax = _common_D_limits(D_map)

    fig, ax = plt.subplots(figsize=(8, 8))

    if background_image is not None:
        h, w = background_image.shape
        # Image extent in µm
        extent_img = [0, w * px_to_um, h * px_to_um, 0]
        bg_vmin, bg_vmax = (np.percentile(background_image, [1, 99]))
        ax.imshow(background_image, cmap='gray', vmin=bg_vmin, vmax=bg_vmax,
                  extent=extent_img, origin='upper', aspect='equal', zorder=0)

    # D map as semi-transparent overlay
    # Mask NaN regions
    D_masked = np.ma.masked_invalid(D_map)
    pcm = ax.pcolormesh(grid_x, grid_y, D_masked,
                        cmap=COLORMAP, vmin=vmin, vmax=vmax,
                        alpha=0.70 if background_image is not None else 1.0,
                        shading='auto', zorder=2)
    plt.colorbar(pcm, ax=ax, label='D (µm²/s)', shrink=0.7)

    # ROI overlays
    if rois_um:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(rois_um), 1)))
        for i, (roi_id, roi) in enumerate(rois_um.items()):
            poly = MplPolygon(list(zip(roi['x_um'], roi['y_um'])),
                              closed=True, fill=False,
                              edgecolor=colors[i % len(colors)],
                              linewidth=2, zorder=5)
            ax.add_patch(poly)
            ax.text(roi['x_um'].mean(), roi['y_um'].mean(),
                    roi_id.split('-')[0][:8], ha='center', va='center',
                    fontsize=7, color='white',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))

    ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
    roi_str = ' + drawn ROIs' if rois_um else ''
    ax.set_title(f'Spatial D map (Gaussian kernel σ={KERNEL_BANDWIDTH_UM} µm)\n'
                 f'{sample_id}{roi_str}', fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def save_map_csv(grid_x, grid_y, D_map, n_eff_map, output_path):
    rows = []
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            D = D_map[i, j]
            if np.isnan(D):
                continue
            rows.append({
                'x_um':     round(float(grid_x[i, j]), 4),
                'y_um':     round(float(grid_y[i, j]), 4),
                'D_median': round(D, 6),
                'n_eff':    int(n_eff_map[i, j]),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Moran's I output
# ─────────────────────────────────────────────────────────────
def save_morans_result(result, sample_id, output_path):
    if result is None:
        with open(output_path, 'w') as f:
            f.write("Insufficient data for Moran's I\n")
        return
    lines = [
        f"Moran's I Spatial Autocorrelation — {sample_id}",
        "=" * 50,
        f"  n trajectories : {result['n']}",
        f"  Neighbourhood  : {MORANS_NEIGHBOUR_UM} µm",
        f"",
        f"  Moran's I      : {result['I']:.4f}",
        f"  Expected I     : {result['expected_I']:.4f}",
        f"  z-score        : {result['z_score']:.3f}",
        f"  p-value (perm) : {result['p_value']:.4f}",
        f"",
        "Interpretation:",
    ]
    I, p = result['I'], result['p_value']
    if p < 0.05:
        if I > result['expected_I']:
            lines.append("  SIGNIFICANT positive spatial autocorrelation.")
            lines.append("  Nearby trajectories have MORE similar D than expected.")
            lines.append("  => D is spatially structured across the embryo.")
        else:
            lines.append("  SIGNIFICANT negative spatial autocorrelation.")
            lines.append("  Nearby trajectories have LESS similar D than expected.")
    else:
        lines.append("  No significant spatial autocorrelation (p >= 0.05).")
        lines.append("  D values appear randomly distributed across the embryo.")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    for line in lines:
        print(f"    {line}")
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Cross-sample Moran's I summary
# ─────────────────────────────────────────────────────────────
def plot_morans_summary(all_morans, output_path):
    samples = list(all_morans.keys())
    Is      = [all_morans[s]['I']       if all_morans[s] else np.nan for s in samples]
    ps      = [all_morans[s]['p_value'] if all_morans[s] else np.nan for s in samples]
    zs      = [all_morans[s]['z_score'] if all_morans[s] else np.nan for s in samples]

    x = np.arange(len(samples))
    colors = ['limegreen' if p < 0.05 else 'steelblue' for p in ps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(samples) * 1.5 + 4), 5))

    ax1.bar(x, Is, color=colors, alpha=0.85)
    expected = -1.0 / max(1, len(samples))
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels(samples, rotation=45, ha='right')
    ax1.set_ylabel("Moran's I"); ax1.set_title("Moran's I per sample\n(green = p<0.05)")
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in ps],
            color=colors, alpha=0.85)
    ax2.axhline(-np.log10(0.05), color='gray', linestyle='--',
                linewidth=1, label='p=0.05')
    ax2.set_xticks(x); ax2.set_xticklabels(samples, rotation=45, ha='right')
    ax2.set_ylabel("−log₁₀(p)"); ax2.set_title("Moran's I significance\n(above line = p<0.05)")
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    plt.suptitle("Spatial autocorrelation of D across samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Per-sample pipeline
# ─────────────────────────────────────────────────────────────
def process_sample(sample_id, pkl_path, mem_tif, roi_zip, folder):
    print(f"\n{'='*60}")
    print(f"  {sample_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(folder, f'{sample_id}_spatial_diffusion_map')
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectories
    print("  Loading trajectory pkl...")
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
    except Exception as e:
        print(f"  ERROR loading pkl: {e}"); return None, None

    ct     = pkl_data.get('coordinate_transform', {})
    px_to_um = ct.get('pixel_to_micron', DEFAULT_PX_TO_UM)

    trajs = get_all_trajectories_um(pkl_data)
    if len(trajs) < MIN_TRAJECTORIES:
        print(f"  Only {len(trajs)} trajectories with valid D — skipping."); return None, None
    print(f"  {len(trajs)} trajectories with valid D")

    # Optional: membrane image
    image = None
    if mem_tif is not None:
        try:
            image = load_membrane_image(mem_tif)
            print(f"  Loaded membrane image: {image.shape}")
        except Exception as e:
            print(f"  Could not load membrane image: {e}")

    # Optional: ROI polygons (for overlay)
    rois_um = {}
    if roi_zip is not None:
        try:
            rois_um = load_roi_polygons_um(roi_zip, px_to_um)
            print(f"  Loaded {len(rois_um)} ROI(s) for overlay")
        except Exception as e:
            print(f"  Could not load ROIs: {e}")

    # ── Build spatial D map ──────────────────────────────────
    print(f"  Building spatial D map (σ={KERNEL_BANDWIDTH_UM} µm, "
          f"grid={GRID_SPACING_UM} µm)...")
    grid_x, grid_y, D_map, weight_map, n_eff_map = build_spatial_grid(
        trajs, KERNEL_BANDWIDTH_UM, GRID_SPACING_UM
    )
    if D_map is None:
        print("  Failed to build grid — skipping."); return None, None

    n_valid = int(np.sum(~np.isnan(D_map)))
    print(f"  Grid: {D_map.shape[1]}×{D_map.shape[0]}, "
          f"{n_valid} valid grid points")

    # ── Scatter plot ─────────────────────────────────────────
    plot_scatter_by_D(trajs, rois_um, sample_id,
                      os.path.join(output_dir, 'trajectory_positions.png'))

    # ── D map (no background) ────────────────────────────────
    plot_D_map(grid_x, grid_y, D_map, rois_um, sample_id,
               os.path.join(output_dir, 'spatial_D_map.png'),
               px_to_um=px_to_um)

    # ── D map overlay on membrane ────────────────────────────
    if image is not None:
        plot_D_map(grid_x, grid_y, D_map, rois_um, sample_id,
                   os.path.join(output_dir, 'spatial_D_overlay.png'),
                   background_image=image, px_to_um=px_to_um)

    # ── CSV ──────────────────────────────────────────────────
    save_map_csv(grid_x, grid_y, D_map, n_eff_map,
                 os.path.join(output_dir, 'spatial_map_data.csv'))

    # ── Moran's I ────────────────────────────────────────────
    print(f"\n  Computing Moran's I (neighbourhood {MORANS_NEIGHBOUR_UM} µm)...")
    # Subsample for speed if very large dataset
    trajs_for_morans = trajs
    if len(trajs) > 2000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(trajs), 2000, replace=False)
        trajs_for_morans = [trajs[i] for i in idx]
        print(f"  Subsampled to 2000 trajectories for Moran's I computation")
    morans = compute_morans_I(trajs_for_morans, MORANS_NEIGHBOUR_UM)
    save_morans_result(morans,
                       sample_id,
                       os.path.join(output_dir, 'morans_I_result.txt'))

    print(f"\n  Output: {output_dir}")
    return D_map, morans


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global KERNEL_BANDWIDTH_UM
    print("Spatial Diffusion Map")
    print("======================\n")

    folder = input("Enter path to data folder: ").strip()
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}"); sys.exit(1)

    # Optional: prompt for bandwidth
    bw_input = input(f"Kernel bandwidth (µm) [{KERNEL_BANDWIDTH_UM}]: ").strip()
    if bw_input:
        try:
            KERNEL_BANDWIDTH_UM = float(bw_input)
        except ValueError:
            pass
    print(f"Using kernel bandwidth = {KERNEL_BANDWIDTH_UM} µm\n")

    samples = find_sample_files(folder)
    if not samples:
        print("No pkl files found. Exiting."); sys.exit(1)

    print(f"\n{len(samples)} sample(s) to process.\n")
    all_morans = {}
    for sample_id, pkl_path, mem_tif, roi_zip in samples:
        _, morans = process_sample(sample_id, pkl_path, mem_tif, roi_zip, folder)
        all_morans[sample_id] = morans

    if not any(v is not None for v in all_morans.values()):
        print("\nNo samples processed successfully."); sys.exit(1)

    # Cross-sample Moran's I summary
    print(f"\n{'='*60}")
    print("Cross-sample Moran's I summary")
    print(f"{'='*60}")
    summary_dir = os.path.join(
        folder,
        f'spatial_map_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(summary_dir, exist_ok=True)

    morans_clean = {k: v for k, v in all_morans.items() if v is not None}
    if morans_clean:
        plot_morans_summary(morans_clean,
                            os.path.join(summary_dir, 'cross_sample_morans_I.png'))
        # CSV summary
        rows = []
        for sid, m in morans_clean.items():
            rows.append({
                'sample_id': sid,
                'morans_I':  round(m['I'], 4),
                'expected_I':round(m['expected_I'], 4),
                'z_score':   round(m['z_score'], 3),
                'p_value':   m['p_value'],
                'n':         m['n'],
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(summary_dir, 'morans_I_summary.csv'), index=False
        )
        print(f"  Saved: morans_I_summary.csv")

    print(f"\nSummary saved to: {summary_dir}")
    print("\nDone.")


if __name__ == '__main__':
    main()

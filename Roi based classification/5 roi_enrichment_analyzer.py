#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 roi_enrichment_analyzer.py

Tests whether GEM trajectories are disproportionately enriched or depleted
inside ROIs compared to what would be expected from random distribution across
the embryo.

Normalization uses the embryo area derived from an Otsu-thresholded membrane
image, NOT the full image area, since trajectories can only exist within the
embryo.

Statistical test: two-sided binomial test.
  H0: P(trajectory inside ROI) = ROI_area / embryo_area
  H1: P(trajectory inside ROI) ≠ ROI_area / embryo_area

Enrichment ratio = (observed / total) / (ROI_area / embryo_area)
  > 1 : enriched inside ROI
  < 1 : depleted inside ROI

Input files matched by sample ID (e.g. 'Em1.nd2'):
  Em1.nd2_membrane_.tif           membrane channel image
  Em1.nd2_membrane_.tif_rois.zip  ImageJ ROI zip
  roi_trajectory_data_Em1.nd2.pkl ROI-assigned trajectory pkl (from script 1)

Output per sample → {sample_id}_enrichment/:
  membrane_otsu_mask.png
  enrichment_overlay.png
  observed_vs_expected_barplot.png
  enrichment_ratio_plot.png
  pooled_bar.png
  enrichment_results.csv

Cross-sample summary → enrichment_summary_{timestamp}/:
  all_samples_pooled_enrichment.png
  grand_pooled_enrichment.png
  summary_table.csv

Usage:
  python "5 roi_enrichment_analyzer.py"

Notes:
  - Assumes ROIs are non-overlapping. Overlapping ROIs will cause the pooled
    ROI area to be overestimated.
  - Morphological closing is applied to the Otsu mask to fill membrane gaps.
  - Trajectory coordinates in the pkl are in µm; converted to pixels using the
    coordinate_transform stored by script 1.
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
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from datetime import datetime

import read_roi
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk
from skimage import measure

try:
    from scipy.stats import binomtest
    def _binomtest(k, n, p):
        result = binomtest(k, n, p, alternative='two-sided')
        ci = result.proportion_ci(confidence_level=0.95, method='wilson')
        return result.pvalue, ci.low, ci.high
except ImportError:
    # scipy < 1.7 fallback
    from scipy.stats import binom_test
    from scipy.stats import proportion_confint
    def _binomtest(k, n, p):
        pvalue = binom_test(k, n, p, alternative='two-sided')
        ci_low, ci_high = proportion_confint(k, n, alpha=0.05, method='wilson')
        return pvalue, ci_low, ci_high

# ============================================================
# Parameters
# ============================================================
MIN_ROI_AREA_PIXELS  = 50   # ROIs smaller than this are skipped
MIN_TRAJECTORIES     = 5    # skip sample if fewer total trajectories
CLOSING_DISK_RADIUS  = 5    # morphological closing radius for embryo mask
DEFAULT_PX_TO_UM     = 0.094  # fallback if not found in pkl
# ============================================================


# ─────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────

def find_sample_files(folder):
    """
    Scan folder for complete sets of (membrane_tif, roi_zip, pkl).

    Naming convention:
      {sample_id}_membrane_.tif
      {sample_id}_membrane_.tif_rois.zip
      roi_trajectory_data_{sample_id}.pkl

    Returns list of (sample_id, membrane_tif, roi_zip, pkl_path).
    """
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
# Image processing
# ─────────────────────────────────────────────────────────────

def load_membrane_image(tif_path):
    """Load membrane TIFF as a 2D float32 array."""
    img = tifffile.imread(tif_path).astype(np.float32)

    if img.ndim == 2:
        return img
    if img.ndim == 3:
        # (Z/C, Y, X) → first slice; (Y, X, C) → max across last axis
        if img.shape[0] <= img.shape[-1]:
            return img[0]
        return img.max(axis=-1)
    if img.ndim == 4:
        # (T, Z, Y, X) → first timepoint, max-Z projection
        return img[0].max(axis=0)

    raise ValueError(f"Unexpected image dimensions: {img.shape}")


def compute_embryo_mask(image):
    """
    Otsu threshold + morphological closing → binary embryo mask.

    Returns (mask, threshold_value).
    """
    threshold = threshold_otsu(image)
    mask = binary_closing(image > threshold, disk(CLOSING_DISK_RADIUS))
    return mask, threshold


# ─────────────────────────────────────────────────────────────
# ROI geometry
# ─────────────────────────────────────────────────────────────

def polygon_area(x_coords, y_coords):
    """Shoelace formula — polygon area in pixels²."""
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load_roi_polygons(roi_zip_path):
    """
    Load polygon ROIs from ImageJ ZIP.

    Returns dict: roi_id → {'x': array, 'y': array, 'area_pixels': float}
    Skips ROIs without polygon coordinates or below MIN_ROI_AREA_PIXELS.
    """
    rois_raw = read_roi.read_roi_zip(roi_zip_path)
    rois = {}

    for roi_id, roi in rois_raw.items():
        if 'x' not in roi or 'y' not in roi:
            print(f"    Skipping {roi_id}: no polygon coordinates")
            continue
        area = polygon_area(roi['x'], roi['y'])
        if area < MIN_ROI_AREA_PIXELS:
            print(f"    Skipping {roi_id}: area {area:.1f} px² < minimum")
            continue
        rois[roi_id] = {
            'x': np.array(roi['x'], dtype=float),
            'y': np.array(roi['y'], dtype=float),
            'area_pixels': area,
        }

    return rois


# ─────────────────────────────────────────────────────────────
# Trajectory data
# ─────────────────────────────────────────────────────────────

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_traj_pixel_positions(pkl_data):
    """
    Extract mean (x, y) pixel positions for all trajectories grouped by ROI.

    Uses coordinate_transform stored in pkl to convert µm → pixels.

    Returns dict: roi_id → list of (x_px, y_px)
    """
    ct = pkl_data.get('coordinate_transform', {})
    scale   = ct.get('scale_factor', 1.0 / DEFAULT_PX_TO_UM)
    x_off   = ct.get('x_offset', 0.0)
    y_off   = ct.get('y_offset', 0.0)

    positions = {}
    for roi_id, trajs in pkl_data.get('roi_trajectories', {}).items():
        pts = []
        for traj in trajs:
            if 'x' in traj and 'y' in traj:
                xp = float(np.mean(traj['x'])) * scale + x_off
                yp = float(np.mean(traj['y'])) * scale + y_off
            elif 'mean_x' in traj and 'mean_y' in traj:
                xp = traj['mean_x'] * scale + x_off
                yp = traj['mean_y'] * scale + y_off
            else:
                continue
            pts.append((xp, yp))
        positions[roi_id] = pts

    return positions


# ─────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────

def enrichment_stats(n_observed, n_total, roi_area, embryo_area):
    """
    Compute enrichment ratio, 95 % Wilson CI, and binomial p-value.

    Returns a dict of all results.
    """
    expected_fraction   = roi_area / embryo_area
    expected_count      = n_total * expected_fraction
    observed_fraction   = n_observed / n_total if n_total > 0 else 0.0
    enrichment_ratio    = (observed_fraction / expected_fraction
                           if expected_fraction > 0 else np.nan)

    p_value, ci_low_frac, ci_high_frac = _binomtest(n_observed, n_total,
                                                     expected_fraction)

    enrich_ci_low  = ci_low_frac  / expected_fraction if expected_fraction > 0 else np.nan
    enrich_ci_high = ci_high_frac / expected_fraction if expected_fraction > 0 else np.nan

    return {
        'n_observed':        n_observed,
        'n_total':           n_total,
        'expected_count':    expected_count,
        'expected_fraction': expected_fraction,
        'observed_fraction': observed_fraction,
        'enrichment_ratio':  enrichment_ratio,
        'enrichment_ci_low': enrich_ci_low,
        'enrichment_ci_high':enrich_ci_high,
        'p_value':           p_value,
        'significant':       p_value < 0.05,
        'roi_area_pixels':   roi_area,
        'embryo_area_pixels':embryo_area,
    }


def _sig_label(p):
    if p < 0.001: return 'p<0.001'
    if p < 0.01:  return f'p={p:.3f}'
    return f'p={p:.3f}'


def _point_color(significant, ratio):
    if not significant: return 'steelblue'
    return 'darkgreen' if ratio >= 1 else 'firebrick'


# ─────────────────────────────────────────────────────────────
# Plots — per sample
# ─────────────────────────────────────────────────────────────

def plot_otsu_mask(image, mask, threshold, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image, cmap='gray', interpolation='nearest')
    axes[0].set_title('Membrane image (raw)')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'Otsu binary mask  (threshold = {threshold:.1f})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_overlay(image, mask, rois, traj_positions_pixels, output_path):
    """
    Membrane image + embryo contour + ROI polygons +
    trajectory mean positions (inside ROI = colored, outside = gray).
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display image with contrast stretch inside embryo
    vmin, vmax = (np.percentile(image[mask], [1, 99])
                  if mask.sum() > 0 else (image.min(), image.max()))
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Embryo mask contour
    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color='cyan',
                linewidth=1.5, alpha=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rois), 1)))
    legend_handles = []

    for i, (roi_id, roi) in enumerate(rois.items()):
        color    = colors[i % len(colors)]
        short_id = roi_id.split('-')[0][:12]

        # ROI polygon outline
        poly_patch = MplPolygon(list(zip(roi['x'], roi['y'])),
                                closed=True, fill=False,
                                edgecolor=color, linewidth=2)
        ax.add_patch(poly_patch)

        # Trajectory positions inside this ROI
        pts = traj_positions_pixels.get(roi_id, [])
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=8, color=color, alpha=0.6, zorder=3)

        legend_handles.append(
            mpatches.Patch(color=color, label=f'{short_id}  (n={len(pts)})')
        )

    # Unassigned / outside trajectories
    outside = traj_positions_pixels.get('unassigned', [])
    if outside:
        xs, ys = zip(*outside)
        ax.scatter(xs, ys, s=4, color='lightgray', alpha=0.3, zorder=2)
        legend_handles.append(
            mpatches.Patch(color='lightgray', label=f'Outside  (n={len(outside)})')
        )

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Trajectory assignments on membrane image', fontsize=12)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_observed_vs_expected(per_roi_results, output_path):
    labels   = [r['roi_id'].split('-')[0][:12] for r in per_roi_results]
    observed = [r['n_observed']     for r in per_roi_results]
    expected = [r['expected_count'] for r in per_roi_results]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4 + 2), 5))
    ax.bar(x - width / 2, observed, width, label='Observed',
           color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, expected, width, label='Expected',
           color='coral',     alpha=0.85)

    # Significance markers
    for i, r in enumerate(per_roi_results):
        if r['significant']:
            ymax = max(observed[i], expected[i]) * 1.05
            ax.text(x[i], ymax, '*', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('ROI')
    ax.set_ylabel('Number of trajectories')
    ax.set_title('Observed vs Expected trajectory counts per ROI')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_enrichment_ratios(per_roi_results, output_path):
    labels   = [r['roi_id'].split('-')[0][:12] for r in per_roi_results]
    ratios   = [r['enrichment_ratio']  for r in per_roi_results]
    ci_lows  = [r['enrichment_ci_low'] for r in per_roi_results]
    ci_highs = [r['enrichment_ci_high']for r in per_roi_results]

    x      = np.arange(len(labels))
    colors = [_point_color(r['significant'], r['enrichment_ratio'])
              for r in per_roi_results]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4 + 2), 5))

    for i in range(len(labels)):
        ax.errorbar(x[i], ratios[i],
                    yerr=[[ratios[i] - ci_lows[i]],
                          [ci_highs[i] - ratios[i]]],
                    fmt='o', color=colors[i], markersize=9,
                    ecolor=colors[i], elinewidth=2, capsize=5)
        ax.text(x[i], ci_highs[i] * 1.08,
                _sig_label(per_roi_results[i]['p_value']),
                ha='center', va='bottom', fontsize=7)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
               label='Expected (ratio = 1)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('ROI')
    ax.set_ylabel('Enrichment ratio (log scale)')
    ax.set_title('Trajectory enrichment ratio per ROI\n'
                 'green = enriched, red = depleted, blue = n.s.   '
                 '(95 % Wilson CI)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_pooled_bar(pooled, sample_id, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left — counts
    ax1.bar(['Observed', 'Expected'],
            [pooled['n_observed'], pooled['expected_count']],
            color=['steelblue', 'coral'], alpha=0.85, width=0.5)
    ax1.set_ylabel('Trajectories inside ROIs (all combined)')
    ax1.set_title('Pooled: all ROIs combined')
    ax1.grid(axis='y', alpha=0.3)
    p_str = _sig_label(pooled['p_value'])
    ax1.text(0.5, max(pooled['n_observed'], pooled['expected_count']) * 1.03,
             p_str, ha='center', fontsize=11)

    # Right — enrichment ratio
    ratio  = pooled['enrichment_ratio']
    ci_lo  = pooled['enrichment_ci_low']
    ci_hi  = pooled['enrichment_ci_high']
    color  = _point_color(pooled['significant'], ratio)

    ax2.errorbar(0, ratio,
                 yerr=[[ratio - ci_lo], [ci_hi - ratio]],
                 fmt='o', color=color, markersize=12,
                 ecolor=color, elinewidth=3, capsize=8)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_yscale('log')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['All ROIs'])
    ax2.set_ylabel('Enrichment ratio (log scale)')
    ax2.set_title(f'Pooled enrichment\n{p_str}')
    ax2.grid(alpha=0.3)

    plt.suptitle(f'Sample: {sample_id}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def save_sample_csv(per_roi_results, pooled, sample_id, output_path):
    rows = []
    for r in per_roi_results:
        rows.append({
            'sample_id':           sample_id,
            'roi_id':              r['roi_id'],
            'type':                'per_roi',
            'n_observed':          r['n_observed'],
            'n_total':             r['n_total'],
            'expected_count':      round(r['expected_count'], 2),
            'roi_area_pixels2':    round(r['roi_area_pixels'], 1),
            'embryo_area_pixels2': round(r['embryo_area_pixels'], 1),
            'roi_fraction_of_embryo': round(r['expected_fraction'], 6),
            'observed_fraction':   round(r['observed_fraction'], 6),
            'enrichment_ratio':    round(r['enrichment_ratio'], 4),
            'enrichment_ci_low':   round(r['enrichment_ci_low'], 4),
            'enrichment_ci_high':  round(r['enrichment_ci_high'], 4),
            'p_value':             r['p_value'],
            'significant':         r['significant'],
        })

    rows.append({
        'sample_id':           sample_id,
        'roi_id':              'POOLED',
        'type':                'pooled',
        'n_observed':          pooled['n_observed'],
        'n_total':             pooled['n_total'],
        'expected_count':      round(pooled['expected_count'], 2),
        'roi_area_pixels2':    round(pooled['roi_area_pixels'], 1),
        'embryo_area_pixels2': round(pooled['embryo_area_pixels'], 1),
        'roi_fraction_of_embryo': round(pooled['expected_fraction'], 6),
        'observed_fraction':   round(pooled['observed_fraction'], 6),
        'enrichment_ratio':    round(pooled['enrichment_ratio'], 4),
        'enrichment_ci_low':   round(pooled['enrichment_ci_low'], 4),
        'enrichment_ci_high':  round(pooled['enrichment_ci_high'], 4),
        'p_value':             pooled['p_value'],
        'significant':         pooled['significant'],
    })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Plots — cross-sample
# ─────────────────────────────────────────────────────────────

def plot_cross_sample(all_results, output_dir):
    """
    Two figures:
      1. Pooled enrichment ratio per sample (one point per embryo).
      2. Grand total — all samples and ROIs combined.
    Returns grand_result dict.
    """
    sample_ids = list(all_results.keys())
    pooled_per_sample = [all_results[s]['pooled'] for s in sample_ids]

    ratios   = [r['enrichment_ratio']  for r in pooled_per_sample]
    ci_lows  = [r['enrichment_ci_low'] for r in pooled_per_sample]
    ci_highs = [r['enrichment_ci_high']for r in pooled_per_sample]
    colors   = [_point_color(r['significant'], r['enrichment_ratio'])
                for r in pooled_per_sample]

    x = np.arange(len(sample_ids))

    # ── Figure 1: per-sample pooled enrichment ──────────────
    fig, ax = plt.subplots(figsize=(max(6, len(sample_ids) * 1.5 + 2), 5))

    for i in range(len(sample_ids)):
        ax.errorbar(x[i], ratios[i],
                    yerr=[[ratios[i] - ci_lows[i]],
                          [ci_highs[i] - ratios[i]]],
                    fmt='o', color=colors[i], markersize=9,
                    ecolor=colors[i], elinewidth=2, capsize=5)
        ax.text(x[i], ci_highs[i] * 1.08,
                _sig_label(pooled_per_sample[i]['p_value']),
                ha='center', va='bottom', fontsize=7)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids, rotation=45, ha='right')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Enrichment ratio (log scale)')
    ax.set_title('Pooled enrichment ratio per sample  (all ROIs combined)\n'
                 'green = enriched, red = depleted, blue = n.s.   (95 % Wilson CI)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(output_dir, 'all_samples_pooled_enrichment.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out1)}")

    # ── Grand total ──────────────────────────────────────────
    total_observed    = sum(r['n_observed']        for r in pooled_per_sample)
    total_total       = sum(r['n_total']           for r in pooled_per_sample)
    total_roi_area    = sum(r['roi_area_pixels']   for r in pooled_per_sample)
    total_embryo_area = sum(r['embryo_area_pixels']for r in pooled_per_sample)

    grand = enrichment_stats(total_observed, total_total,
                             total_roi_area, total_embryo_area)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.bar(['Observed', 'Expected'],
            [grand['n_observed'], grand['expected_count']],
            color=['steelblue', 'coral'], alpha=0.85, width=0.5)
    ax1.set_ylabel('Trajectories inside ROIs')
    ax1.set_title(f'Grand total  ({len(sample_ids)} samples combined)')
    ax1.grid(axis='y', alpha=0.3)
    p_str = _sig_label(grand['p_value'])
    ax1.text(0.5, max(grand['n_observed'], grand['expected_count']) * 1.03,
             p_str, ha='center', fontsize=11)

    ratio = grand['enrichment_ratio']
    ci_lo = grand['enrichment_ci_low']
    ci_hi = grand['enrichment_ci_high']
    color = _point_color(grand['significant'], ratio)

    ax2.errorbar(0, ratio,
                 yerr=[[ratio - ci_lo], [ci_hi - ratio]],
                 fmt='o', color=color, markersize=12,
                 ecolor=color, elinewidth=3, capsize=8)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_yscale('log')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['All samples'])
    ax2.set_ylabel('Enrichment ratio (log scale)')
    ax2.set_title(f'Grand pooled enrichment\n{p_str}')
    ax2.grid(alpha=0.3)

    plt.suptitle('All samples + ROIs combined', fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(output_dir, 'grand_pooled_enrichment.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out2)}")

    return grand


def save_summary_csv(all_results, grand_result, output_path):
    rows = []
    for sample_id, res in all_results.items():
        p = res['pooled']
        rows.append({
            'sample_id':              sample_id,
            'n_rois':                 len(res['per_roi']),
            'n_observed_in_rois':     p['n_observed'],
            'n_total_trajectories':   p['n_total'],
            'expected_count':         round(p['expected_count'], 2),
            'roi_fraction_of_embryo': round(p['expected_fraction'], 6),
            'roi_area_pixels2':       round(p['roi_area_pixels'], 1),
            'embryo_area_pixels2':    round(p['embryo_area_pixels'], 1),
            'enrichment_ratio':       round(p['enrichment_ratio'], 4),
            'enrichment_ci_low':      round(p['enrichment_ci_low'], 4),
            'enrichment_ci_high':     round(p['enrichment_ci_high'], 4),
            'p_value':                p['p_value'],
            'significant':            p['significant'],
        })

    g = grand_result
    rows.append({
        'sample_id':              'GRAND_TOTAL',
        'n_rois':                 sum(len(r['per_roi']) for r in all_results.values()),
        'n_observed_in_rois':     g['n_observed'],
        'n_total_trajectories':   g['n_total'],
        'expected_count':         round(g['expected_count'], 2),
        'roi_fraction_of_embryo': round(g['expected_fraction'], 6),
        'roi_area_pixels2':       round(g['roi_area_pixels'], 1),
        'embryo_area_pixels2':    round(g['embryo_area_pixels'], 1),
        'enrichment_ratio':       round(g['enrichment_ratio'], 4),
        'enrichment_ci_low':      round(g['enrichment_ci_low'], 4),
        'enrichment_ci_high':     round(g['enrichment_ci_high'], 4),
        'p_value':                g['p_value'],
        'significant':            g['significant'],
    })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Per-sample pipeline
# ─────────────────────────────────────────────────────────────

def process_sample(sample_id, membrane_tif, roi_zip, pkl_path, folder):
    """
    Run full enrichment analysis for one sample.
    Returns {'per_roi': [...], 'pooled': {...}} or None on failure.
    """
    print(f"\n{'='*60}")
    print(f"  {sample_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(folder, f'{sample_id}_enrichment')
    os.makedirs(output_dir, exist_ok=True)

    # Membrane image + Otsu mask
    print("  Loading membrane image and computing embryo mask...")
    try:
        image = load_membrane_image(membrane_tif)
    except Exception as e:
        print(f"  ERROR loading image: {e}")
        return None

    mask, threshold = compute_embryo_mask(image)
    embryo_area     = int(mask.sum())
    print(f"  Embryo area = {embryo_area:,} px²  (Otsu threshold = {threshold:.1f})")

    plot_otsu_mask(image, mask, threshold,
                   os.path.join(output_dir, 'membrane_otsu_mask.png'))

    # ROI polygons
    print("  Loading ROI polygons...")
    try:
        rois = load_roi_polygons(roi_zip)
    except Exception as e:
        print(f"  ERROR loading ROI zip: {e}")
        return None

    if not rois:
        print("  No valid ROIs found — skipping sample.")
        return None

    print(f"  {len(rois)} ROI(s):")
    for roi_id, roi in rois.items():
        pct = roi['area_pixels'] / embryo_area * 100
        print(f"    {roi_id}: {roi['area_pixels']:.0f} px²  ({pct:.2f} % of embryo)")

    # Trajectory assignments from pkl
    print("  Loading trajectory pkl...")
    try:
        pkl_data = load_pkl(pkl_path)
    except Exception as e:
        print(f"  ERROR loading pkl: {e}")
        return None

    roi_assignments = pkl_data.get('roi_assignments', {})
    n_total = sum(len(v) for v in roi_assignments.values())

    if n_total < MIN_TRAJECTORIES:
        print(f"  Only {n_total} trajectories — skipping sample.")
        return None

    print(f"  Total trajectories: {n_total}")

    # Trajectory positions for overlay (µm → pixels)
    traj_px = get_traj_pixel_positions(pkl_data)

    # Per-ROI enrichment stats
    per_roi_results = []
    for roi_id, roi in rois.items():
        n_obs = len(roi_assignments.get(roi_id, []))
        stats = enrichment_stats(n_obs, n_total, roi['area_pixels'], embryo_area)
        stats['roi_id'] = roi_id
        per_roi_results.append(stats)

        direction = 'enriched' if stats['enrichment_ratio'] >= 1 else 'depleted'
        sig       = '(*)' if stats['significant'] else '(n.s.)'
        print(f"  {roi_id}: obs={n_obs}  exp={stats['expected_count']:.1f}  "
              f"ratio={stats['enrichment_ratio']:.3f}  {direction} {sig}  "
              f"p={stats['p_value']:.4f}")

    # Pooled across all ROIs in this sample
    total_roi_area  = sum(roi['area_pixels'] for roi in rois.values())
    n_obs_pooled    = sum(len(roi_assignments.get(r, [])) for r in rois)
    pooled          = enrichment_stats(n_obs_pooled, n_total,
                                       total_roi_area, embryo_area)

    direction = 'enriched' if pooled['enrichment_ratio'] >= 1 else 'depleted'
    sig       = '(*)' if pooled['significant'] else '(n.s.)'
    print(f"\n  POOLED: obs={n_obs_pooled}  exp={pooled['expected_count']:.1f}  "
          f"ratio={pooled['enrichment_ratio']:.3f}  {direction} {sig}  "
          f"p={pooled['p_value']:.4f}")

    # Plots
    print("\n  Generating plots...")
    plot_overlay(image, mask, rois, traj_px,
                 os.path.join(output_dir, 'enrichment_overlay.png'))
    plot_observed_vs_expected(per_roi_results,
                              os.path.join(output_dir, 'observed_vs_expected_barplot.png'))
    plot_enrichment_ratios(per_roi_results,
                           os.path.join(output_dir, 'enrichment_ratio_plot.png'))
    plot_pooled_bar(pooled, sample_id,
                    os.path.join(output_dir, 'pooled_bar.png'))
    save_sample_csv(per_roi_results, pooled, sample_id,
                    os.path.join(output_dir, 'enrichment_results.csv'))

    print(f"\n  Output folder: {output_dir}")
    return {'per_roi': per_roi_results, 'pooled': pooled}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("ROI Trajectory Enrichment Analyzer")
    print("====================================\n")

    folder = input("Enter path to data folder: ").strip()
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    samples = find_sample_files(folder)
    if not samples:
        print("No complete sample sets found. Exiting.")
        sys.exit(1)

    print(f"\n{len(samples)} sample(s) to process.\n")

    all_results = {}
    for sample_id, membrane_tif, roi_zip, pkl_path in samples:
        result = process_sample(sample_id, membrane_tif, roi_zip, pkl_path, folder)
        if result is not None:
            all_results[sample_id] = result

    if not all_results:
        print("\nNo samples processed successfully.")
        sys.exit(1)

    # Cross-sample summary
    print(f"\n{'='*60}")
    print("Cross-sample summary")
    print(f"{'='*60}")

    summary_dir = os.path.join(
        folder,
        f'enrichment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(summary_dir, exist_ok=True)

    grand_result = plot_cross_sample(all_results, summary_dir)
    save_summary_csv(all_results, grand_result,
                     os.path.join(summary_dir, 'summary_table.csv'))

    print(f"\nCross-sample summary saved to: {summary_dir}")
    print("\nDone.")


if __name__ == '__main__':
    main()

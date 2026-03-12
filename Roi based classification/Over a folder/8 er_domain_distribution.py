#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8 er_domain_distribution.py

Automatically characterise the spatial distribution of ER domains from
binarised fluorescence images.

Given a folder of paired images:
  *_membrane_.tif    — raw 16-bit fluorescence (used for display only)
  *_membrane_-1.tif  — binarised mask (0 = background, >0 = ER)

the script:

  1. Loads every binarised mask it finds.
  2. Labels connected components (individual ER domains).
  3. Extracts per-domain morphological features:
       area, perimeter, circularity, aspect ratio, solidity,
       equivalent diameter, centroid.
  4. Classifies each domain as  **tubule**  (elongated / branched) or
     **sheet / cisterna**  (compact).
  5. Saves figures:
       domain_overlay_<id>.png      — colour-coded domain map on raw image
       area_distribution.png        — area histograms (log scale)
       shape_scatter.png            — circularity vs. area scatter
       aspect_ratio_distribution.png
       type_fractions.png           — tubule / sheet fractions
  6. Saves summary_domains.csv      — one row per domain.
  7. Saves summary_per_embryo.csv   — aggregated statistics per embryo.

Usage:
  python "8 er_domain_distribution.py"
  (then enter the folder path when prompted, or set FOLDER below)

Dependencies:
  numpy, scipy, matplotlib  (standard scientific Python stack)

Optional:
  tifffile  — faster / more reliable TIF reading; falls back to a built-in
               pure-Python reader if not available.

Parameters at the top of the script can be adjusted:
  PX_TO_UM         : pixel size in µm (default 0.094)
  MIN_DOMAIN_PX    : discard domains smaller than this (noise removal)
  TUBULE_AR_THRESH : aspect-ratio cut-off for tubule classification
  TUBULE_CIRC_THRESH : circularity cut-off for tubule classification
"""

import os
import sys
import glob
import struct
import zlib
import math
import csv
from datetime import datetime

# ── Optional fast TIF reader ─────────────────────────────────────────────────
try:
    import tifffile as _tifffile
    def _load_tif(path):
        return _tifffile.imread(path)
    _TIF_BACKEND = 'tifffile'
except ImportError:
    _tifffile = None
    _TIF_BACKEND = 'builtin'

import numpy as np
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
PX_TO_UM           = 0.094        # µm per pixel (Nikon calibration)
MIN_DOMAIN_PX      = 10           # minimum domain size in pixels (noise filter)
TUBULE_AR_THRESH   = 3.0          # aspect ratio ≥ this → classify as tubule
TUBULE_CIRC_THRESH = 0.15         # circularity < this → also classify as tubule

# For overlay image: colour small domains grey, tubules red, sheets blue
COLOR_TUBULE  = np.array([0.95, 0.25, 0.10])   # red
COLOR_SHEET   = np.array([0.10, 0.45, 0.85])   # blue
COLOR_TINY    = np.array([0.65, 0.65, 0.65])   # grey (below MIN_DOMAIN_PX)

# Matching suffix conventions
BINARY_SUFFIX = '_membrane_-1.tif'
RAW_SUFFIX    = '_membrane_.tif'

# ─────────────────────────────────────────────────────────────────────────────
# Built-in TIFF reader (no external dependencies)
# Handles: uncompressed or DEFLATE-compressed, 8-bit or 16-bit, grayscale,
# single-page, little- or big-endian TIFF (covers all four project files).
# ─────────────────────────────────────────────────────────────────────────────
class _TIFFReadError(Exception):
    pass

def _read_tif_builtin(path):
    """
    Minimal pure-Python TIFF reader.
    Returns a 2-D numpy array (uint8 or uint16).
    Supports:
      - Uncompressed (compression=1)
      - DEFLATE/zlib (compression=8 or 32946)
      - LZW (compression=5) — partial: raises if encountered
    """
    with open(path, 'rb') as f:
        data = f.read()

    bo = data[:2]
    if bo == b'II':
        endian = '<'
    elif bo == b'MM':
        endian = '>'
    else:
        raise _TIFFReadError(f"Not a TIFF: {path}")

    def u16(offset):  return struct.unpack_from(endian + 'H', data, offset)[0]
    def u32(offset):  return struct.unpack_from(endian + 'I', data, offset)[0]
    def i32(offset):  return struct.unpack_from(endian + 'i', data, offset)[0]

    magic = u16(2)
    if magic != 42:
        raise _TIFFReadError(f"Bad TIFF magic {magic}")

    ifd_offset = u32(4)

    # ── Parse first IFD ──────────────────────────────────────────────────────
    n_entries = u16(ifd_offset)
    tags = {}
    pos = ifd_offset + 2
    for _ in range(n_entries):
        tag  = u16(pos)
        typ  = u16(pos + 2)
        cnt  = u32(pos + 4)
        voff = pos + 8     # value or offset to value

        type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 8: 2, 9: 4,
                      10: 8, 11: 4, 12: 8}
        tsz = type_sizes.get(typ, 4)

        if cnt * tsz <= 4:
            raw = data[voff: voff + 4]
        else:
            off = u32(voff)
            raw = data[off: off + cnt * tsz]

        def _vals(t, r, c):
            fmt_map = {1: 'B', 2: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h',
                       9: 'i', 11: 'f', 12: 'd'}
            fmt = fmt_map.get(t)
            if fmt is None:
                return None
            nbytes = struct.calcsize(fmt)
            vals = [struct.unpack_from(endian + fmt, r, i * nbytes)[0]
                    for i in range(c)]
            return vals[0] if c == 1 else vals

        tags[tag] = _vals(typ, raw, cnt)
        pos += 12

    width       = tags.get(256)
    height      = tags.get(257)
    bits        = tags.get(258, 8)
    compression = tags.get(259, 1)
    photometric = tags.get(262, 1)
    strip_offs  = tags.get(273)
    rows_per_strip = tags.get(278, height)
    strip_bytes = tags.get(279)
    samples_pp  = tags.get(277, 1)

    if isinstance(bits, list):     bits         = bits[0]
    if isinstance(samples_pp, list): samples_pp = samples_pp[0]
    if isinstance(rows_per_strip, list): rows_per_strip = rows_per_strip[0]
    if not isinstance(strip_offs,  list): strip_offs  = [strip_offs]
    if not isinstance(strip_bytes, list): strip_bytes = [strip_bytes]

    if samples_pp != 1:
        raise _TIFFReadError(f"Multi-channel TIFF not supported: {path}")
    if compression not in (1, 8, 32946):
        raise _TIFFReadError(
            f"Unsupported compression {compression} in {path}. "
            "Install tifffile for full TIFF support.")

    # ── Decode strips ────────────────────────────────────────────────────────
    raw_rows = []
    for off, nbytes in zip(strip_offs, strip_bytes):
        strip_data = data[off: off + nbytes]
        if compression in (8, 32946):
            strip_data = zlib.decompress(strip_data)
        raw_rows.append(strip_data)

    full_raw = b''.join(raw_rows)

    dtype = np.uint8 if bits == 8 else np.uint16
    arr   = np.frombuffer(full_raw, dtype=dtype)

    # Correct endianness if needed
    if bits == 16 and endian == '>':
        arr = arr.byteswap()

    return arr.reshape(height, width)


def load_tif(path):
    """Load a TIFF and return a 2-D numpy array."""
    if _tifffile is not None:
        return _tifffile.imread(path)
    return _read_tif_builtin(path)


# ─────────────────────────────────────────────────────────────────────────────
# Morphological feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_pixels(mask):
    """Count boundary pixels of a binary mask using erosion."""
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3)))
    return np.sum(mask & ~eroded)


def _inertia_axes(label_mask):
    """
    Return (major_axis_length, minor_axis_length) in pixels using the
    second-order central moments of the labelled region.
    Equivalent to skimage.measure.regionprops major_axis_length.
    """
    ys, xs = np.where(label_mask)
    if len(xs) < 3:
        return 1.0, 1.0

    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy

    Ixx = np.sum(dx * dx) / len(xs)
    Iyy = np.sum(dy * dy) / len(ys)
    Ixy = np.sum(dx * dy) / len(xs)

    # Eigenvalues of the 2x2 inertia tensor
    trace  = Ixx + Iyy
    det    = Ixx * Iyy - Ixy ** 2
    disc   = max(0.0, (trace / 2) ** 2 - det)
    lam1   = trace / 2 + math.sqrt(disc)
    lam2   = trace / 2 - math.sqrt(disc)
    lam2   = max(lam2, 1e-9)

    # major/minor axis ~ 4*sqrt(eigenvalue)  (matches skimage convention)
    major = 4 * math.sqrt(lam1)
    minor = 4 * math.sqrt(lam2)
    return major, minor


def _convex_hull_area(label_mask):
    """
    Approximate convex-hull area via scipy.ndimage convex_hull or
    fall back to bounding-box area if unavailable.
    """
    try:
        from scipy.spatial import ConvexHull
        ys, xs = np.where(label_mask)
        pts = np.column_stack([xs, ys])
        if len(pts) < 4:
            return float(np.sum(label_mask))
        hull = ConvexHull(pts)
        return hull.volume   # 2-D ConvexHull: .volume = area
    except Exception:
        # Fallback: bounding box area
        ys, xs = np.where(label_mask)
        if len(xs) == 0:
            return 1.0
        return float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))


def extract_domain_features(binary_mask, px_to_um=PX_TO_UM):
    """
    Label connected components in *binary_mask* and compute per-domain
    morphological features.

    Parameters
    ----------
    binary_mask : 2-D bool/uint array  (>0 = ER)
    px_to_um    : float  pixel size in µm

    Returns
    -------
    list of dicts, one per labelled component, sorted by area descending.
    Each dict contains:
      label, area_px, area_um2, perimeter_px, perimeter_um,
      circularity, major_axis_px, minor_axis_px, aspect_ratio,
      solidity, equiv_diameter_um, centroid_x_px, centroid_y_px,
      domain_type   ('tubule' | 'sheet' | 'tiny')
    """
    mask = (binary_mask > 0).astype(np.uint8)
    labelled, n_labels = ndi.label(mask)
    print(f"    Found {n_labels} connected components")

    px2  = px_to_um ** 2

    domains = []
    for lbl in range(1, n_labels + 1):
        region = labelled == lbl
        area_px = int(np.sum(region))

        if area_px < MIN_DOMAIN_PX:
            # Still record as 'tiny' for completeness
            ys, xs = np.where(region)
            cx = float(xs.mean()) if len(xs) else 0.0
            cy = float(ys.mean()) if len(ys) else 0.0
            domains.append({
                'label': lbl, 'area_px': area_px,
                'area_um2': area_px * px2,
                'perimeter_px': 0, 'perimeter_um': 0,
                'circularity': 0, 'major_axis_px': 1, 'minor_axis_px': 1,
                'aspect_ratio': 1, 'solidity': 1,
                'equiv_diameter_um': math.sqrt(4 * area_px * px2 / math.pi),
                'centroid_x_px': cx, 'centroid_y_px': cy,
                'domain_type': 'tiny',
            })
            continue

        perim_px  = _boundary_pixels(region)
        perim_px  = max(perim_px, 1)
        circ      = 4 * math.pi * area_px / (perim_px ** 2)

        major, minor = _inertia_axes(region)
        minor = max(minor, 1e-3)
        ar    = major / minor

        ch_area  = _convex_hull_area(region)
        solidity = area_px / max(ch_area, 1.0)

        equiv_d_um = math.sqrt(4 * area_px * px2 / math.pi)

        ys, xs = np.where(region)
        cx = float(xs.mean())
        cy = float(ys.mean())

        # Classification
        if ar >= TUBULE_AR_THRESH or circ < TUBULE_CIRC_THRESH:
            dtype = 'tubule'
        else:
            dtype = 'sheet'

        domains.append({
            'label':            lbl,
            'area_px':          area_px,
            'area_um2':         area_px * px2,
            'perimeter_px':     perim_px,
            'perimeter_um':     perim_px * px_to_um,
            'circularity':      circ,
            'major_axis_px':    major,
            'minor_axis_px':    minor,
            'aspect_ratio':     ar,
            'solidity':         solidity,
            'equiv_diameter_um': equiv_d_um,
            'centroid_x_px':    cx,
            'centroid_y_px':    cy,
            'domain_type':      dtype,
        })

    domains.sort(key=lambda d: d['area_px'], reverse=True)
    return domains, labelled


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay(raw_img, labelled, domains, sample_id, output_path):
    """
    Colour-coded domain overlay on the raw fluorescence image.
    Tubules → red, Sheets → blue, tiny → grey.
    """
    # Normalise raw to [0,1] for display
    img_f = raw_img.astype(np.float32)
    lo, hi = np.percentile(img_f, 1), np.percentile(img_f, 99)
    img_norm = np.clip((img_f - lo) / max(hi - lo, 1), 0, 1)

    # RGB base
    rgb = np.stack([img_norm] * 3, axis=-1) * 0.6   # dim the background

    type_map = {d['label']: d['domain_type'] for d in domains}
    color_map = {'tubule': COLOR_TUBULE, 'sheet': COLOR_SHEET,
                 'tiny': COLOR_TINY}

    for lbl in range(1, labelled.max() + 1):
        dtype = type_map.get(lbl, 'tiny')
        c = color_map[dtype]
        where = labelled == lbl
        for ch in range(3):
            rgb[:, :, ch][where] = c[ch] * 0.7 + img_norm[where] * 0.3

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin='upper')
    ax.set_title(f'ER domain map — {sample_id}\n'
                 f'Red=tubule  Blue=sheet  Grey=tiny (<{MIN_DOMAIN_PX}px)',
                 fontsize=10)
    ax.axis('off')

    n_t = sum(1 for d in domains if d['domain_type'] == 'tubule')
    n_s = sum(1 for d in domains if d['domain_type'] == 'sheet')
    n_n = sum(1 for d in domains if d['domain_type'] == 'tiny')
    leg = [
        mpatches.Patch(color=COLOR_TUBULE, label=f'Tubule ({n_t})'),
        mpatches.Patch(color=COLOR_SHEET,  label=f'Sheet  ({n_s})'),
        mpatches.Patch(color=COLOR_TINY,   label=f'Tiny   ({n_n})'),
    ]
    ax.legend(handles=leg, loc='lower right', fontsize=9,
              framealpha=0.85)

    px = 1 / plt.rcParams['figure.dpi']
    scalebar_um = 10.0
    scalebar_px = scalebar_um / PX_TO_UM
    h, w = raw_img.shape
    x0, y0 = 0.05 * w, 0.95 * h
    ax.plot([x0, x0 + scalebar_px], [y0, y0], 'w-', linewidth=3)
    ax.text(x0 + scalebar_px / 2, y0 - 10, f'{scalebar_um:.0f} µm',
            color='white', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_area_distribution(all_domains_by_sample, output_path):
    """
    Overlaid log-scale area histograms, one per embryo.
    Only domains >= MIN_DOMAIN_PX are included.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_domains_by_sample), 1)))
    bins   = np.logspace(-2, 3, 50)   # 0.01 – 1000 µm²

    for ax, dtype, title in zip(
            axes,
            ['all', 'by_type'],
            ['Area distribution (all domains)', 'Area by domain type']):

        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7,
                   label='1 µm²')
        ax.set_xscale('log')
        ax.set_xlabel('Domain area (µm²)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(alpha=0.3)

    # Left: per-embryo overlay
    ax = axes[0]
    for i, (sid, doms) in enumerate(all_domains_by_sample.items()):
        areas = [d['area_um2'] for d in doms if d['domain_type'] != 'tiny']
        if areas:
            ax.hist(areas, bins=bins, alpha=0.55, color=colors[i],
                    label=sid[:20], histtype='stepfilled')
    ax.set_title('Area distribution (all domains ≥ threshold)', fontsize=10)
    ax.legend(fontsize=8)

    # Right: pooled, split by type
    ax = axes[1]
    all_tubule = []
    all_sheet  = []
    for doms in all_domains_by_sample.values():
        all_tubule += [d['area_um2'] for d in doms if d['domain_type'] == 'tubule']
        all_sheet  += [d['area_um2'] for d in doms if d['domain_type'] == 'sheet']
    if all_tubule:
        ax.hist(all_tubule, bins=bins, alpha=0.6, color=COLOR_TUBULE,
                label=f'Tubule (n={len(all_tubule)})', histtype='stepfilled')
    if all_sheet:
        ax.hist(all_sheet, bins=bins, alpha=0.6, color=COLOR_SHEET,
                label=f'Sheet (n={len(all_sheet)})', histtype='stepfilled')
    ax.set_title('Area by domain type (all embryos pooled)', fontsize=10)
    ax.legend(fontsize=9)

    plt.suptitle('ER Domain Area Distributions', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_shape_scatter(all_domains_by_sample, output_path):
    """
    Circularity vs. area scatter for all domains (colour = type).
    The vertical dashed line at TUBULE_CIRC_THRESH and horizontal at
    TUBULE_AR_THRESH illustrate the classification boundary.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale('log')

    plotted_types = set()
    for sid, doms in all_domains_by_sample.items():
        for d in doms:
            if d['domain_type'] == 'tiny':
                continue
            c = COLOR_TUBULE if d['domain_type'] == 'tubule' else COLOR_SHEET
            label = d['domain_type'] if d['domain_type'] not in plotted_types else None
            ax.scatter(d['area_um2'], d['circularity'],
                       color=c, alpha=0.45, s=15, label=label)
            plotted_types.add(d['domain_type'])

    ax.axhline(TUBULE_CIRC_THRESH, color='gray', linestyle='--', linewidth=1,
               label=f'Circ. threshold = {TUBULE_CIRC_THRESH}')
    ax.set_xlabel('Domain area (µm²)', fontsize=11)
    ax.set_ylabel('Circularity  (4π·A / P²)', fontsize=11)
    ax.set_title('Morphological fingerprint: circularity vs. area\n'
                 'Red = tubule  Blue = sheet', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_aspect_ratio(all_domains_by_sample, output_path):
    """Aspect-ratio violin/histogram split by sample."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(1, 20, 40)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_domains_by_sample), 1)))

    for i, (sid, doms) in enumerate(all_domains_by_sample.items()):
        ars = [min(d['aspect_ratio'], 20) for d in doms
               if d['domain_type'] != 'tiny']
        if ars:
            ax.hist(ars, bins=bins, alpha=0.55, color=colors[i],
                    label=sid[:20], histtype='stepfilled')

    ax.axvline(TUBULE_AR_THRESH, color='black', linestyle='--', linewidth=1.5,
               label=f'Tubule cut-off (AR ≥ {TUBULE_AR_THRESH})')
    ax.set_xlabel('Aspect ratio (major / minor axis)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Aspect ratio distribution (capped at 20)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_type_fractions(all_domains_by_sample, output_path):
    """
    Grouped bar chart: % tubules vs sheets per embryo
    (by domain count and by total area).
    """
    samples = list(all_domains_by_sample.keys())
    n = len(samples)
    x = np.arange(n)
    w = 0.35

    frac_count_t = []
    frac_area_t  = []

    for sid in samples:
        doms = [d for d in all_domains_by_sample[sid]
                if d['domain_type'] != 'tiny']
        if not doms:
            frac_count_t.append(0.0)
            frac_area_t.append(0.0)
            continue
        n_t = sum(1 for d in doms if d['domain_type'] == 'tubule')
        a_t = sum(d['area_um2'] for d in doms if d['domain_type'] == 'tubule')
        a_all = sum(d['area_um2'] for d in doms)
        frac_count_t.append(100.0 * n_t / len(doms))
        frac_area_t.append(100.0 * a_t / max(a_all, 1e-9))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, fracs, title in zip(axes,
                                 [frac_count_t, frac_area_t],
                                 ['% tubules by domain count',
                                  '% tubule area of total ER area']):
        bars_t = ax.bar(x - w / 2, fracs, w,
                        color=COLOR_TUBULE, alpha=0.8, label='Tubule %')
        bars_s = ax.bar(x + w / 2, [100 - f for f in fracs], w,
                        color=COLOR_SHEET, alpha=0.8, label='Sheet %')
        ax.set_xticks(x)
        ax.set_xticklabels(samples, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 110)
        ax.axhline(50, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars_t:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Tubule vs. Sheet fractions', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_domain_csv(all_domains_by_sample, output_path):
    fieldnames = [
        'sample_id', 'label', 'domain_type',
        'area_px', 'area_um2', 'equiv_diameter_um',
        'perimeter_um', 'circularity',
        'aspect_ratio', 'solidity',
        'centroid_x_px', 'centroid_y_px',
    ]
    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid, doms in all_domains_by_sample.items():
            for d in doms:
                w.writerow({
                    'sample_id':         sid,
                    'label':             d['label'],
                    'domain_type':       d['domain_type'],
                    'area_px':           d['area_px'],
                    'area_um2':          round(d['area_um2'], 4),
                    'equiv_diameter_um': round(d['equiv_diameter_um'], 4),
                    'perimeter_um':      round(d['perimeter_um'], 4),
                    'circularity':       round(d['circularity'], 4),
                    'aspect_ratio':      round(d['aspect_ratio'], 3),
                    'solidity':          round(d['solidity'], 4),
                    'centroid_x_px':     round(d['centroid_x_px'], 1),
                    'centroid_y_px':     round(d['centroid_y_px'], 1),
                })
    print(f"  Saved: {os.path.basename(output_path)}")


def save_per_embryo_csv(all_domains_by_sample, output_path):
    fieldnames = [
        'sample_id',
        'n_domains_total', 'n_tubules', 'n_sheets', 'n_tiny',
        'pct_tubule_by_count', 'pct_tubule_by_area',
        'total_er_area_um2', 'tubule_area_um2', 'sheet_area_um2',
        'er_coverage_fraction',
        'median_area_um2', 'mean_area_um2',
        'median_circularity', 'median_aspect_ratio',
        'image_area_um2',
    ]
    image_area_um2 = (998 * PX_TO_UM) ** 2   # assumes 998×998 px

    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid, doms in all_domains_by_sample.items():
            useful = [d for d in doms if d['domain_type'] != 'tiny']
            tubules = [d for d in useful if d['domain_type'] == 'tubule']
            sheets  = [d for d in useful if d['domain_type'] == 'sheet']
            tiny    = [d for d in doms   if d['domain_type'] == 'tiny']

            total_er_area = sum(d['area_um2'] for d in useful)
            tub_area      = sum(d['area_um2'] for d in tubules)
            sh_area       = sum(d['area_um2'] for d in sheets)
            areas = [d['area_um2'] for d in useful]
            circs = [d['circularity'] for d in useful]
            ars   = [d['aspect_ratio'] for d in useful]

            def _med(lst): return float(np.median(lst)) if lst else float('nan')
            def _mn(lst):  return float(np.mean(lst))   if lst else float('nan')

            n_all = len(useful)
            w.writerow({
                'sample_id':               sid,
                'n_domains_total':         n_all,
                'n_tubules':               len(tubules),
                'n_sheets':                len(sheets),
                'n_tiny':                  len(tiny),
                'pct_tubule_by_count':     round(100 * len(tubules) / max(n_all, 1), 2),
                'pct_tubule_by_area':      round(100 * tub_area / max(total_er_area, 1e-9), 2),
                'total_er_area_um2':       round(total_er_area, 2),
                'tubule_area_um2':         round(tub_area, 2),
                'sheet_area_um2':          round(sh_area, 2),
                'er_coverage_fraction':    round(total_er_area / image_area_um2, 4),
                'median_area_um2':         round(_med(areas), 4),
                'mean_area_um2':           round(_mn(areas), 4),
                'median_circularity':      round(_med(circs), 4),
                'median_aspect_ratio':     round(_med(ars), 3),
                'image_area_um2':          round(image_area_um2, 2),
            })
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_image_pairs(folder):
    """
    Find all binarised TIFs (*_membrane_-1.tif) and match to raw (*_membrane_.tif).
    Returns list of (sample_id, bin_path, raw_path_or_None).
    """
    bin_files = sorted(glob.glob(os.path.join(folder, f'*{BINARY_SUFFIX}')))
    if not bin_files:
        # Also try underscore-only naming
        bin_files = sorted(glob.glob(os.path.join(folder, '*.tif')))
        bin_files = [f for f in bin_files if '-1.tif' in f]

    pairs = []
    for bp in bin_files:
        sample_id = os.path.basename(bp).replace(BINARY_SUFFIX, '')
        raw_path  = os.path.join(folder,
                                  sample_id + RAW_SUFFIX)
        if not os.path.exists(raw_path):
            raw_path = None
        pairs.append((sample_id, bp, raw_path))
        raw_str = os.path.basename(raw_path) if raw_path else '(not found)'
        print(f"  [{sample_id}]  binary: {os.path.basename(bp)}"
              f"   raw: {raw_str}")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(folder):
    print(f"\nTIF backend : {_TIF_BACKEND}")
    print(f"Pixel size  : {PX_TO_UM} µm/px")
    print(f"Min domain  : {MIN_DOMAIN_PX} px")
    print(f"Classification: AR ≥ {TUBULE_AR_THRESH}  OR  circ < {TUBULE_CIRC_THRESH} → tubule\n")

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(folder, f'er_domain_results_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("Searching for image pairs...")
    pairs = find_image_pairs(folder)
    if not pairs:
        print("No binarised TIFs found (*_membrane_-1.tif). Exiting.")
        sys.exit(1)

    all_domains_by_sample = {}

    for sample_id, bin_path, raw_path in pairs:
        print(f"\n{'='*60}")
        print(f"  {sample_id}")
        print(f"{'='*60}")

        # ── Load binarised mask ──────────────────────────────────────────────
        print(f"  Loading binary mask: {os.path.basename(bin_path)}")
        try:
            bin_img = load_tif(bin_path)
        except Exception as e:
            print(f"  ERROR loading binary: {e}"); continue

        print(f"  Shape: {bin_img.shape}, dtype: {bin_img.dtype}, "
              f"max: {bin_img.max()}")

        # Ensure it is really binary (>0 = ER)
        bin_mask = (bin_img > 0).astype(np.uint8)
        coverage = 100 * bin_mask.mean()
        print(f"  ER coverage: {coverage:.1f}%")

        # ── Load raw for display ─────────────────────────────────────────────
        raw_img = None
        if raw_path:
            print(f"  Loading raw image : {os.path.basename(raw_path)}")
            try:
                raw_img = load_tif(raw_path).astype(np.float32)
            except Exception as e:
                print(f"  Warning: could not load raw image: {e}")

        if raw_img is None:
            raw_img = bin_mask.astype(np.float32)

        # ── Domain analysis ──────────────────────────────────────────────────
        print("  Extracting domain features...")
        domains, labelled = extract_domain_features(bin_mask)

        n_useful = sum(1 for d in domains if d['domain_type'] != 'tiny')
        n_tub    = sum(1 for d in domains if d['domain_type'] == 'tubule')
        n_sh     = sum(1 for d in domains if d['domain_type'] == 'sheet')
        n_tiny   = sum(1 for d in domains if d['domain_type'] == 'tiny')

        print(f"  Domains ≥ {MIN_DOMAIN_PX}px : {n_useful}  "
              f"(tubule={n_tub}, sheet={n_sh})  |  tiny={n_tiny}")

        all_domains_by_sample[sample_id] = domains

        # ── Per-sample overlay ───────────────────────────────────────────────
        overlay_path = os.path.join(output_dir,
                                     f'domain_overlay_{sample_id}.png')
        make_overlay(raw_img, labelled, domains, sample_id, overlay_path)

    if not all_domains_by_sample:
        print("\nNo samples processed successfully.")
        return

    # ── Cross-sample figures ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross-sample summary plots")
    print(f"{'='*60}")

    plot_area_distribution(
        all_domains_by_sample,
        os.path.join(output_dir, 'area_distribution.png'))

    plot_shape_scatter(
        all_domains_by_sample,
        os.path.join(output_dir, 'shape_scatter.png'))

    plot_aspect_ratio(
        all_domains_by_sample,
        os.path.join(output_dir, 'aspect_ratio_distribution.png'))

    plot_type_fractions(
        all_domains_by_sample,
        os.path.join(output_dir, 'type_fractions.png'))

    # ── CSV exports ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving CSVs")
    print(f"{'='*60}")

    save_domain_csv(
        all_domains_by_sample,
        os.path.join(output_dir, 'summary_domains.csv'))

    save_per_embryo_csv(
        all_domains_by_sample,
        os.path.join(output_dir, 'summary_per_embryo.csv'))

    # ── Console summary table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Sample':<30} {'Domains':>8} {'Tubule%':>9} {'Area%Tub':>10} "
          f"{'ER cover%':>10}")
    print('-' * 70)
    image_area_um2 = (998 * PX_TO_UM) ** 2
    for sid, doms in all_domains_by_sample.items():
        useful = [d for d in doms if d['domain_type'] != 'tiny']
        n_t    = sum(1 for d in useful if d['domain_type'] == 'tubule')
        a_t    = sum(d['area_um2'] for d in useful if d['domain_type'] == 'tubule')
        a_all  = sum(d['area_um2'] for d in useful)
        pct_count = 100 * n_t / max(len(useful), 1)
        pct_area  = 100 * a_t / max(a_all, 1e-9)
        coverage  = 100 * a_all / image_area_um2
        print(f"{sid:<30} {len(useful):>8} {pct_count:>9.1f} {pct_area:>10.1f} "
              f"{coverage:>10.2f}")

    print(f"\nAll outputs saved to:\n  {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("ER Domain Distribution Analyser")
    print("=================================\n")

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = input(
            "Enter path to folder containing *_membrane_.tif and "
            "*_membrane_-1.tif files\n"
            "(press Enter to use current directory): "
        ).strip()
        if not folder:
            folder = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)

    process_folder(folder)


if __name__ == '__main__':
    main()

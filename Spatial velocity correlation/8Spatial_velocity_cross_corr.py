#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8Spatial_velocity_cross_corr.py

Spatial velocity cross-correlation — estimating ER domain size from
correlated instantaneous motion of co-tracked GEM particles.

═══════════════════════════════════════════════════════════════════════════════
CONCEPT
═══════════════════════════════════════════════════════════════════════════════

For every pair of GEMs (i, j) that are simultaneously tracked, compute:

    Cv(r) = < v̂_i(t) · v̂_j(t) >    averaged over all pairs
                                       with |x_i(t) - x_j(t)| ≈ r

where v̂ = v / |v| is the unit velocity vector (forward finite difference).

Interpretation:
  Cv(r) > 0  →  nearby particles tend to move in the SAME direction
  Cv(r) = 0  →  velocities are uncorrelated (purely random, no shared domain)
  Cv(r) < 0  →  nearby particles move in OPPOSITE directions

If GEMs sitting on the same ER domain are driven by the same membrane
fluctuations or structural constraints, they will share directional bias:
Cv(r) remains positive until r exceeds the domain boundary, then drops
toward zero. The zero-crossing of Cv(r) estimates the ER domain size.

Why this is INDEPENDENT of the Moran's I D-value correlogram:
  - Moran's I: time-averaged diffusion coefficient D per trajectory (seconds)
  - Cv(r):     instantaneous velocity direction at each frame (milliseconds)
  - Different physical quantity, different timescale, same underlying structure

═══════════════════════════════════════════════════════════════════════════════
DRIFT CORRECTION
═══════════════════════════════════════════════════════════════════════════════

Systematic cell drift or flow will cause Cv(r) > 0 at ALL separations,
shifting the baseline away from zero. To correct this, at each frame the
mean velocity of all particles is subtracted before computing pair
correlations. This removes the common-mode component (drift/flow) and
isolates the spatially-structured component (domain-driven correlation).

  ṽ_i(t) = v_i(t) − <v(t)>_all_particles

The corrected Cv(r) is then computed from ṽ values.

═══════════════════════════════════════════════════════════════════════════════
DATA REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

Trajectories must have either:
  - traj['frame']  — list of absolute frame indices (integer)  [preferred]
  - traj['time']   — list of timestamps in seconds             [converted]

Without one of these, co-temporal matching is impossible and the script
will skip the sample with a warning.

Input: analyzed_*.pkl files (standard pipeline output, one per embryo/movie)

═══════════════════════════════════════════════════════════════════════════════
ENSEMBLE APPROACH
═══════════════════════════════════════════════════════════════════════════════

Mirrors the approach in 7 domain_size_correlogram.py (Moran's I) and the
6Ensemble_velocity_autocorr.py scripts:

  1. Compute Cv(r) per sample — one curve per file/embryo.
  2. Weighted ensemble mean: weight = n_pairs at each r bin (Kish ESS for SE).
  3. Bootstrap CI: resample FILES (not pairs) to propagate between-embryo
     variance into the CI on the ensemble curve and on the zero-crossing.
  4. Per-sample zero-crossing → one domain-size estimate per embryo →
     Mann-Whitney U + Cliff's delta for multi-condition comparison.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Written to  spatial_Cv_<timestamp>/

  spatial_Cv_curves.png         ensemble curves + CI + per-sample overlay
  per_sample_domain_size.png    per-sample zero-crossing (domain size) by cond.
  bootstrap_distribution.png    bootstrap distribution of ensemble domain size
  spatial_map.png               spatial heatmap of Cv(r) for largest sample
  ensemble_data.csv             r, Cv, SE, CI_lo, CI_hi per condition
  per_sample_summary.csv        per-sample zero-crossing, n_pairs, condition
  pairwise_statistics.csv       Mann-Whitney p, Cliff's delta (if 2+ cond.)

Usage:
  python "8Spatial_velocity_cross_corr.py"
"""

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
from datetime import datetime

# ============================================================
# Parameters — match to your data / Moran's I settings
# ============================================================
DT              = 0.1    # seconds per frame
R_MIN_UM        = 0.5    # µm — minimum separation (start of sweep)
R_MAX_UM        = 20.0   # µm — maximum separation (end of sweep)
R_STEP_UM       = 0.5    # µm — bin width (= ring width, same as Moran's I)
MIN_PAIRS       = 20     # minimum pair-frame observations per bin
MIN_LENGTH      = 5      # minimum trajectory length (frames)
MIN_TRAJS       = 5      # minimum trajectories per sample for inclusion
N_BOOTSTRAP     = 500    # bootstrap replicates for ensemble CI
ALPHA           = 0.05   # significance level
CONFIDENCE      = 0.95   # CI level
RANDOM_SEED     = 42
FIGURE_DPI      = 300

# ER domain reference range — shown on all plots for comparison with Moran's I
ER_DOMAIN_MIN_UM = 5.0
ER_DOMAIN_MAX_UM = 7.0
# ============================================================


# ─────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────

def load_pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def find_sample_files(folder):
    paths = sorted(glob.glob(os.path.join(folder, 'analyzed_*.pkl')))
    if not paths:
        print(f"  [WARN] No analyzed_*.pkl found in {folder}")
        return []
    return [(os.path.splitext(os.path.basename(p))[0].replace('analyzed_', ''), p)
            for p in paths]


def extract_trajectories(data):
    if data is None:
        return []
    if 'trajectories' in data:
        return data['trajectories']
    for key in data:
        val = data[key]
        if isinstance(val, list) and val and isinstance(val[0], dict) and 'x' in val[0]:
            return val
    return []


# ─────────────────────────────────────────────────────────────
# Build frame index: frame → list of particle records
# ─────────────────────────────────────────────────────────────

def build_velocity_index(trajectories, dt=DT, min_length=MIN_LENGTH):
    """
    Returns a dict:  absolute_frame → list of (x, y, vx, vy)

    Velocity at frame t is the forward finite difference:
        v(t) = [x(t+1) - x(t)] / dt

    so it is defined at every frame except the last frame of each trajectory.

    Frame matching requires absolute frame numbers from either
    traj['frame'] or traj['time'] / dt.  Returns None (with warning)
    if neither field is present in any trajectory.
    """
    frame_index = {}   # frame_int -> list of [x, y, vx, vy]
    has_frame_info = False

    for traj_idx, traj in enumerate(trajectories):
        x_arr = np.asarray(traj.get('x', []), dtype=float)
        y_arr = np.asarray(traj.get('y', []), dtype=float)
        n = len(x_arr)

        if n < max(min_length, 2):
            continue

        # --- Determine absolute frame numbers ---
        if 'frame' in traj and len(traj['frame']) == n:
            frames = np.asarray(traj['frame'], dtype=int)
            has_frame_info = True
        elif 'time' in traj and len(traj['time']) == n:
            frames = np.round(np.asarray(traj['time'], dtype=float) / dt).astype(int)
            has_frame_info = True
        else:
            # No absolute time info: skip (cannot do cross-trajectory matching)
            continue

        # Forward difference velocities defined at frames[0] … frames[n-2]
        vx = np.diff(x_arr) / dt
        vy = np.diff(y_arr) / dt

        for i in range(n - 1):
            f = int(frames[i])
            if f not in frame_index:
                frame_index[f] = []
            frame_index[f].append([float(x_arr[i]), float(y_arr[i]),
                                    float(vx[i]),    float(vy[i])])

    if not has_frame_info:
        return None   # caller will warn

    # Convert to numpy arrays
    return {f: np.array(pts) for f, pts in frame_index.items() if len(pts) >= 2}


# ─────────────────────────────────────────────────────────────
# Core: per-sample Cv(r) calculation
# ─────────────────────────────────────────────────────────────

def compute_sample_Cv_r(trajectories, dt=DT, r_min=R_MIN_UM, r_max=R_MAX_UM,
                         r_step=R_STEP_UM, min_pairs=MIN_PAIRS,
                         min_length=MIN_LENGTH):
    """
    Compute the spatial velocity cross-correlation Cv(r) for one sample.

    Steps:
      1. Build frame-indexed velocity data (requires absolute frame numbers).
      2. For each frame with >= 2 particles:
           a. Subtract mean velocity (drift correction).
           b. Normalise residual velocities to unit vectors.
           c. Compute pairwise dot products and separation distances.
           d. Accumulate into r bins.
      3. Return mean Cv per bin, n_pairs per bin, and zero-crossing.

    Returns None if no absolute frame information is available.
    """
    frame_index = build_velocity_index(trajectories, dt, min_length)

    if frame_index is None:
        return None   # no frame info

    # Bin edges and centres
    r_edges   = np.arange(r_min, r_max + r_step * 0.5, r_step)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins    = len(r_centers)

    pools   = [[] for _ in range(n_bins)]   # one list of Cv values per bin

    for f, pts in frame_index.items():
        # pts: (n_part, 4)  columns: x, y, vx, vy
        n_part = len(pts)
        if n_part < 2:
            continue

        positions = pts[:, :2]     # (n, 2)
        velocities = pts[:, 2:4]   # (n, 2)

        # ── Drift correction: subtract frame-mean velocity ──────────────
        mean_v = velocities.mean(axis=0)       # (2,)
        v_res  = velocities - mean_v           # residual velocities

        # ── Normalise to unit vectors ────────────────────────────────────
        v_mag  = np.linalg.norm(v_res, axis=1)  # (n,)
        valid  = v_mag > 0
        if valid.sum() < 2:
            continue

        v_hat        = np.zeros_like(v_res)
        v_hat[valid] = v_res[valid] / v_mag[valid, np.newaxis]

        # ── Pairwise computation (upper triangle, vectorised) ─────────────
        # Cv dot-product matrix
        Cv_mat = v_hat @ v_hat.T                           # (n, n)

        # Pairwise separation distances
        diff   = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r_mat  = np.sqrt((diff ** 2).sum(axis=2))          # (n, n)

        # Upper triangle indices
        rows, cols = np.triu_indices(n_part, k=1)
        r_vals  = r_mat[rows, cols]
        cv_vals = np.clip(Cv_mat[rows, cols], -1.0, 1.0)

        # Only use pairs where both residual velocities are non-zero
        both_valid = valid[rows] & valid[cols]
        r_vals  = r_vals[both_valid]
        cv_vals = cv_vals[both_valid]

        # Bin assignment
        bin_idx = np.floor((r_vals - r_min) / r_step).astype(int)
        in_range = (bin_idx >= 0) & (bin_idx < n_bins)
        for k in np.where(in_range)[0]:
            pools[bin_idx[k]].append(cv_vals[k])

    # ── Aggregate bins ────────────────────────────────────────────────────
    Cv      = np.full(n_bins, np.nan)
    Cv_se   = np.full(n_bins, np.nan)
    n_pairs = np.zeros(n_bins, dtype=np.int64)

    for k, pool in enumerate(pools):
        if len(pool) >= min_pairs:
            arr       = np.asarray(pool)
            Cv[k]     = float(np.mean(arr))
            Cv_se[k]  = float(np.std(arr) / np.sqrt(len(arr)))
            n_pairs[k] = len(arr)

    # ── Zero-crossing (domain size estimate) ─────────────────────────────
    zero_crossing = _find_zero_crossing(Cv, r_centers)

    return {
        'Cv':           Cv,
        'Cv_se':        Cv_se,
        'n_pairs':      n_pairs,
        'r_centers':    r_centers,
        'zero_crossing': zero_crossing,
        'n_frames_used': len(frame_index),
        'n_trajs':       len([t for t in trajectories
                              if len(t.get('x', [])) >= min_length]),
    }


# ─────────────────────────────────────────────────────────────
# Summary statistic helper
# ─────────────────────────────────────────────────────────────

def _find_zero_crossing(Cv, r_centers):
    """
    Return first separation r where Cv crosses from positive to <= 0,
    linearly interpolated.  This is the domain size estimate.
    """
    valid = ~np.isnan(Cv)
    if valid.sum() < 2:
        return np.nan
    cv = Cv[valid]
    rc = r_centers[valid]
    for i in range(len(cv) - 1):
        if cv[i] > 0 >= cv[i + 1]:
            frac = cv[i] / (cv[i] - cv[i + 1])
            return float(rc[i] + frac * (rc[i + 1] - rc[i]))
    return np.nan   # never crosses zero in the sweep range


# ─────────────────────────────────────────────────────────────
# Ensemble aggregation
# ─────────────────────────────────────────────────────────────

def compute_ensemble(per_sample, r_centers):
    """
    n_pairs-weighted ensemble mean Cv(r) with Kish ESS-based SE.

    Weights w_s(r) = n_pairs_s(r): bins with more pair-frame observations
    are weighted proportionally more — exactly as in the Moran's I ensemble.
    """
    n_s      = len(per_sample)
    n_bins   = len(r_centers)
    Cv_mat   = np.array([r['Cv']      for r in per_sample], dtype=float)
    np_mat   = np.array([r['n_pairs'] for r in per_sample], dtype=float)

    ens_Cv = np.full(n_bins, np.nan)
    ens_se = np.full(n_bins, np.nan)

    for k in range(n_bins):
        cv  = Cv_mat[:, k]
        w   = np_mat[:, k]
        ok  = ~np.isnan(cv) & (w > 0)
        if ok.sum() < 2:
            continue
        cv_ok = cv[ok];  w_ok = w[ok]
        w_sum = w_ok.sum()
        mean_k = float(np.dot(w_ok, cv_ok) / w_sum)
        ens_Cv[k] = mean_k
        n_eff = w_sum ** 2 / np.dot(w_ok, w_ok)
        if n_eff > 1:
            wvar = float(np.dot(w_ok, (cv_ok - mean_k) ** 2) / w_sum)
            ens_se[k] = float(np.sqrt(wvar / n_eff))

    return ens_Cv, ens_se


# ─────────────────────────────────────────────────────────────
# Bootstrap CI (sample-level)
# ─────────────────────────────────────────────────────────────

def bootstrap_ensemble(per_sample, r_centers, n_bootstrap=N_BOOTSTRAP,
                       rng=None):
    """
    Resample FILES (not pairs) to propagate between-embryo variance into:
      - CI on the ensemble Cv(r) curve at each bin
      - CI on the zero-crossing (domain size)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_s    = len(per_sample)
    n_bins = len(r_centers)

    boot_Cv  = np.full((n_bootstrap, n_bins), np.nan)
    boot_zc  = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        idx      = rng.integers(0, n_s, size=n_s)
        boot_res = [per_sample[i] for i in idx]
        bc, _    = compute_ensemble(boot_res, r_centers)
        boot_Cv[b]  = bc
        boot_zc[b]  = _find_zero_crossing(bc, r_centers)

    alpha_tail = (1.0 - CONFIDENCE) / 2.0 * 100

    def pci(arr):
        a = arr[~np.isnan(arr)]
        if len(a) < 10:
            return np.nan, np.nan
        return (float(np.percentile(a, alpha_tail)),
                float(np.percentile(a, 100 - alpha_tail)))

    ci_lo = np.nanpercentile(boot_Cv, alpha_tail,      axis=0)
    ci_hi = np.nanpercentile(boot_Cv, 100-alpha_tail,  axis=0)

    return {
        'ci_lo':   ci_lo,
        'ci_hi':   ci_hi,
        'zc_ci':   pci(boot_zc),
        'boot_zc': boot_zc,
    }


# ─────────────────────────────────────────────────────────────
# Load one condition
# ─────────────────────────────────────────────────────────────

def load_condition(folder, name):
    samples    = find_sample_files(folder)
    if not samples:
        return None

    r_edges    = np.arange(R_MIN_UM, R_MAX_UM + R_STEP_UM * 0.5, R_STEP_UM)
    r_centers  = 0.5 * (r_edges[:-1] + r_edges[1:])
    per_sample = []

    for sid, path in samples:
        data  = load_pkl(path)
        trajs = extract_trajectories(data)
        res   = compute_sample_Cv_r(trajs)

        if res is None:
            print(f"  [SKIP] {sid}: no frame/time field — cannot match co-temporal particles")
            continue
        if res['n_trajs'] < MIN_TRAJS:
            print(f"  [SKIP] {sid}: only {res['n_trajs']} qualifying trajectories")
            continue

        n_valid = int(np.sum(res['n_pairs'] >= MIN_PAIRS))
        if n_valid < 3:
            print(f"  [SKIP] {sid}: fewer than 3 r-bins have enough pairs")
            continue

        res['sample_id'] = sid
        per_sample.append(res)
        zcstr = f"{res['zero_crossing']:.2f} µm" \
                if not np.isnan(res['zero_crossing']) else "NaN"
        print(f"  {sid}: {res['n_frames_used']} frames | "
              f"zero-crossing = {zcstr}")

    if len(per_sample) < 2:
        print(f"  [ERROR] Need >= 2 valid samples, got {len(per_sample)}")
        return None

    rng           = np.random.default_rng(RANDOM_SEED)
    ens_Cv, ens_se = compute_ensemble(per_sample, r_centers)
    boot          = bootstrap_ensemble(per_sample, r_centers, rng=rng)

    zc_vals    = np.array([r['zero_crossing'] for r in per_sample])
    ens_zc     = _find_zero_crossing(ens_Cv, r_centers)

    return {
        'name':       name,
        'folder':     folder,
        'per_sample': per_sample,
        'r_centers':  r_centers,
        'ens_Cv':     ens_Cv,
        'ens_se':     ens_se,
        'boot':       boot,
        'ens_zc':     ens_zc,
        'zc_vals':    zc_vals,
        'n_samples':  len(per_sample),
    }


# ─────────────────────────────────────────────────────────────
# Statistics between conditions
# ─────────────────────────────────────────────────────────────

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (len(a) * len(b)))

def interpret_cliffs(d):
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.330: return "small"
    if ad < 0.474: return "medium"
    return "large"

def compare_conditions(conditions):
    rows = []
    for i, j in combinations(range(len(conditions)), 2):
        cA, cB = conditions[i], conditions[j]
        a = cA['zc_vals'][~np.isnan(cA['zc_vals'])]
        b = cB['zc_vals'][~np.isnan(cB['zc_vals'])]
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        d    = cliffs_delta(a, b)
        rows.append({
            'condition_A':    cA['name'],
            'condition_B':    cB['name'],
            'n_A':            len(a),
            'n_B':            len(b),
            'median_A_um':    float(np.median(a)),
            'median_B_um':    float(np.median(b)),
            'mean_A_um':      float(np.mean(a)),
            'mean_B_um':      float(np.mean(b)),
            'mann_whitney_U': float(u),
            'p_value':        float(p),
            'significant':    bool(p < ALPHA),
            'cliffs_delta':   d,
            'effect_size':    interpret_cliffs(d),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

CMAP = plt.cm.tab10

def _cond_colors(n):
    return [CMAP(i / max(n - 1, 1)) for i in range(n)]


def plot_ensemble_curves(conditions, output_dir):
    """
    Ensemble Cv(r) curves with CI bands and per-sample overlay.
    ER domain reference range marked as shaded band.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = _cond_colors(len(conditions))

    # ER domain reference band
    ax.axvspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM,
               color='gold', alpha=0.18, label='ER domain reference (5–7 µm)')

    for col, cond in zip(colors, conditions):
        rc    = cond['r_centers']
        ens   = cond['ens_Cv']
        ci_lo = cond['boot']['ci_lo']
        ci_hi = cond['boot']['ci_hi']

        # Per-sample curves (faint)
        for r in cond['per_sample']:
            ax.plot(rc, r['Cv'], color=col, alpha=0.10, linewidth=0.7)

        # Ensemble mean
        ax.plot(rc, ens, color=col, linewidth=2.2,
                label=f"{cond['name']}  (n={cond['n_samples']} samples)")

        # CI band
        ax.fill_between(rc, ci_lo, ci_hi, color=col, alpha=0.18)

        # Mark zero-crossing
        if not np.isnan(cond['ens_zc']):
            ax.axvline(cond['ens_zc'], color=col, linestyle='--',
                       linewidth=1.1, alpha=0.8)
            ax.text(cond['ens_zc'], ax.get_ylim()[1] * 0.98,
                    f" {cond['ens_zc']:.1f} µm",
                    color=col, fontsize=8, va='top')

    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5, linestyle='-')

    ax.set_xlabel('Separation  r  (µm)', fontsize=12)
    ax.set_ylabel(r'Spatial velocity cross-correlation  $C_v(r)$', fontsize=12)
    ax.set_title(
        'Spatial Velocity Cross-Correlation\n'
        '(drift-corrected; shaded = 95% bootstrap CI; '
        'dashed = zero-crossing / domain size)',
        fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_Cv_curves.png'), dpi=FIGURE_DPI)
    plt.close()
    print("  Saved: spatial_Cv_curves.png")


def plot_per_sample_domain_size(conditions, stats_rows, output_dir):
    """
    Boxplot + jitter of per-sample zero-crossing (domain size) per condition.
    """
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(conditions) + 2), 6))
    colors  = _cond_colors(len(conditions))
    rng_jit = np.random.default_rng(0)

    for xi, (col, cond) in enumerate(zip(colors, conditions)):
        vals = cond['zc_vals']
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue

        ax.boxplot(vals, positions=[xi], widths=0.38,
                   patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.35),
                   medianprops=dict(color='black', linewidth=1.8),
                   whiskerprops=dict(linewidth=1.0),
                   capprops=dict(linewidth=1.0),
                   showfliers=False)
        jitter = rng_jit.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + jitter, vals, color=col, s=32, alpha=0.78,
                   zorder=5, edgecolors='white', linewidths=0.4)

    # ER domain reference band
    ax.axhspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM,
               color='gold', alpha=0.2, label='ER domain reference (5–7 µm)')

    # Significance bars
    if stats_rows:
        sig   = [r for r in stats_rows if r['significant']]
        y_top = max(
            (np.nanmax(c['zc_vals']) for c in conditions
             if np.any(~np.isnan(c['zc_vals']))),
            default=10.0
        )
        bar_step = y_top * 0.10
        names    = [c['name'] for c in conditions]
        for k, comp in enumerate(sig):
            try:
                xA = names.index(comp['condition_A'])
                xB = names.index(comp['condition_B'])
            except ValueError:
                continue
            ybar = y_top + bar_step * (k + 1)
            ax.plot([xA, xA, xB, xB],
                    [ybar - bar_step * 0.1, ybar, ybar, ybar - bar_step * 0.1],
                    color='k', linewidth=1.0)
            pval = comp['p_value']
            pstr = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            ax.text((xA + xB) / 2, ybar + bar_step * 0.05, pstr,
                    ha='center', va='bottom', fontsize=11)

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([c['name'] for c in conditions],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Domain size estimate  (µm)\n[zero-crossing of Cv(r)]', fontsize=11)
    ax.set_title('Per-sample Spatial Correlation Domain Size\n'
                 '(each point = one embryo/movie; gold band = ER reference)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_sample_domain_size.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: per_sample_domain_size.png")


def plot_bootstrap_distribution(conditions, output_dir):
    """Bootstrap distribution of ensemble zero-crossing per condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = _cond_colors(len(conditions))

    for col, cond in zip(colors, conditions):
        arr = cond['boot']['boot_zc']
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        ci  = cond['boot']['zc_ci']
        ax.hist(arr, bins=30, color=col, alpha=0.45,
                label=f"{cond['name']}  ({len(arr)}/{N_BOOTSTRAP} converged)")
        if not np.isnan(ci[0]):
            ax.axvline(ci[0], color=col, linestyle=':', linewidth=1.3)
            ax.axvline(ci[1], color=col, linestyle=':', linewidth=1.3)

    ax.axvspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM,
               color='gold', alpha=0.25, label='ER domain reference')
    ax.set_xlabel('Bootstrap ensemble zero-crossing  (µm)', fontsize=11)
    ax.set_ylabel('Bootstrap replicates', fontsize=11)
    ax.set_title('Bootstrap CI on ensemble domain-size estimate\n'
                 '(dotted = 95% CI boundaries)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_distribution.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: bootstrap_distribution.png")


def plot_n_pairs_profile(conditions, output_dir):
    """
    Diagnostic: number of co-temporal particle pairs per r bin.
    Low n_pairs bins drive up noise — important to inspect.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = _cond_colors(len(conditions))

    for col, cond in zip(colors, conditions):
        rc = cond['r_centers']
        # Sum n_pairs across all samples
        np_total = np.zeros(len(rc))
        for r in cond['per_sample']:
            np_total += r['n_pairs'].astype(float)
        ax.bar(rc, np_total, width=R_STEP_UM * 0.8,
               color=col, alpha=0.5,
               label=f"{cond['name']}  (total across all samples)")

    ax.axhline(MIN_PAIRS * cond['n_samples'], color='red', linestyle='--',
               linewidth=1.0, label=f'MIN_PAIRS × n_samples threshold')
    ax.set_xlabel('Separation  r  (µm)', fontsize=11)
    ax.set_ylabel('Total pair-frame observations', fontsize=11)
    ax.set_title('Diagnostic: pair counts per r bin\n'
                 '(bins below threshold shown as dashed in curve plots)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pair_counts_diagnostic.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: pair_counts_diagnostic.png")


def plot_individual_overlay(conditions, output_dir):
    """
    Multi-panel: one panel per condition, per-sample Cv(r) coloured by
    their individual zero-crossing (domain size).
    Reveals heterogeneity between embryos.
    """
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        rc     = cond['r_centers']
        zcvals = cond['zc_vals']
        vmin   = np.nanmin(zcvals) if not np.all(np.isnan(zcvals)) else 0
        vmax   = np.nanmax(zcvals) if not np.all(np.isnan(zcvals)) else 15
        cmap   = plt.cm.plasma_r

        for r in cond['per_sample']:
            zc  = r['zero_crossing']
            col = cmap((zc - vmin) / max(vmax - vmin, 1e-9)) \
                  if not np.isnan(zc) else 'grey'
            # Show only reliable bins
            rel = r['n_pairs'] >= MIN_PAIRS
            rc_rel = rc.copy().astype(float)
            cv_rel = r['Cv'].copy()
            cv_rel[~rel] = np.nan
            ax.plot(rc_rel, cv_rel, color=col, alpha=0.55, linewidth=0.9)

        # Ensemble overlay
        ax.plot(rc, cond['ens_Cv'], 'k-', linewidth=2.0, label='ensemble mean')
        ax.fill_between(rc, cond['boot']['ci_lo'], cond['boot']['ci_hi'],
                        color='k', alpha=0.10)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.axvspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM,
                   color='gold', alpha=0.18)
        ax.set_xlabel('Separation  r  (µm)', fontsize=11)
        ax.set_title(f"{cond['name']}  (n={cond['n_samples']})", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='zero-crossing (µm)')

    axes[0].set_ylabel(r'$C_v(r)$', fontsize=12)
    plt.suptitle('Per-sample Cv(r) coloured by domain-size estimate',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_sample_overlay.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: individual_sample_overlay.png")


# ─────────────────────────────────────────────────────────────
# CSV exports
# ─────────────────────────────────────────────────────────────

def export_ensemble_data(conditions, output_dir):
    rows = []
    for cond in conditions:
        rc    = cond['r_centers']
        ens   = cond['ens_Cv']
        se    = cond['ens_se']
        ci_lo = cond['boot']['ci_lo']
        ci_hi = cond['boot']['ci_hi']
        for k in range(len(rc)):
            rows.append({
                'condition':       cond['name'],
                'separation_um':   float(rc[k]),
                'ensemble_Cv':     float(ens[k]) if not np.isnan(ens[k]) else '',
                'ensemble_SE':     float(se[k])  if not np.isnan(se[k])  else '',
                'CI_lo_95':        float(ci_lo[k]) if not np.isnan(ci_lo[k]) else '',
                'CI_hi_95':        float(ci_hi[k]) if not np.isnan(ci_hi[k]) else '',
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'ensemble_data.csv'), index=False)
    print("  Saved: ensemble_data.csv")


def export_per_sample_summary(conditions, output_dir):
    rows = []
    for cond in conditions:
        for r in cond['per_sample']:
            rows.append({
                'condition':      cond['name'],
                'sample_id':      r['sample_id'],
                'n_trajs':        r['n_trajs'],
                'n_frames_used':  r['n_frames_used'],
                'domain_size_um': r['zero_crossing']
                                  if not np.isnan(r['zero_crossing']) else '',
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'per_sample_summary.csv'), index=False)
    print("  Saved: per_sample_summary.csv")


def export_statistics(stats_rows, output_dir):
    if not stats_rows:
        return
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(output_dir, 'pairwise_statistics.csv'), index=False)
    print("  Saved: pairwise_statistics.csv")


# ─────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────

def print_summary(conditions, stats_rows):
    print("\n" + "=" * 62)
    print("SPATIAL VELOCITY CROSS-CORRELATION — SUMMARY")
    print("=" * 62)
    print(f"ER domain reference: {ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm")

    for cond in conditions:
        print(f"\n{cond['name']}  ({cond['n_samples']} samples)")
        if not np.isnan(cond['ens_zc']):
            print(f"  Ensemble domain size (zero-crossing): {cond['ens_zc']:.2f} µm")
            zc_ci = cond['boot']['zc_ci']
            if not np.isnan(zc_ci[0]):
                print(f"  95% bootstrap CI:  [{zc_ci[0]:.2f}, {zc_ci[1]:.2f}] µm")
            in_range = ER_DOMAIN_MIN_UM <= cond['ens_zc'] <= ER_DOMAIN_MAX_UM
            print(f"  Consistent with ER domain reference: {'YES' if in_range else 'NO'}")
        else:
            print("  Zero-crossing not found in sweep range "
                  f"[{R_MIN_UM}, {R_MAX_UM}] µm")

        zc = cond['zc_vals'][~np.isnan(cond['zc_vals'])]
        if len(zc):
            print(f"  Per-sample median: {np.median(zc):.2f} µm  "
                  f"IQR=[{np.percentile(zc,25):.2f}, {np.percentile(zc,75):.2f}] µm")

    if stats_rows:
        print("\nPairwise comparisons (per-sample domain size):")
        for r in stats_rows:
            sig = "SIGNIFICANT" if r['significant'] else "n.s."
            print(f"  {r['condition_A']} vs {r['condition_B']}")
            print(f"    p={r['p_value']:.3e}  {sig}  "
                  f"Cliff's d={r['cliffs_delta']:.3f}  ({r['effect_size']})")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Spatial Velocity Cross-Correlation Analysis")
    print("=" * 46)
    print("Requires trajectories with 'frame' or 'time' fields.\n")
    print(f"r sweep: {R_MIN_UM} – {R_MAX_UM} µm  (step {R_STEP_UM} µm)")
    print(f"ER domain reference: {ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm\n")

    try:
        n = int(input("Number of conditions to analyse (minimum 1): "))
        if n < 1:
            print("Need at least 1 condition."); return
    except ValueError:
        print("Invalid input."); return

    conditions = []
    for i in range(n):
        folder = input(f"Directory for condition {i+1}: ").strip()
        if not os.path.isdir(folder):
            print(f"  Directory not found: {folder}"); return
        default_name = os.path.basename(os.path.normpath(folder)) or f"Cond_{i+1}"
        name = input(f"  Name [Enter = '{default_name}']: ").strip() or default_name
        print(f"\nLoading '{name}' ...")
        cond = load_condition(folder, name)
        if cond is None:
            print(f"  Failed to load '{name}'."); return
        conditions.append(cond)
        print(f"  → {cond['n_samples']} valid samples.")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"spatial_Cv_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")

    stats_rows = compare_conditions(conditions) if len(conditions) >= 2 else []

    print("Generating plots ...")
    plot_ensemble_curves(conditions, output_dir)
    plot_per_sample_domain_size(conditions, stats_rows, output_dir)
    plot_bootstrap_distribution(conditions, output_dir)
    plot_n_pairs_profile(conditions, output_dir)
    plot_individual_overlay(conditions, output_dir)

    print("Exporting data ...")
    export_ensemble_data(conditions, output_dir)
    export_per_sample_summary(conditions, output_dir)
    export_statistics(stats_rows, output_dir)

    print_summary(conditions, stats_rows)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

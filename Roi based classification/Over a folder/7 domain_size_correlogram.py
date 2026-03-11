#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7 domain_size_correlogram.py

Estimates the characteristic spatial domain size of diffusion coefficient (D)
organisation using a ring-based Moran's I spatial correlogram.

The core question:
  "At what spatial scale do nearby GEMs still have similar D values?"

If this scale matches the ER domain size (5–7 µm), it provides strong
ROI-independent evidence that ER structure organises GEM diffusion.

Method — Spatial Correlogram:
  1. Subsample trajectories to MAX_N_TRAJS (for speed).
  2. Compute the full n×n pairwise distance matrix once.
  3. For each lag r in a sweep [R_MIN_UM … R_MAX_UM] at step R_STEP_UM:
       - Select trajectory pairs in the ring (r − R_STEP_UM/2, r + R_STEP_UM/2]
       - Compute Moran's I using inverse-distance weights for those pairs
       - Estimate p-value by permutation test (N_PERMUTATIONS shuffles)
  4. Define the CORRELATION LENGTH as the smallest r where either:
       (a) I(r) first crosses zero  (zero-crossing, interpolated), OR
       (b) I(r) first becomes non-significant (p > P_THRESHOLD)
     whichever comes first.

If the correlation length falls in [ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM],
it is highlighted as consistent with ER-mediated confinement.

Input:
  roi_trajectory_data_{sample_id}.pkl    (from script 1)

Output per sample → {sample_id}_domain_size/:
  moran_correlogram.png     I(r) + p(r) curves, ER band, correlation length
  correlogram_data.csv      r, I, p, n_pairs, significant, reliable

Cross-sample → domain_size_summary_{timestamp}/:
  cross_sample_correlogram.png   overlay of all I(r) curves
  domain_size_comparison.png     bar chart of correlation lengths per embryo
  summary_table.csv

Usage:
  python "7 domain_size_correlogram.py"

Notes:
  - Sparse ring representation makes each lag O(n_pairs), not O(n²).
  - Rings with fewer than MIN_PAIRS pairs are shown dashed (unreliable).
  - Correlation length 'always_significant' means I(r) > 0 and p < 0.05
    across the entire sweep — consider extending R_MAX_UM.
  - Correlation length 'always_negative' means no positive autocorrelation
    was found at any scale.
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
from datetime import datetime

# ============================================================
# Parameters
# ============================================================
R_MIN_UM          = 0.5    # µm — start of lag sweep
R_MAX_UM          = 20.0   # µm — end of lag sweep
R_STEP_UM         = 0.5    # µm — lag step = ring width
N_PERMUTATIONS    = 199    # permutations per lag (low for speed; ≥499 for publication)
MIN_PAIRS         = 20     # minimum pairs in ring for reliable estimate
MAX_N_TRAJS       = 1000   # subsample limit (O(n²) distance matrix)
MIN_TRAJECTORIES  = 20     # skip sample below this count
P_THRESHOLD       = 0.05   # significance threshold
DEFAULT_PX_TO_UM  = 0.094
RANDOM_SEED       = 42

# ER domain size reference range (µm) — marked on all plots
ER_DOMAIN_MIN_UM  = 5.0
ER_DOMAIN_MAX_UM  = 7.0


# ─────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────
def find_sample_files(folder):
    pkl_files = sorted(glob.glob(os.path.join(folder, 'roi_trajectory_data_*.pkl')))
    if not pkl_files:
        print(f"No roi_trajectory_data_*.pkl files found in {folder}")
        return []
    samples = []
    for pkl_path in pkl_files:
        basename  = os.path.basename(pkl_path)
        sample_id = basename.replace('roi_trajectory_data_', '').replace('.pkl', '')
        print(f"[{sample_id}] Found")
        samples.append((sample_id, pkl_path))
    return samples


# ─────────────────────────────────────────────────────────────
# Trajectory extraction
# ─────────────────────────────────────────────────────────────
def get_all_trajectories_um(pkl_data):
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
# Moran's I — sparse ring implementation
# ─────────────────────────────────────────────────────────────
def _morans_I_sparse(D, rows, cols, weights):
    """
    Compute Moran's I given sparse pair representation.
    rows, cols : indices of pairs (upper triangle only, i < j)
    weights    : inverse-distance weights for each pair
    Symmetric weight matrix assumed: W[i,j] = W[j,i].
    """
    n     = len(D)
    W_sum = 2.0 * weights.sum()   # factor 2 for symmetric matrix
    if W_sum == 0:
        return np.nan
    D_mean = D.mean()
    D_dev  = D - D_mean
    denom  = np.sum(D_dev ** 2)
    if denom == 0:
        return np.nan
    # Symmetric contribution: each pair (i,j) contributes dev[i]*dev[j] twice
    cross = np.sum(weights * D_dev[rows] * D_dev[cols])
    numerator = n * 2.0 * cross
    return numerator / (W_sum * denom)


def compute_ring_morans(Ds, dist_matrix, r_center, half_width, n_perms, rng):
    """
    Compute ring Moran's I for pairs in (r_center - half_width, r_center + half_width].
    Uses upper triangle only (i < j) for efficiency.
    Returns dict: {I, p, n_pairs}
    """
    r_lo = r_center - half_width
    r_hi = r_center + half_width
    n    = len(Ds)

    # Upper triangle mask
    rows, cols = np.triu_indices(n, k=1)
    dists      = dist_matrix[rows, cols]
    in_ring    = (dists > r_lo) & (dists <= r_hi)

    r_rows    = rows[in_ring]
    r_cols    = cols[in_ring]
    r_dists   = dists[in_ring]
    n_pairs   = int(in_ring.sum())

    if n_pairs < MIN_PAIRS:
        return {'I': np.nan, 'p': np.nan, 'n_pairs': n_pairs}

    weights = 1.0 / np.maximum(r_dists, 1e-9)
    E_I     = -1.0 / (n - 1)

    I_obs = _morans_I_sparse(Ds, r_rows, r_cols, weights)
    if np.isnan(I_obs):
        return {'I': np.nan, 'p': np.nan, 'n_pairs': n_pairs}

    # Permutation p-value
    I_perm = np.empty(n_perms)
    for k in range(n_perms):
        I_perm[k] = _morans_I_sparse(rng.permutation(Ds), r_rows, r_cols, weights)

    I_clean = I_perm[~np.isnan(I_perm)]
    if len(I_clean) == 0:
        return {'I': I_obs, 'p': np.nan, 'n_pairs': n_pairs}

    p = float(np.mean(np.abs(I_clean - E_I) >= abs(I_obs - E_I)))
    return {'I': I_obs, 'p': p, 'n_pairs': n_pairs}


# ─────────────────────────────────────────────────────────────
# Correlation length detection
# ─────────────────────────────────────────────────────────────
def find_correlation_length(radii, Is, ps):
    """
    Returns (corr_length_um, method) where method is one of:
      'zero_crossing'      — I(r) interpolated to 0
      'non_significant'    — first radius with p > P_THRESHOLD
      'always_significant' — never lost significance within sweep
      'always_negative'    — I was never positive
      None                 — insufficient data
    """
    radii = np.asarray(radii)
    Is    = np.asarray(Is, dtype=float)
    ps    = np.asarray(ps, dtype=float)

    # Only reliable, valid points
    valid = ~np.isnan(Is) & ~np.isnan(ps)
    if valid.sum() < 2:
        return np.nan, None

    rv = radii[valid]
    Iv = Is[valid]
    pv = ps[valid]

    if np.all(Iv <= 0):
        return np.nan, 'always_negative'

    # (a) Zero crossing: first place I goes from > 0 to ≤ 0
    for i in range(len(rv) - 1):
        if Iv[i] > 0 and Iv[i + 1] <= 0:
            frac = Iv[i] / (Iv[i] - Iv[i + 1])
            return float(rv[i] + frac * (rv[i + 1] - rv[i])), 'zero_crossing'

    # (b) Non-significant: first radius with p > P_THRESHOLD
    for r, I, p in zip(rv, Iv, pv):
        if p > P_THRESHOLD:
            return float(r), 'non_significant'

    # Significant across entire sweep
    return float(rv[-1]), 'always_significant'


# ─────────────────────────────────────────────────────────────
# Plotting — per sample
# ─────────────────────────────────────────────────────────────
def plot_correlogram(radii, Is, ps, n_pairs_list, corr_length, corr_method,
                     sample_id, output_path):
    radii   = np.asarray(radii)
    Is      = np.asarray(Is, dtype=float)
    ps      = np.asarray(ps, dtype=float)
    n_pairs = np.asarray(n_pairs_list)

    valid      = ~np.isnan(Is)
    reliable   = valid & (n_pairs >= MIN_PAIRS)
    unreliable = valid & (n_pairs < MIN_PAIRS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # ── Top: Moran's I ───────────────────────────────────────
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.axvspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM, alpha=0.14, color='limegreen',
                label=f'Expected ER domain ({ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm)')

    if reliable.any():
        sig    = reliable & (ps <= P_THRESHOLD)
        nonsig = reliable & (ps >  P_THRESHOLD)
        ax1.plot(radii[reliable], Is[reliable], '-', color='steelblue',
                 linewidth=2, zorder=3)
        if sig.any():
            ax1.scatter(radii[sig], Is[sig], color='steelblue', s=35,
                        zorder=4, label='p ≤ 0.05')
        if nonsig.any():
            ax1.scatter(radii[nonsig], Is[nonsig], color='lightsteelblue',
                        s=35, zorder=4, label='p > 0.05')

    if unreliable.any():
        ax1.plot(radii[unreliable], Is[unreliable], ':', color='gray',
                 linewidth=1.5, label=f'< {MIN_PAIRS} pairs (unreliable)')

    # Annotate correlation length
    if not np.isnan(corr_length):
        I_interp = float(np.interp(corr_length,
                                   radii[reliable] if reliable.any() else [corr_length],
                                   Is[reliable]    if reliable.any() else [0]))
        method_label = {
            'zero_crossing':     'zero crossing',
            'non_significant':   'first non-significant',
            'always_significant':'end of sweep (always sig.)',
        }.get(corr_method, corr_method)
        ax1.axvline(corr_length, color='darkorange', linewidth=2, linestyle='--',
                    label=f'Correlation length = {corr_length:.1f} µm\n({method_label})')
        ax1.scatter([corr_length], [I_interp], color='darkorange', s=100,
                    zorder=6, edgecolors='black', linewidths=0.8)

    ax1.set_ylabel("Moran's I", fontsize=11)
    ax1.set_title(f"Spatial D correlogram — {sample_id}\n"
                  f"Ring-based Moran's I vs. lag distance", fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)

    # ── Bottom: p-value ──────────────────────────────────────
    ax2.axhline(P_THRESHOLD, color='gray', linestyle='--', linewidth=1,
                label=f'p = {P_THRESHOLD}')
    ax2.axvspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM, alpha=0.14, color='limegreen')

    if reliable.any():
        ax2.plot(radii[reliable], ps[reliable], 'o-', color='firebrick',
                 linewidth=1.5, markersize=4)
        sig_mask = reliable & (ps <= P_THRESHOLD)
        if sig_mask.any():
            ax2.fill_between(radii, ps, P_THRESHOLD,
                             where=sig_mask & reliable,
                             alpha=0.15, color='firebrick', interpolate=True)

    if not np.isnan(corr_length):
        ax2.axvline(corr_length, color='darkorange', linewidth=2, linestyle='--')

    ax2.set_xlabel('Lag distance r (µm)', fontsize=11)
    ax2.set_ylabel('p-value', fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def save_correlogram_csv(radii, Is, ps, n_pairs_list, sample_id, output_path):
    rows = []
    for r, I, p, np_ in zip(radii, Is, ps, n_pairs_list):
        rows.append({
            'sample_id':   sample_id,
            'lag_um':      round(float(r), 3),
            'morans_I':    round(float(I), 6) if not np.isnan(I) else np.nan,
            'p_value':     round(float(p), 4) if not np.isnan(p) else np.nan,
            'n_pairs':     int(np_),
            'reliable':    bool(np_ >= MIN_PAIRS),
            'significant': bool((not np.isnan(p)) and p <= P_THRESHOLD),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Plotting — cross-sample
# ─────────────────────────────────────────────────────────────
def plot_cross_sample_correlogram(all_results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 1)))

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM, alpha=0.12, color='limegreen',
               label=f'Expected ER domain ({ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm)')

    for i, (sample_id, res) in enumerate(all_results.items()):
        radii = np.asarray(res['radii'])
        Is    = np.asarray(res['Is'], dtype=float)
        n_pairs = np.asarray(res['n_pairs'])
        reliable = (~np.isnan(Is)) & (n_pairs >= MIN_PAIRS)
        if reliable.any():
            ax.plot(radii[reliable], Is[reliable], '-', color=colors[i],
                    linewidth=1.8, alpha=0.85, label=sample_id[:22])
        cl = res['corr_length']
        if not np.isnan(cl):
            ax.axvline(cl, color=colors[i], linewidth=1, linestyle=':',
                       alpha=0.7)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Lag distance r (µm)', fontsize=11)
    ax.set_ylabel("Moran's I", fontsize=11)
    ax.set_title("Spatial D correlogram — all embryos\n"
                 "(dotted verticals = per-embryo correlation length)", fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_domain_size_comparison(all_results, output_path):
    samples = list(all_results.keys())
    cls     = [all_results[s]['corr_length'] for s in samples]
    methods = [all_results[s]['corr_method']  for s in samples]

    # Bar colour: green if in ER range, blue otherwise, grey if NaN
    bar_colors = []
    for cl in cls:
        if np.isnan(cl):
            bar_colors.append('lightgray')
        elif ER_DOMAIN_MIN_UM <= cl <= ER_DOMAIN_MAX_UM:
            bar_colors.append('limegreen')
        else:
            bar_colors.append('steelblue')

    x = np.arange(len(samples))
    fig, ax = plt.subplots(figsize=(max(8, len(samples) * 1.3 + 3), 5))

    ax.bar(x, [cl if not np.isnan(cl) else 0 for cl in cls],
           color=bar_colors, alpha=0.85, edgecolor='gray', linewidth=0.5)

    # ER reference band
    ax.axhspan(ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM, alpha=0.14, color='limegreen',
               label=f'Expected ER domain ({ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm)')

    # Mean line
    valid_cls = [c for c in cls if not np.isnan(c)]
    if valid_cls:
        mean_cl = float(np.mean(valid_cls))
        ax.axhline(mean_cl, color='darkorange', linewidth=1.8, linestyle='--',
                   label=f'Mean = {mean_cl:.1f} µm')

    # Annotate bars
    method_abbrev = {
        'zero_crossing':     'ZC',
        'non_significant':   'NS',
        'always_significant':'AS',
        'always_negative':   'AN',
    }
    for xi, (cl, method) in enumerate(zip(cls, methods)):
        if not np.isnan(cl):
            ax.text(xi, cl + 0.15, f'{cl:.1f} µm', ha='center', va='bottom',
                    fontsize=8)
            ax.text(xi, 0.2, method_abbrev.get(method, '?'), ha='center',
                    va='bottom', fontsize=7, color='gray')
        else:
            ax.text(xi, 0.2, 'N/A', ha='center', va='bottom',
                    fontsize=7, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Correlation length (µm)', fontsize=11)
    ax.set_title('D domain size per embryo\n'
                 'Green = within ER range  |  '
                 'ZC=zero crossing  NS=non-significant  '
                 'AS=always significant  AN=always negative',
                 fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def save_summary_csv(all_results, output_path):
    rows = []
    for sample_id, res in all_results.items():
        cl     = res['corr_length']
        Is     = np.asarray(res['Is'], dtype=float)
        radii  = np.asarray(res['radii'])
        valid  = ~np.isnan(Is)
        max_I  = float(np.nanmax(Is)) if valid.any() else np.nan
        r_maxI = float(radii[np.nanargmax(Is)]) if valid.any() else np.nan
        rows.append({
            'sample_id':             sample_id,
            'n_trajectories':        res['n_trajs'],
            'correlation_length_um': round(cl, 3) if not np.isnan(cl) else np.nan,
            'method':                res['corr_method'],
            'in_ER_range':           bool(ER_DOMAIN_MIN_UM <= cl <= ER_DOMAIN_MAX_UM)
                                     if not np.isnan(cl) else False,
            'peak_morans_I':         round(max_I, 4) if not np.isnan(max_I) else np.nan,
            'radius_at_peak_I_um':   round(r_maxI, 2) if not np.isnan(r_maxI) else np.nan,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────────────────────
# Per-sample pipeline
# ─────────────────────────────────────────────────────────────
def process_sample(sample_id, pkl_path, folder, radii_sweep, rng,
                   summary_only=False):
    print(f"\n{'='*60}")
    print(f"  {sample_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(folder, f'{sample_id}_domain_size')
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectories
    print("  Loading trajectory pkl...")
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
    except Exception as e:
        print(f"  ERROR: {e}"); return None

    trajs = get_all_trajectories_um(pkl_data)
    if len(trajs) < MIN_TRAJECTORIES:
        print(f"  Only {len(trajs)} trajectories with valid D — skipping.")
        return None
    print(f"  {len(trajs)} trajectories with valid D")

    # Subsample
    if len(trajs) > MAX_N_TRAJS:
        idx   = rng.choice(len(trajs), MAX_N_TRAJS, replace=False)
        trajs = [trajs[i] for i in idx]
        print(f"  Subsampled to {MAX_N_TRAJS} trajectories for speed")

    n  = len(trajs)
    xs = np.array([t['x_um'] for t in trajs])
    ys = np.array([t['y_um'] for t in trajs])
    Ds = np.array([t['D']    for t in trajs])

    # Pre-compute pairwise distance matrix (done once)
    print(f"  Computing {n}×{n} pairwise distance matrix...")
    dx   = xs[:, None] - xs[None, :]
    dy   = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)

    # ── Correlogram sweep ─────────────────────────────────────
    half_w   = R_STEP_UM / 2.0
    Is_list  = []
    ps_list  = []
    np_list  = []

    n_lags = len(radii_sweep)
    print(f"  Sweeping {n_lags} lags "
          f"({radii_sweep[0]:.1f}–{radii_sweep[-1]:.1f} µm), "
          f"{N_PERMUTATIONS} permutations each...")

    for k, r in enumerate(radii_sweep):
        res = compute_ring_morans(Ds, dist, r, half_w, N_PERMUTATIONS, rng)
        Is_list.append(res['I'])
        ps_list.append(res['p'])
        np_list.append(res['n_pairs'])

        sig_flag = ('*' if (not np.isnan(res['p']) and res['p'] <= P_THRESHOLD)
                    else ' ')
        reliable_flag = '' if res['n_pairs'] >= MIN_PAIRS else ' [unreliable]'
        print(f"    r={r:5.1f} µm  pairs={res['n_pairs']:6d}  "
              f"I={res['I']:+.4f}  p={res['p']:.3f} {sig_flag}{reliable_flag}")

    # Correlation length
    corr_length, corr_method = find_correlation_length(radii_sweep, Is_list, ps_list)

    if not np.isnan(corr_length):
        in_er = ER_DOMAIN_MIN_UM <= corr_length <= ER_DOMAIN_MAX_UM
        print(f"\n  --> Correlation length : {corr_length:.2f} µm  ({corr_method})")
        print(f"      ER reference range  : {ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm")
        print(f"      Match               : {'YES ✓' if in_er else 'no'}")
    else:
        print(f"\n  --> Correlation length : not determined ({corr_method})")

    if not summary_only:
        plot_correlogram(radii_sweep, Is_list, ps_list, np_list,
                         corr_length, corr_method, sample_id,
                         os.path.join(output_dir, 'moran_correlogram.png'))
        save_correlogram_csv(radii_sweep, Is_list, ps_list, np_list,
                             sample_id,
                             os.path.join(output_dir, 'correlogram_data.csv'))
        print(f"  Output: {output_dir}")
    else:
        print("  [Per-embryo plots/CSV skipped — summary-only mode]")

    return {
        'radii':       list(radii_sweep),
        'Is':          Is_list,
        'ps':          ps_list,
        'n_pairs':     np_list,
        'corr_length': corr_length,
        'corr_method': corr_method if corr_method else 'none',
        'n_trajs':     n,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global N_PERMUTATIONS, R_MIN_UM, R_MAX_UM, R_STEP_UM
    global ER_DOMAIN_MIN_UM, ER_DOMAIN_MAX_UM

    print("D Domain Size Estimator — Spatial Correlogram")
    print("===============================================\n")

    folder = input("Enter path to data folder: ").strip()
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}"); sys.exit(1)

    # Radius sweep
    r_min_in = input(f"Min lag radius (µm) [{R_MIN_UM}]: ").strip()
    if r_min_in:
        try: R_MIN_UM = float(r_min_in)
        except ValueError: pass

    r_max_in = input(f"Max lag radius (µm) [{R_MAX_UM}]: ").strip()
    if r_max_in:
        try: R_MAX_UM = float(r_max_in)
        except ValueError: pass

    r_step_in = input(f"Lag step / ring width (µm) [{R_STEP_UM}]: ").strip()
    if r_step_in:
        try: R_STEP_UM = float(r_step_in)
        except ValueError: pass

    # ER reference range
    er_min_in = input(f"Expected ER domain min (µm) [{ER_DOMAIN_MIN_UM}]: ").strip()
    if er_min_in:
        try: ER_DOMAIN_MIN_UM = float(er_min_in)
        except ValueError: pass

    er_max_in = input(f"Expected ER domain max (µm) [{ER_DOMAIN_MAX_UM}]: ").strip()
    if er_max_in:
        try: ER_DOMAIN_MAX_UM = float(er_max_in)
        except ValueError: pass

    so_input = input("Save per-embryo plots/CSVs? [Y/n]: ").strip().lower()
    summary_only = so_input in ('n', 'no')
    if summary_only:
        print("Summary-only mode: per-embryo outputs will be skipped.\n")

    radii_sweep = np.arange(R_MIN_UM, R_MAX_UM + R_STEP_UM / 2.0, R_STEP_UM)
    print(f"\nLag sweep  : {radii_sweep[0]:.1f} → {radii_sweep[-1]:.1f} µm "
          f"in {len(radii_sweep)} steps of {R_STEP_UM} µm")
    print(f"ER range   : {ER_DOMAIN_MIN_UM}–{ER_DOMAIN_MAX_UM} µm")
    print(f"Permutations per lag: {N_PERMUTATIONS}\n")

    rng = np.random.default_rng(RANDOM_SEED)

    samples = find_sample_files(folder)
    if not samples:
        print("No pkl files found. Exiting."); sys.exit(1)

    print(f"\n{len(samples)} sample(s) to process.\n")
    all_results = {}
    for sample_id, pkl_path in samples:
        result = process_sample(sample_id, pkl_path, folder, radii_sweep, rng,
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
        f'domain_size_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(summary_dir, exist_ok=True)

    plot_cross_sample_correlogram(
        all_results, os.path.join(summary_dir, 'cross_sample_correlogram.png'))
    plot_domain_size_comparison(
        all_results, os.path.join(summary_dir, 'domain_size_comparison.png'))
    save_summary_csv(
        all_results, os.path.join(summary_dir, 'summary_table.csv'))

    print(f"\nSummary saved to: {summary_dir}")
    print("\nDone.")


if __name__ == '__main__':
    main()

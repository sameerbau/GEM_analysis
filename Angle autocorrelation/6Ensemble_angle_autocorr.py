#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6Ensemble_angle_autocorr.py

Ensemble-averaged angle autocorrelation analysis.

Unlike the flat-pool approach (Comparision_angle_autocorel.py), this script
treats each sample file as one biological replicate:

  1. Compute <cos theta(tau)> independently for each sample file.
  2. Aggregate into a weighted ensemble mean curve,
     where weights = n_pairs at each lag (how many displacement-pair
     measurements that sample contributes at every lag).
  3. Quantify uncertainty via sample-level bootstrap (resample FILES, not
     individual trajectories), giving a CI on the ensemble curve and on the
     summary statistic t_cross (zero-crossing time).
  4. Extract per-sample t_cross values (one per file/embryo) to enable proper
     non-parametric statistics (Mann-Whitney U, Cliff's delta) when comparing
     conditions — analogous to comparing per-embryo D values inside/outside ROIs.

This mirrors the ensemble-correlogram approach in
'7 domain_size_correlogram.py' (Moran's I, spatial), here applied to
temporal autocorrelation of step directions.

Summary statistic:
  t_cross  — persistence time: first lag where <cos theta> crosses zero,
              linearly interpolated (positive = correlated, negative = anti-correlated)

Input:
  analyzed_*.pkl files — standard pipeline output, one file per embryo/movie
  (from diffusion_analyzer.py or trajectory_data_pooler.py)
  Each pkl must contain data['trajectories'] with 'x' and 'y' position lists.

Output (written to angle_ensemble_<timestamp>/):
  ensemble_angle_curves.png    ensemble mean curves + 95% CI bands +
                               individual per-sample curves (faint overlay)
  per_sample_summary.png       per-sample t_cross by condition
                               (strip/box plot, same style as D-value plots)
  ensemble_data.csv            tau, mean_cos, SE, CI_lo, CI_hi per condition
  per_sample_summary.csv       per-sample t_cross, n_trajs, condition
  pairwise_statistics.csv      Mann-Whitney p, Cliff's delta, effect size

Usage:
  python "6Ensemble_angle_autocorr.py"
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
from datetime import datetime

# ============================================================
# Parameters — edit these to match your data
# ============================================================
DT              = 0.1   # seconds per frame
MAX_LAG         = 50    # maximum lag in frames (ANGLE_PLOT_CUTOFF)
MIN_LENGTH      = 15    # minimum trajectory length to include (frames)
MIN_TRAJS       = 3     # minimum qualifying trajectories per sample
N_BOOTSTRAP     = 500   # bootstrap replicates for ensemble CI
ALPHA           = 0.05  # significance level
CONFIDENCE      = 0.95  # CI level (two-tailed -> 2.5 / 97.5 percentiles)
RANDOM_SEED     = 42
FIGURE_DPI      = 300
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
    """Return sorted list of (sample_id, path) for analyzed_*.pkl."""
    paths = sorted(glob.glob(os.path.join(folder, 'analyzed_*.pkl')))
    if not paths:
        print(f"  [WARN] No analyzed_*.pkl found in {folder}")
        return []
    samples = []
    for p in paths:
        sid = os.path.splitext(os.path.basename(p))[0].replace('analyzed_', '')
        samples.append((sid, p))
    return samples


def extract_trajectories(data):
    """Pull trajectory list from a loaded pkl dict."""
    if data is None:
        return []
    if 'trajectories' in data:
        return data['trajectories']
    for key in data:
        if isinstance(data[key], list) and len(data[key]) > 0:
            if isinstance(data[key][0], dict) and 'x' in data[key][0]:
                return data[key]
    return []


# ─────────────────────────────────────────────────────────────
# Core: per-sample <cos theta(tau)> calculation
# ─────────────────────────────────────────────────────────────

def compute_sample_angle_autocorr(trajectories, dt=DT, max_lag=MAX_LAG,
                                   min_length=MIN_LENGTH):
    """
    Compute the angle (directional) autocorrelation function for one sample.

    <cos theta(tau)> = mean of cos(angle between displacement d(t) and d(t+tau))
                     = mean of (d(t) . d(t+tau)) / (|d(t)| |d(t+tau)|)

    Note: this uses DISPLACEMENT vectors (step vectors), not velocity vectors.
    It is purely geometric (direction only, no speed weighting), and therefore
    complementary to the velocity autocorrelation.

    Returns dict with:
      cos_angle  (max_lag,) — mean <cos theta> at each lag
      n_pairs    (max_lag,) — number of valid pairs (weight for ensemble)
      t_cross    float      — zero-crossing time (NaN if not found)
      n_trajs    int        — trajectories that passed the length filter
    """
    filtered = [t for t in trajectories
                if isinstance(t.get('x'), (list, np.ndarray))
                and len(t['x']) > min_length]

    cos_angle = np.full(max_lag, np.nan)
    n_pairs   = np.zeros(max_lag, dtype=np.int64)

    if len(filtered) < 1:
        return {'cos_angle': cos_angle, 'n_pairs': n_pairs,
                't_cross': np.nan, 'n_trajs': 0}

    # Pool per lag
    pools = [[] for _ in range(max_lag)]

    for traj in filtered:
        x      = np.asarray(traj['x'], dtype=float)
        y      = np.asarray(traj['y'], dtype=float)
        dx     = np.diff(x)
        dy     = np.diff(y)
        n_disp = len(dx)

        for lag_idx in range(max_lag):
            lag = lag_idx + 1
            if n_disp <= lag:
                continue
            # All pairs (k, k+lag)
            dx0  = dx[:n_disp - lag];  dy0  = dy[:n_disp - lag]
            dx_l = dx[lag:];           dy_l = dy[lag:]
            mag0 = np.sqrt(dx0**2  + dy0**2)
            magl = np.sqrt(dx_l**2 + dy_l**2)
            ok   = (mag0 > 0) & (magl > 0)
            if not np.any(ok):
                continue
            cos_vals = ((dx0[ok]*dx_l[ok] + dy0[ok]*dy_l[ok])
                        / (mag0[ok] * magl[ok]))
            cos_vals = np.clip(cos_vals, -1.0, 1.0)
            pools[lag_idx].extend(cos_vals.tolist())

    time_lags = np.arange(1, max_lag + 1) * dt

    for k, pool in enumerate(pools):
        if pool:
            arr = np.asarray(pool)
            arr = arr[~np.isnan(arr)]
            if len(arr) > 0:
                cos_angle[k] = float(np.mean(arr))
                n_pairs[k]   = len(arr)

    t_cross = _find_zero_crossing(cos_angle, time_lags)

    return {
        'cos_angle': cos_angle,
        'n_pairs':   n_pairs,
        't_cross':   t_cross,
        'n_trajs':   len(filtered),
    }


# ─────────────────────────────────────────────────────────────
# Summary-statistic helper
# ─────────────────────────────────────────────────────────────

def _find_zero_crossing(cos_angle, time_lags):
    """
    Return first lag where <cos theta> crosses zero (positive -> negative),
    linearly interpolated between the two bracketing frames.
    """
    valid = ~np.isnan(cos_angle)
    if valid.sum() < 2:
        return np.nan
    ca = cos_angle[valid]
    tl = time_lags[valid]
    for i in range(len(ca) - 1):
        if ca[i] > 0 >= ca[i + 1]:
            frac = ca[i] / (ca[i] - ca[i + 1])
            return float(tl[i] + frac * (tl[i + 1] - tl[i]))
    return np.nan


# ─────────────────────────────────────────────────────────────
# Ensemble aggregation
# ─────────────────────────────────────────────────────────────

def compute_ensemble(per_sample, time_lags):
    """
    n_pairs-weighted ensemble mean <cos theta(tau)> and SE.

    Weights w_s(tau) = n_pairs_s(tau): samples contributing more independent
    displacement-pair measurements are weighted proportionally more.
    SE uses the Kish effective sample size for unequal weights.
    """
    max_lag = len(time_lags)
    ca_mat  = np.array([r['cos_angle'] for r in per_sample], dtype=float)
    np_mat  = np.array([r['n_pairs']   for r in per_sample], dtype=float)

    ens_ca = np.full(max_lag, np.nan)
    ens_se = np.full(max_lag, np.nan)

    for k in range(max_lag):
        ca  = ca_mat[:, k]
        w   = np_mat[:, k]
        ok  = ~np.isnan(ca) & (w > 0)
        if ok.sum() < 2:
            continue
        ca_ok = ca[ok];  w_ok = w[ok]
        w_sum = w_ok.sum()

        # Weighted mean
        mean_k     = float(np.dot(w_ok, ca_ok) / w_sum)
        ens_ca[k]  = mean_k

        # Kish ESS-based SE
        n_eff = w_sum**2 / np.dot(w_ok, w_ok)
        if n_eff > 1:
            wvar       = float(np.dot(w_ok, (ca_ok - mean_k)**2) / w_sum)
            ens_se[k]  = float(np.sqrt(wvar / n_eff))

    return ens_ca, ens_se


# ─────────────────────────────────────────────────────────────
# Bootstrap CI (sample-level resampling)
# ─────────────────────────────────────────────────────────────

def bootstrap_ensemble(per_sample, time_lags, n_bootstrap=N_BOOTSTRAP,
                       rng=None):
    """
    Resample samples (not trajectories) with replacement.
    Propagates between-sample biological variance into:
      - CI on the ensemble <cos theta> curve at each lag
      - CI on t_cross (zero-crossing time)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_s     = len(per_sample)
    max_lag = len(time_lags)

    boot_ca     = np.full((n_bootstrap, max_lag), np.nan)
    boot_tcross = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        idx      = rng.integers(0, n_s, size=n_s)
        boot_res = [per_sample[i] for i in idx]
        bc, _    = compute_ensemble(boot_res, time_lags)
        boot_ca[b]     = bc
        boot_tcross[b] = _find_zero_crossing(bc, time_lags)

    alpha_tail = (1.0 - CONFIDENCE) / 2.0 * 100

    def pci(arr):
        a = arr[~np.isnan(arr)]
        if len(a) < 10:
            return np.nan, np.nan
        return (float(np.percentile(a, alpha_tail)),
                float(np.percentile(a, 100 - alpha_tail)))

    ci_lo = np.nanpercentile(boot_ca, alpha_tail,      axis=0)
    ci_hi = np.nanpercentile(boot_ca, 100 - alpha_tail, axis=0)

    return {
        'ci_lo':       ci_lo,
        'ci_hi':       ci_hi,
        'tcross_ci':   pci(boot_tcross),
        'boot_tcross': boot_tcross,
    }


# ─────────────────────────────────────────────────────────────
# Load one condition
# ─────────────────────────────────────────────────────────────

def load_condition(folder, name):
    """
    Process all sample files in a directory.

    Returns a condition dict with ensemble curve, bootstrap CI,
    and per-sample t_cross array for statistics.
    """
    samples = find_sample_files(folder)
    if not samples:
        return None

    time_lags  = np.arange(1, MAX_LAG + 1) * DT
    per_sample = []

    for sid, path in samples:
        data  = load_pkl(path)
        trajs = extract_trajectories(data)
        res   = compute_sample_angle_autocorr(trajs)
        if res['n_trajs'] < MIN_TRAJS:
            print(f"  [SKIP] {sid}: only {res['n_trajs']} qualifying trajectories")
            continue
        res['sample_id'] = sid
        per_sample.append(res)
        tcstr = f"{res['t_cross']:.3f}s" if not np.isnan(res['t_cross']) else "NaN"
        print(f"  {sid}: {res['n_trajs']} trajs | t_cross={tcstr}")

    if len(per_sample) < 2:
        print(f"  [ERROR] Need at least 2 valid samples, got {len(per_sample)}")
        return None

    rng           = np.random.default_rng(RANDOM_SEED)
    ens_ca, ens_se = compute_ensemble(per_sample, time_lags)
    boot          = bootstrap_ensemble(per_sample, time_lags, rng=rng)

    tcross_vals   = np.array([r['t_cross'] for r in per_sample])
    ens_tcross    = _find_zero_crossing(ens_ca, time_lags)

    return {
        'name':        name,
        'folder':      folder,
        'per_sample':  per_sample,
        'time_lags':   time_lags,
        'ens_ca':      ens_ca,
        'ens_se':      ens_se,
        'boot':        boot,
        'ens_tcross':  ens_tcross,
        'tcross_vals': tcross_vals,
        'n_samples':   len(per_sample),
    }


# ─────────────────────────────────────────────────────────────
# Statistics
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
    """
    Pairwise Mann-Whitney U + Cliff's delta on per-sample t_cross values.

    One t_cross per file/embryo = one biological replicate.
    """
    rows = []
    for (i, j) in combinations(range(len(conditions)), 2):
        cA, cB = conditions[i], conditions[j]
        a = cA['tcross_vals'][~np.isnan(cA['tcross_vals'])]
        b = cB['tcross_vals'][~np.isnan(cB['tcross_vals'])]
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        d    = cliffs_delta(a, b)
        rows.append({
            'condition_A':    cA['name'],
            'condition_B':    cB['name'],
            'n_A':            len(a),
            'n_B':            len(b),
            'median_A_s':     float(np.median(a)),
            'median_B_s':     float(np.median(b)),
            'mean_A_s':       float(np.mean(a)),
            'mean_B_s':       float(np.mean(b)),
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
    Ensemble <cos theta(tau)> curves with CI bands and per-sample overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = _cond_colors(len(conditions))

    for col, cond in zip(colors, conditions):
        tl    = cond['time_lags']
        ens   = cond['ens_ca']
        ci_lo = cond['boot']['ci_lo']
        ci_hi = cond['boot']['ci_hi']

        # Per-sample curves (faint background)
        for r in cond['per_sample']:
            ax.plot(tl, r['cos_angle'], color=col, alpha=0.12, linewidth=0.8)

        # Ensemble mean
        ax.plot(tl, ens, color=col, linewidth=2.2,
                label=f"{cond['name']}  (n={cond['n_samples']} samples)")

        # CI band
        ax.fill_between(tl, ci_lo, ci_hi, color=col, alpha=0.18)

        # Mark t_cross
        if not np.isnan(cond['ens_tcross']):
            ax.axvline(cond['ens_tcross'], color=col, linestyle='--',
                       linewidth=1.0, alpha=0.7)

    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.4,
               label='zero line (persistence boundary)')

    ax.set_xlabel('Time lag  τ  (s)', fontsize=12)
    ax.set_ylabel(r'$\langle \cos\theta(\tau) \rangle$', fontsize=13)
    ax.set_title('Ensemble Angle Autocorrelation\n'
                 '(shaded = 95% bootstrap CI; faint lines = individual samples)',
                 fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_angle_curves.png'), dpi=FIGURE_DPI)
    plt.close()
    print("  Saved: ensemble_angle_curves.png")


def plot_per_sample_summary(conditions, stats_rows, output_dir):
    """
    Boxplot + jitter of per-sample t_cross per condition.
    Significance bars for pairwise significant comparisons.
    """
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(conditions) + 2), 6))
    colors  = _cond_colors(len(conditions))
    rng_jit = np.random.default_rng(0)

    for xi, (col, cond) in enumerate(zip(colors, conditions)):
        vals = cond['tcross_vals']
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue

        ax.boxplot(vals, positions=[xi], widths=0.38,
                   patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.35),
                   medianprops=dict(color='black', linewidth=1.8),
                   whiskerprops=dict(linewidth=1.0),
                   capprops=dict(linewidth=1.0),
                   flierprops=dict(marker='', linestyle='none'),
                   showfliers=False)

        jitter = rng_jit.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + jitter, vals, color=col, s=32, alpha=0.78,
                   zorder=5, edgecolors='white', linewidths=0.4)

    # Significance bars
    if stats_rows:
        sig   = [r for r in stats_rows if r['significant']]
        y_top = max(
            (np.nanmax(cond['tcross_vals']) for cond in conditions
             if np.any(~np.isnan(cond['tcross_vals']))),
            default=1.0
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
                    [ybar - bar_step*0.1, ybar, ybar, ybar - bar_step*0.1],
                    color='k', linewidth=1.0)
            pval = comp['p_value']
            pstr = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            ax.text((xA + xB) / 2, ybar + bar_step * 0.05, pstr,
                    ha='center', va='bottom', fontsize=11)

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([c['name'] for c in conditions],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(r'Zero-crossing time  $t_{cross}$  (s)', fontsize=11)
    ax.set_title('Per-sample Zero-Crossing Times\n'
                 '(each point = one embryo/movie; box = median + IQR)',
                 fontsize=11)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_sample_summary.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: per_sample_summary.png")


def plot_bootstrap_distribution(conditions, output_dir):
    """
    Histogram of bootstrap t_cross distributions per condition.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = _cond_colors(len(conditions))

    for col, cond in zip(colors, conditions):
        arr = cond['boot']['boot_tcross']
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        ci = cond['boot']['tcross_ci']
        ax.hist(arr, bins=30, color=col, alpha=0.45,
                label=f"{cond['name']}  ({len(arr)}/{N_BOOTSTRAP} converged)")
        if not np.isnan(ci[0]):
            ax.axvline(ci[0], color=col, linestyle=':', linewidth=1.2)
            ax.axvline(ci[1], color=col, linestyle=':', linewidth=1.2)

    ax.set_xlabel(r'Bootstrap ensemble  $t_{cross}$  (s)', fontsize=11)
    ax.set_ylabel('Bootstrap replicates', fontsize=11)
    ax.set_title('Bootstrap CI on ensemble zero-crossing time\n'
                 '(dotted lines = 95% CI boundaries)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_distribution.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: bootstrap_distribution.png")


def plot_individual_overlay(conditions, output_dir):
    """
    Multi-panel plot: one panel per condition, showing all per-sample
    <cos theta> curves coloured by their t_cross value (cool-to-warm scale).
    Useful for spotting heterogeneity within a condition.
    """
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        tl     = cond['time_lags']
        tcvals = cond['tcross_vals']
        vmin   = np.nanmin(tcvals) if not np.all(np.isnan(tcvals)) else 0
        vmax   = np.nanmax(tcvals) if not np.all(np.isnan(tcvals)) else 1
        cmap   = plt.cm.coolwarm

        for r in cond['per_sample']:
            tc  = r['t_cross']
            col = cmap((tc - vmin) / max(vmax - vmin, 1e-9)) \
                  if not np.isnan(tc) else 'grey'
            ax.plot(tl, r['cos_angle'], color=col, alpha=0.5, linewidth=0.9)

        # Ensemble overlay
        ax.plot(tl, cond['ens_ca'], 'k-', linewidth=2.0,
                label='ensemble mean')
        ax.fill_between(tl, cond['boot']['ci_lo'], cond['boot']['ci_hi'],
                        color='k', alpha=0.12)

        ax.axhline(0, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_xlabel('Time lag  τ  (s)', fontsize=11)
        ax.set_title(f"{cond['name']}  (n={cond['n_samples']})", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=r'$t_{cross}$ (s)')

    axes[0].set_ylabel(r'$\langle \cos\theta(\tau) \rangle$', fontsize=12)
    plt.suptitle('Per-sample angle autocorrelation coloured by t_cross',
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
        tl    = cond['time_lags']
        ens   = cond['ens_ca']
        se    = cond['ens_se']
        ci_lo = cond['boot']['ci_lo']
        ci_hi = cond['boot']['ci_hi']
        for k in range(len(tl)):
            rows.append({
                'condition':  cond['name'],
                'time_lag_s': float(tl[k]),
                'ensemble_cos_angle': float(ens[k]) if not np.isnan(ens[k]) else '',
                'ensemble_SE':        float(se[k])  if not np.isnan(se[k])  else '',
                'CI_lo_95':           float(ci_lo[k]) if not np.isnan(ci_lo[k]) else '',
                'CI_hi_95':           float(ci_hi[k]) if not np.isnan(ci_hi[k]) else '',
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'ensemble_data.csv'), index=False)
    print("  Saved: ensemble_data.csv")


def export_per_sample_summary(conditions, output_dir):
    rows = []
    for cond in conditions:
        for r in cond['per_sample']:
            rows.append({
                'condition':  cond['name'],
                'sample_id':  r['sample_id'],
                'n_trajs':    r['n_trajs'],
                't_cross_s':  r['t_cross'] if not np.isnan(r['t_cross']) else '',
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
    print("\n" + "=" * 60)
    print("ENSEMBLE ANGLE AUTOCORRELATION — SUMMARY")
    print("=" * 60)

    for cond in conditions:
        print(f"\n{cond['name']}  ({cond['n_samples']} samples)")
        if not np.isnan(cond['ens_tcross']):
            print(f"  Ensemble t_cross : {cond['ens_tcross']:.3f} s")
        else:
            print("  Ensemble t_cross : not found (no zero-crossing in range)")
        ci = cond['boot']['tcross_ci']
        if not np.isnan(ci[0]):
            print(f"  95% CI (boot)    : [{ci[0]:.3f}, {ci[1]:.3f}] s")
        tc = cond['tcross_vals'][~np.isnan(cond['tcross_vals'])]
        if len(tc):
            print(f"  Per-sample t_cross: median={np.median(tc):.3f}s, "
                  f"IQR=[{np.percentile(tc,25):.3f}, {np.percentile(tc,75):.3f}]")

    if stats_rows:
        print("\nPairwise comparisons (per-sample t_cross):")
        for r in stats_rows:
            sig = "SIGNIFICANT" if r['significant'] else "n.s."
            print(f"  {r['condition_A']} vs {r['condition_B']}")
            print(f"    p={r['p_value']:.4e}  {sig}  "
                  f"Cliff's d={r['cliffs_delta']:.3f}  ({r['effect_size']})")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Ensemble Angle Autocorrelation Analysis")
    print("=" * 43)
    print("Treats each analyzed_*.pkl as one biological replicate.\n")

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
        default_name = os.path.basename(os.path.normpath(folder)) or f"Condition_{i+1}"
        name = input(f"  Name for condition {i+1} [Enter = '{default_name}']: ").strip()
        if not name:
            name = default_name
        print(f"\nLoading '{name}' from {folder} ...")
        cond = load_condition(folder, name)
        if cond is None:
            print(f"  Failed to load condition '{name}'"); return
        conditions.append(cond)
        print(f"  -> {cond['n_samples']} valid samples loaded.")

    # Output directory
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"angle_ensemble_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Statistics
    stats_rows = compare_conditions(conditions) if len(conditions) >= 2 else []

    # Plots
    print("Generating plots ...")
    plot_ensemble_curves(conditions, output_dir)
    plot_per_sample_summary(conditions, stats_rows, output_dir)
    plot_bootstrap_distribution(conditions, output_dir)
    plot_individual_overlay(conditions, output_dir)

    # CSV exports
    print("Exporting data ...")
    export_ensemble_data(conditions, output_dir)
    export_per_sample_summary(conditions, output_dir)
    export_statistics(stats_rows, output_dir)

    # Console summary
    print_summary(conditions, stats_rows)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

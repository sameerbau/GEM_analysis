#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6Ensemble_velocity_autocorr.py

Ensemble-averaged velocity autocorrelation analysis.

Unlike the flat-pool approach (5Compare_Velocity_autocorrel.py), this script
treats each sample file as one biological replicate:

  1. Compute Cv(tau) independently for each sample file.
  2. Aggregate into a weighted ensemble mean curve,
     where weights = n_pairs at each lag (not just n_trajectories — this
     properly accounts for how many independent correlation estimates each
     sample contributes at every lag, including short-trajectory dropout at
     long lags).
  3. Quantify uncertainty via sample-level bootstrap (resample FILES, not
     individual trajectories), giving a CI on the ensemble curve and on the
     summary statistics tau_c and tau_zero.
  4. Extract per-sample tau_c values (one per file/embryo) to enable proper
     non-parametric statistics (Mann-Whitney U, Cliff's delta) when comparing
     conditions — analogous to comparing per-embryo D values inside/outside ROIs.

This mirrors the ensemble-correlogram approach in
'7 domain_size_correlogram.py' (Moran's I, spatial), here applied to
temporal autocorrelation.

Summary statistics extracted per sample:
  tau_c    — correlation time: lag where Cv decays to Cv(tau=1)/e
             (linear interpolation between frames)
  tau_zero — zero-crossing time: first lag where Cv < 0
             (linear interpolation; indicates transition to anti-correlation)

Input:
  analyzed_*.pkl files — standard pipeline output, one file per embryo/movie
  (from diffusion_analyzer.py or trajectory_data_pooler.py)
  Each pkl must contain data['trajectories'] with 'x' and 'y' position lists.

Output (written to velocity_ensemble_<timestamp>/):
  ensemble_Cv_curves.png       ensemble mean curves + 95% CI bands +
                               individual per-sample curves (faint overlay)
  per_sample_summary.png       per-sample tau_c and tau_zero by condition
                               (strip/box plot, same style as D-value plots)
  ensemble_data.csv            tau, Cv, SE, CI_lo, CI_hi per condition
  per_sample_summary.csv       per-sample tau_c, tau_zero, n_trajs, condition
  pairwise_statistics.csv      Mann-Whitney p, Cliff's delta, effect size

Usage:
  python "6Ensemble_velocity_autocorr.py"
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
DT          = 0.1   # seconds per frame
MAX_TAU     = 50    # maximum lag in frames
MIN_LENGTH  = 15    # minimum trajectory length to include (frames)
MIN_TRAJS   = 3     # minimum qualifying trajectories per sample
N_BOOTSTRAP = 500   # bootstrap replicates for ensemble CI
ALPHA       = 0.05  # significance level
CONFIDENCE  = 0.95  # CI level (two-tailed -> percentiles 2.5 / 97.5)
RANDOM_SEED = 42
FIGURE_DPI  = 300
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
    """Return sorted list of (sample_id, path) tuples for analyzed_*.pkl."""
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
    # Fallback: some pooler outputs store under a different key
    for key in data:
        if isinstance(data[key], list) and len(data[key]) > 0:
            if isinstance(data[key][0], dict) and 'x' in data[key][0]:
                return data[key]
    return []


# ─────────────────────────────────────────────────────────────
# Core: per-sample Cv(tau) calculation
# ─────────────────────────────────────────────────────────────

def compute_sample_velocity_autocorr(trajectories, dt=DT, max_tau=MAX_TAU,
                                     min_length=MIN_LENGTH):
    """
    Compute the velocity autocorrelation function for one sample.

    Cv(tau) = < v_hat(t) . v_hat(t+tau) >
    where v_hat = v / |v| (unit velocity vector).

    Returns dict with:
      Cv       (max_tau,) — mean correlation at each lag
      n_pairs  (max_tau,) — number of valid pairs used (weight for ensemble)
      tau_c    float      — 1/e decay time (NaN if not found)
      tau_zero float      — first zero-crossing time (NaN if not found)
      n_trajs  int        — number of trajectories that passed the length filter
    """
    filtered = [t for t in trajectories
                if isinstance(t.get('x'), (list, np.ndarray))
                and len(t['x']) > min_length]

    Cv      = np.full(max_tau, np.nan)
    n_pairs = np.zeros(max_tau, dtype=np.int64)

    if len(filtered) < 1:
        return {'Cv': Cv, 'n_pairs': n_pairs,
                'tau_c': np.nan, 'tau_zero': np.nan, 'n_trajs': 0}

    # Pre-compute velocity arrays once per trajectory
    vel_cache = []
    for traj in filtered:
        x  = np.asarray(traj['x'], dtype=float)
        y  = np.asarray(traj['y'], dtype=float)
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        vel_cache.append((vx, vy))

    for tau_idx in range(max_tau):
        tau = tau_idx + 1
        pool = []

        for vx, vy in vel_cache:
            if len(vx) <= tau:
                continue
            vx0   = vx[:-tau];  vy0   = vy[:-tau]
            vx_t  = vx[tau:];   vy_t  = vy[tau:]
            mag0  = np.sqrt(vx0**2  + vy0**2)
            mag_t = np.sqrt(vx_t**2 + vy_t**2)
            ok    = (mag0 > 0) & (mag_t > 0)
            if not np.any(ok):
                continue
            corr = (vx0[ok]*vx_t[ok] + vy0[ok]*vy_t[ok]) / (mag0[ok]*mag_t[ok])
            pool.extend(corr.tolist())

        if pool:
            arr = np.asarray(pool)
            arr = arr[~np.isnan(arr)]
            if len(arr) > 0:
                Cv[tau_idx]      = float(np.mean(arr))
                n_pairs[tau_idx] = len(arr)

    time_lags = np.arange(1, max_tau + 1) * dt
    tau_c     = _find_tau_c(Cv, time_lags)
    tau_zero  = _find_zero_crossing(Cv, time_lags)

    return {
        'Cv':      Cv,
        'n_pairs': n_pairs,
        'tau_c':   tau_c,
        'tau_zero': tau_zero,
        'n_trajs': len(filtered),
    }


# ─────────────────────────────────────────────────────────────
# Summary-statistic helpers (with linear interpolation)
# ─────────────────────────────────────────────────────────────

def _find_tau_c(Cv, time_lags):
    """Return lag where Cv decays to Cv[0]/e (linearly interpolated)."""
    valid = ~np.isnan(Cv)
    if valid.sum() < 2:
        return np.nan
    cv  = Cv[valid]
    tl  = time_lags[valid]
    if cv[0] <= 0:
        return np.nan
    threshold = cv[0] / np.e
    for i in range(len(cv) - 1):
        if cv[i] >= threshold > cv[i + 1]:
            frac = (cv[i] - threshold) / (cv[i] - cv[i + 1])
            return float(tl[i] + frac * (tl[i + 1] - tl[i]))
    return np.nan


def _find_zero_crossing(Cv, time_lags):
    """Return first lag where Cv crosses from positive to negative."""
    valid = ~np.isnan(Cv)
    if valid.sum() < 2:
        return np.nan
    cv = Cv[valid]
    tl = time_lags[valid]
    for i in range(len(cv) - 1):
        if cv[i] > 0 >= cv[i + 1]:
            frac = cv[i] / (cv[i] - cv[i + 1])
            return float(tl[i] + frac * (tl[i + 1] - tl[i]))
    return np.nan


# ─────────────────────────────────────────────────────────────
# Ensemble aggregation
# ─────────────────────────────────────────────────────────────

def compute_ensemble(per_sample, time_lags):
    """
    Compute n_pairs-weighted ensemble mean Cv(tau) and SE.

    Weights w_s(tau) = n_pairs_s(tau): samples that contribute more
    independent correlation measurements receive proportionally more weight.
    SE uses the Kish effective sample size to avoid underestimation when
    weights are unequal.

    Returns:
      ens_Cv   (max_tau,)  weighted ensemble mean
      ens_se   (max_tau,)  weighted SE (Kish ESS)
    """
    max_tau = len(time_lags)
    n_s     = len(per_sample)

    Cv_mat  = np.array([r['Cv']      for r in per_sample], dtype=float)  # (n_s, max_tau)
    np_mat  = np.array([r['n_pairs'] for r in per_sample], dtype=float)  # (n_s, max_tau)

    ens_Cv = np.full(max_tau, np.nan)
    ens_se = np.full(max_tau, np.nan)

    for k in range(max_tau):
        cv  = Cv_mat[:, k]
        w   = np_mat[:, k]
        ok  = ~np.isnan(cv) & (w > 0)
        if ok.sum() < 2:
            continue
        cv_ok = cv[ok];  w_ok = w[ok]
        w_sum = w_ok.sum()

        # Weighted mean
        mean_k = float(np.dot(w_ok, cv_ok) / w_sum)
        ens_Cv[k] = mean_k

        # Kish effective n
        n_eff = w_sum**2 / np.dot(w_ok, w_ok)
        if n_eff > 1:
            wvar = float(np.dot(w_ok, (cv_ok - mean_k)**2) / w_sum)
            ens_se[k] = float(np.sqrt(wvar / n_eff))

    return ens_Cv, ens_se


# ─────────────────────────────────────────────────────────────
# Bootstrap CI (sample-level resampling)
# ─────────────────────────────────────────────────────────────

def bootstrap_ensemble(per_sample, time_lags, n_bootstrap=N_BOOTSTRAP,
                       rng=None):
    """
    Resample samples (not trajectories) with replacement to get:
      - CI on the ensemble Cv curve at each lag
      - CI on tau_c (1/e correlation time)
      - CI on tau_zero (zero-crossing time)

    This propagates between-sample biological variance into the CI.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_s    = len(per_sample)
    max_tau = len(time_lags)

    boot_Cv    = np.full((n_bootstrap, max_tau), np.nan)
    boot_tau_c = np.full(n_bootstrap, np.nan)
    boot_tau_z = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        idx      = rng.integers(0, n_s, size=n_s)
        boot_res = [per_sample[i] for i in idx]
        bc, _    = compute_ensemble(boot_res, time_lags)
        boot_Cv[b]    = bc
        boot_tau_c[b] = _find_tau_c(bc, time_lags)
        boot_tau_z[b] = _find_zero_crossing(bc, time_lags)

    alpha_tail = (1.0 - CONFIDENCE) / 2.0 * 100

    def pci(arr):
        a = arr[~np.isnan(arr)]
        if len(a) < 10:
            return np.nan, np.nan
        return float(np.percentile(a, alpha_tail)), \
               float(np.percentile(a, 100 - alpha_tail))

    ci_lo = np.nanpercentile(boot_Cv, alpha_tail,     axis=0)
    ci_hi = np.nanpercentile(boot_Cv, 100-alpha_tail, axis=0)

    tau_c_ci   = pci(boot_tau_c)
    tau_zero_ci = pci(boot_tau_z)

    return {
        'ci_lo':       ci_lo,
        'ci_hi':       ci_hi,
        'tau_c_ci':    tau_c_ci,
        'tau_zero_ci': tau_zero_ci,
        'boot_tau_c':  boot_tau_c,
        'boot_tau_z':  boot_tau_z,
    }


# ─────────────────────────────────────────────────────────────
# Load one condition (directory of analyzed_*.pkl files)
# ─────────────────────────────────────────────────────────────

def load_condition(folder, name):
    """
    Process all sample files in a directory.

    Returns a condition dict with:
      name, per_sample, ensemble_Cv, ensemble_se, boot, time_lags
      and per-sample tau_c / tau_zero arrays for statistics.
    """
    samples = find_sample_files(folder)
    if not samples:
        return None

    time_lags   = np.arange(1, MAX_TAU + 1) * DT
    per_sample  = []

    for sid, path in samples:
        data  = load_pkl(path)
        trajs = extract_trajectories(data)
        res   = compute_sample_velocity_autocorr(trajs)
        if res['n_trajs'] < MIN_TRAJS:
            print(f"  [SKIP] {sid}: only {res['n_trajs']} qualifying trajectories")
            continue
        res['sample_id'] = sid
        per_sample.append(res)
        print(f"  {sid}: {res['n_trajs']} trajs | "
              f"tau_c={res['tau_c']:.3f}s | "
              f"tau_zero={('%.3f' % res['tau_zero']) if not np.isnan(res['tau_zero']) else 'NaN'}s")

    if len(per_sample) < 2:
        print(f"  [ERROR] Need at least 2 valid samples for ensemble, got {len(per_sample)}")
        return None

    rng           = np.random.default_rng(RANDOM_SEED)
    ens_Cv, ens_se = compute_ensemble(per_sample, time_lags)
    boot          = bootstrap_ensemble(per_sample, time_lags, rng=rng)

    tau_c_vals  = np.array([r['tau_c']   for r in per_sample])
    tau_z_vals  = np.array([r['tau_zero'] for r in per_sample])

    # Ensemble-level summary statistics
    ens_tau_c   = _find_tau_c(ens_Cv, time_lags)
    ens_tau_z   = _find_zero_crossing(ens_Cv, time_lags)

    return {
        'name':       name,
        'folder':     folder,
        'per_sample': per_sample,
        'time_lags':  time_lags,
        'ens_Cv':     ens_Cv,
        'ens_se':     ens_se,
        'boot':       boot,
        'ens_tau_c':  ens_tau_c,
        'ens_tau_z':  ens_tau_z,
        'tau_c_vals': tau_c_vals,
        'tau_z_vals': tau_z_vals,
        'n_samples':  len(per_sample),
    }


# ─────────────────────────────────────────────────────────────
# Statistics between conditions
# ─────────────────────────────────────────────────────────────

def cliffs_delta(a, b):
    """Non-parametric effect size: proportion of pairs where a > b."""
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
    Pairwise Mann-Whitney U + Cliff's delta on per-sample tau_c and tau_zero.

    Per-sample values (one per file/embryo) are the proper unit of comparison —
    this avoids pseudo-replication from treating individual trajectories as
    independent biological observations.
    """
    rows = []
    for (i, j) in combinations(range(len(conditions)), 2):
        cA, cB = conditions[i], conditions[j]
        for metric, key in [('tau_c', 'tau_c_vals'), ('tau_zero', 'tau_z_vals')]:
            a = cA[key][~np.isnan(cA[key])]
            b = cB[key][~np.isnan(cB[key])]
            if len(a) < 2 or len(b) < 2:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            d    = cliffs_delta(a, b)
            rows.append({
                'condition_A': cA['name'],
                'condition_B': cB['name'],
                'metric':      metric,
                'n_A':         len(a),
                'n_B':         len(b),
                'median_A':    float(np.median(a)),
                'median_B':    float(np.median(b)),
                'mean_A':      float(np.mean(a)),
                'mean_B':      float(np.mean(b)),
                'mann_whitney_U': float(u),
                'p_value':       float(p),
                'significant':   bool(p < ALPHA),
                'cliffs_delta':  d,
                'effect_size':   interpret_cliffs(d),
            })
    return rows


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

CMAP = plt.cm.tab10

def _condition_colors(n):
    return [CMAP(i / max(n - 1, 1)) for i in range(n)]


def plot_ensemble_curves(conditions, output_dir):
    """
    One panel per condition (or overlay if <=3).
    Each panel shows:
      - Individual per-sample Cv(tau) curves (thin, semi-transparent)
      - Ensemble weighted-mean Cv(tau) (thick solid line)
      - Bootstrap 95% CI band (shaded)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = _condition_colors(len(conditions))

    for col, cond in zip(colors, conditions):
        tl    = cond['time_lags']
        ens   = cond['ens_Cv']
        ci_lo = cond['boot']['ci_lo']
        ci_hi = cond['boot']['ci_hi']

        # Per-sample curves (faint background)
        for r in cond['per_sample']:
            ax.plot(tl, r['Cv'], color=col, alpha=0.12, linewidth=0.8)

        # Ensemble mean
        ax.plot(tl, ens, color=col, linewidth=2.2,
                label=f"{cond['name']}  (n={cond['n_samples']} samples)")

        # CI band
        ax.fill_between(tl, ci_lo, ci_hi, color=col, alpha=0.18)

        # Mark tau_c
        if not np.isnan(cond['ens_tau_c']):
            ax.axvline(cond['ens_tau_c'], color=col, linestyle='--',
                       linewidth=1.0, alpha=0.7)

    ax.axhline(0,           color='k',    linestyle='-',  linewidth=0.6, alpha=0.4)
    ax.axhline(1.0 / np.e,  color='grey', linestyle=':',  linewidth=0.9, alpha=0.6,
               label='1/e threshold')

    ax.set_xlabel('Time lag  τ  (s)', fontsize=12)
    ax.set_ylabel(r'Velocity autocorrelation  $C_v(\tau)$', fontsize=12)
    ax.set_title('Ensemble Velocity Autocorrelation\n'
                 '(shaded = 95% bootstrap CI; faint lines = individual samples)',
                 fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_Cv_curves.png'), dpi=FIGURE_DPI)
    plt.close()
    print("  Saved: ensemble_Cv_curves.png")


def plot_per_sample_summary(conditions, stats_rows, output_dir):
    """
    Boxplot + strip (jitter) of per-sample tau_c and tau_zero per condition.
    Adds significance bars for pairwise significant comparisons.
    """
    metrics = [
        ('tau_c',   'tau_c_vals',  r'Correlation time  $\tau_c$  (s)',
         'Per-sample Correlation Times'),
        ('tau_zero', 'tau_z_vals', r'Zero-crossing time  $\tau_0$  (s)',
         'Per-sample Zero-Crossing Times'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors    = _condition_colors(len(conditions))
    rng_jit   = np.random.default_rng(0)

    for ax, (metric, key, ylabel, title) in zip(axes, metrics):
        x_positions = np.arange(len(conditions))
        valid_conds = []

        for xi, (col, cond) in enumerate(zip(colors, conditions)):
            vals = cond[key]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            valid_conds.append((xi, cond['name'], vals))

            # Box
            bp = ax.boxplot(vals, positions=[xi], widths=0.35,
                            patch_artist=True,
                            boxprops=dict(facecolor=col, alpha=0.35),
                            medianprops=dict(color='black', linewidth=1.5),
                            whiskerprops=dict(linewidth=1.0),
                            capprops=dict(linewidth=1.0),
                            flierprops=dict(marker='', linestyle='none'),
                            showfliers=False)

            # Jitter
            jitter = rng_jit.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals, color=col, s=28, alpha=0.75,
                       zorder=5, edgecolors='white', linewidths=0.4)

        # Significance bars
        if stats_rows:
            sig = [r for r in stats_rows
                   if r['metric'] == metric and r['significant']]
            y_top = max(
                (np.nanmax(cond[key]) for cond in conditions
                 if np.any(~np.isnan(cond[key]))),
                default=1.0
            )
            bar_step = y_top * 0.10
            for k, comp in enumerate(sig):
                names = [c['name'] for c in conditions]
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
                pstr = ('***' if pval < 0.001 else
                        '**'  if pval < 0.01  else '*')
                ax.text((xA + xB) / 2, ybar + bar_step * 0.05, pstr,
                        ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([c['name'] for c in conditions],
                           rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.25)

    plt.suptitle('Per-sample Summary Statistics\n'
                 '(each point = one embryo/movie; box = median + IQR)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_sample_summary.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: per_sample_summary.png")


def plot_bootstrap_distributions(conditions, output_dir):
    """
    Histogram of bootstrap tau_c distributions per condition.
    Shows how much sample-level variance contributes to ensemble uncertainty.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = _condition_colors(len(conditions))

    panels = [
        ('boot_tau_c', 'tau_c_ci',
         r'Bootstrap ensemble  $\tau_c$  (s)',
         r'Bootstrap distribution of ensemble $\tau_c$'),
        ('boot_tau_z', 'tau_zero_ci',
         r'Bootstrap ensemble  $\tau_0$  (s)',
         r'Bootstrap distribution of ensemble $\tau_0$'),
    ]

    for ax, (bkey, cikey, xlabel, title) in zip(axes, panels):
        for col, cond in zip(colors, conditions):
            arr = cond['boot'][bkey]
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            ci  = cond['boot'][cikey]
            ax.hist(arr, bins=30, color=col, alpha=0.45,
                    label=f"{cond['name']} ({len(arr)}/{N_BOOTSTRAP})")
            if not np.isnan(ci[0]):
                ax.axvline(ci[0], color=col, linestyle=':', linewidth=1.2)
                ax.axvline(ci[1], color=col, linestyle=':', linewidth=1.2)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Bootstrap replicates', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle('Bootstrap CI on ensemble summary statistics\n'
                 '(dotted lines = 95% CI boundaries)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_distributions.png'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: bootstrap_distributions.png")


# ─────────────────────────────────────────────────────────────
# CSV exports
# ─────────────────────────────────────────────────────────────

def export_ensemble_data(conditions, output_dir):
    rows = []
    for cond in conditions:
        tl    = cond['time_lags']
        ens   = cond['ens_Cv']
        se    = cond['ens_se']
        ci_lo = cond['boot']['ci_lo']
        ci_hi = cond['boot']['ci_hi']
        for k in range(len(tl)):
            rows.append({
                'condition': cond['name'],
                'time_lag_s': float(tl[k]),
                'ensemble_Cv': float(ens[k]) if not np.isnan(ens[k]) else '',
                'ensemble_SE': float(se[k])  if not np.isnan(se[k])  else '',
                'CI_lo_95':   float(ci_lo[k]) if not np.isnan(ci_lo[k]) else '',
                'CI_hi_95':   float(ci_hi[k]) if not np.isnan(ci_hi[k]) else '',
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
                'tau_c_s':    r['tau_c']   if not np.isnan(r['tau_c'])   else '',
                'tau_zero_s': r['tau_zero'] if not np.isnan(r['tau_zero']) else '',
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
    print("ENSEMBLE VELOCITY AUTOCORRELATION — SUMMARY")
    print("=" * 60)

    for cond in conditions:
        print(f"\n{cond['name']}  ({cond['n_samples']} samples)")
        print(f"  Ensemble tau_c  : {cond['ens_tau_c']:.3f} s"
              if not np.isnan(cond['ens_tau_c']) else
              "  Ensemble tau_c  : not found")
        tau_c_ci = cond['boot']['tau_c_ci']
        if not np.isnan(tau_c_ci[0]):
            print(f"  95% CI (boot)   : [{tau_c_ci[0]:.3f}, {tau_c_ci[1]:.3f}] s")
        print(f"  Ensemble tau_zero: {cond['ens_tau_z']:.3f} s"
              if not np.isnan(cond['ens_tau_z']) else
              "  Ensemble tau_zero: not found")
        tau_z_ci = cond['boot']['tau_zero_ci']
        if not np.isnan(tau_z_ci[0]):
            print(f"  95% CI (boot)   : [{tau_z_ci[0]:.3f}, {tau_z_ci[1]:.3f}] s")
        tc = cond['tau_c_vals'][~np.isnan(cond['tau_c_vals'])]
        if len(tc):
            print(f"  Per-sample tau_c: median={np.median(tc):.3f}s, "
                  f"IQR=[{np.percentile(tc,25):.3f}, {np.percentile(tc,75):.3f}]")

    if stats_rows:
        print("\nPairwise comparisons (per-sample tau_c / tau_zero):")
        for r in stats_rows:
            sig = "SIGNIFICANT" if r['significant'] else "n.s."
            print(f"  {r['condition_A']} vs {r['condition_B']}  [{r['metric']}]")
            print(f"    p={r['p_value']:.4e}  {sig}  "
                  f"Cliff's d={r['cliffs_delta']:.3f}  ({r['effect_size']})")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Ensemble Velocity Autocorrelation Analysis")
    print("=" * 45)
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
    output_dir = os.path.join(os.getcwd(), f"velocity_ensemble_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Statistics (only meaningful for >= 2 conditions)
    stats_rows = compare_conditions(conditions) if len(conditions) >= 2 else []

    # Plots
    print("Generating plots ...")
    plot_ensemble_curves(conditions, output_dir)
    plot_per_sample_summary(conditions, stats_rows, output_dir)
    plot_bootstrap_distributions(conditions, output_dir)

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

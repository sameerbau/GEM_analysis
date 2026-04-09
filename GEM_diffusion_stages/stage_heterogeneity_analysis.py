#!/usr/bin/env python3
"""
stage_heterogeneity_analysis.py

Tests whether between-cell GEM diffusion variability increases with
developmental stage in Drosophila embryos:

  Syncytial  →  Cellularization  →  MZT  →  Gastrulation

Biological hypothesis: in the syncytial blastoderm all nuclei share a
connected cytoplasm, so GEM diffusion environments are similar.  Once
cells individualise and differentiate, their cytoplasmic properties
diverge and between-cell variability in the diffusion coefficient D
should grow.

Input
-----
A manifest CSV with columns:
  stage               – one of the four canonical stages (see STAGE_ORDER)
  experiment_label    – unique name for the experiment
  diffusion_per_cell_csv  – path to the diffusion_per_cell.csv produced by
                            GEM_mitosis/2_diffusion_per_cell.py

Workflow
--------
  Step 1 (existing pipeline, run once per stage):
    python GEM_mitosis/1_cell_trajectory_classifier.py
    python GEM_mitosis/2_diffusion_per_cell.py
    -> produces diffusion_per_cell.csv for each stage experiment

  Step 2: edit stage_manifest.csv (copy stage_manifest_template.csv)

  Step 3: python GEM_diffusion_stages/stage_heterogeneity_analysis.py

Outputs → results/
  stage_pooled_cells.csv          – all cells with stage labels
  stage_variability_metrics.csv   – per-stage CV / SD / fold-range
  stage_stats_tests.csv           – overall variance tests + trend
  stage_pairwise_levene.csv       – pairwise Levene p-values
  stage_nested_anova.csv          – eta-squared decomposition
  fig1_violin_logD_per_stage.png  – main violin panel
  fig2_cv_per_stage.png           – CV and SD bar charts
  fig3_distribution_overlay.png   – KDE + ECDF overlays
  fig4_variance_decomposition.png – nested ANOVA stacked bar
  fig5_fold_range.png             – fold-range per stage
  stage_analysis_report.txt       – human-readable statistics report

References
----------
Hubatsch et al. (2023) Biophys J PMC10027447 – nested ANOVA approach,
fold-range metric, and the log-normal framework for diffusion variability.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path
from itertools import combinations

# ============================================================
# Configuration — edit these before running
# ============================================================
SCRIPT_DIR   = Path(__file__).parent
MANIFEST_CSV = SCRIPT_DIR / "stage_manifest.csv"
OUTPUT_DIR   = SCRIPT_DIR / "results"

# Stage order, labels and colours are derived automatically from the manifest
# (first-appearance order in the CSV sets the axis order).
# You can use ANY stage names — PB, NC, Cellularization, Gastrulation, etc.
# These globals are populated by _init_stage_config() at startup.
STAGE_ORDER   : list = []
STAGE_LABELS  : dict = {}
STAGE_COLOURS : dict = {}

# Colour palette (Nature journal style, cycles for >8 stages)
_STAGE_PALETTE = [
    "#4dbbd5", "#00a087", "#f39b7f", "#e64b35",
    "#3c5488", "#b09c85", "#7e6ebf", "#91d1c2",
]


def _init_stage_config(manifest_path: Path) -> None:
    """
    Read stage names from the manifest and populate STAGE_ORDER, STAGE_LABELS,
    STAGE_COLOURS.  Stage order = first-appearance order in the CSV, so you
    control the axis ordering simply by putting rows in the order you want.
    Any stage name is accepted (PB, NC, Cellularization, Gastrulation, …).
    """
    global STAGE_ORDER, STAGE_LABELS, STAGE_COLOURS
    df = pd.read_csv(manifest_path)
    if "stage" not in df.columns:
        raise ValueError("Manifest must have a 'stage' column.")
    seen: dict = {}
    for s in df["stage"].astype(str).str.strip():
        if s not in seen:
            seen[s] = len(seen)
    STAGE_ORDER  = sorted(seen, key=seen.__getitem__)
    # Display label: replace underscores/hyphens with spaces, title-case
    STAGE_LABELS = {s: s.replace("_", " ").replace("-", " ") for s in STAGE_ORDER}
    STAGE_COLOURS = {
        s: _STAGE_PALETTE[i % len(_STAGE_PALETTE)]
        for i, s in enumerate(STAGE_ORDER)
    }

# Quality filter: minimum valid trajectories per cell (matches GEM_mitosis default)
MIN_TRAJ_PER_CELL  = 5
# Warn (but do not abort) if a stage has fewer than this many cells
MIN_CELLS_PER_STAGE = 5

# Subtract per-experiment median log10(D) before pooling stages.
# This removes systematic batch offsets in absolute diffusivity while
# preserving each experiment's within-experiment spread.
NORMALIZE_WITHIN_EXPERIMENT = True

# Fold-range formula: 10^(FOLD_RANGE_N_SD * sigma_log10D)
# Following Hubatsch et al. 2023 Fig 2C (they use 5σ = 2 × 2.5σ)
FOLD_RANGE_N_SD = 5.0

# Bootstrap iterations for confidence intervals
N_BOOT = 1000

# Random seed for bootstrap and jitter reproducibility
SEED = 42

# Statistical significance threshold
ALPHA = 0.05

# Output figure DPI (matches existing pipeline scripts)
DPI = 300

# ============================================================
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Helper utilities
# (sig_stars and cliffs_delta copied from existing pipeline scripts)
# ============================================================

def sig_stars(p: float) -> str:
    """Return significance stars string.  Identical to GEM_mitosis helpers."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Non-parametric effect size (Cliff's delta) between arrays a and b.
    Adapted from 6Get_median_diffusion_v2.py calculate_effect_size().
    Returns value in [-1, 1]; magnitude guide:
      negligible <0.147, small <0.33, medium <0.474, large >=0.474
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = np.sum(a[:, None] > b[None, :])
    lesser  = np.sum(a[:, None] < b[None, :])
    return (greater - lesser) / (len(a) * len(b))


def interpret_cliffs_delta(d: float) -> str:
    ad = abs(d)
    if ad >= 0.474: return "large"
    if ad >= 0.330: return "medium"
    if ad >= 0.147: return "small"
    return "negligible"


def bootstrap_ci(values: np.ndarray, stat_fn,
                  n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap 95 % CI for stat_fn applied to values."""
    rng = np.random.default_rng(seed)
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


# ============================================================
# Data loading
# ============================================================

def load_manifest(path: Path) -> list:
    """
    Read the stage manifest CSV.
    Required columns: stage, experiment_label, diffusion_per_cell_csv
    Returns list of dicts with validated paths.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            f"Copy stage_manifest_template.csv to stage_manifest.csv "
            f"and fill in your experiment paths."
        )
    df = pd.read_csv(path)
    required = {"stage", "experiment_label", "diffusion_per_cell_csv"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        stage = str(row["stage"]).strip()
        csv_path = Path(str(row["diffusion_per_cell_csv"]).strip())
        if not csv_path.exists():
            raise FileNotFoundError(
                f"diffusion_per_cell.csv not found: {csv_path}\n"
                f"Run GEM_mitosis steps 1-2 first to generate this file."
            )
        rows.append({
            "stage":      stage,
            "exp_label":  str(row["experiment_label"]).strip(),
            "csv_path":   csv_path,
        })
    return rows


def load_experiment(csv_path: Path, exp_label: str,
                    min_traj: int = MIN_TRAJ_PER_CELL) -> pd.DataFrame:
    """
    Load a single diffusion_per_cell.csv.
    Adds 'experiment' column and filters by minimum trajectory count.
    """
    df = pd.read_csv(csv_path)
    if "n_valid_traj" not in df.columns:
        warnings.warn(f"'n_valid_traj' column missing in {csv_path}; "
                      "skipping trajectory-count filter.")
    else:
        df = df[df["n_valid_traj"] >= min_traj].copy()
    df["experiment"] = exp_label
    return df


def build_pooled_df(manifest: list,
                    normalize: bool = NORMALIZE_WITHIN_EXPERIMENT) -> pd.DataFrame:
    """
    Load all experiments, tag with stage, compute log10(D_median).
    Optionally subtract per-experiment median to remove batch offsets.
    Returns pooled DataFrame ready for analysis.
    """
    frames = []
    for row in manifest:
        df = load_experiment(row["csv_path"], row["exp_label"])
        df["stage"]           = row["stage"]
        df["stage_order_idx"] = STAGE_ORDER.index(row["stage"])
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)

    if "D_median" not in pooled.columns:
        raise ValueError("diffusion_per_cell.csv must contain a 'D_median' column.")

    pooled = pooled[pooled["D_median"] > 0].copy()
    pooled["log10D"] = np.log10(pooled["D_median"])

    if normalize:
        # Subtract per-experiment median log10D so only spread is compared
        exp_medians = pooled.groupby("experiment")["log10D"].median()
        pooled["log10D_norm"] = (
            pooled["log10D"] - pooled["experiment"].map(exp_medians)
        )
    else:
        pooled["log10D_norm"] = pooled["log10D"]

    # Unique cell identifier
    pooled["cell_uid"] = pooled["experiment"] + "_" + pooled["cell_label"].astype(str)

    return pooled


# ============================================================
# Statistical analysis
# ============================================================

def compute_stage_variability(df: pd.DataFrame,
                               value_col: str = "log10D_norm") -> pd.DataFrame:
    """
    Compute between-cell variability metrics per stage.

    Primary metric: SD of log10(D_median) across cells (log10D_SD).
    Also reports CV on the linear scale and fold range inspired by
    Hubatsch et al. 2023 Fig 2C: fold_range = 10^(FOLD_RANGE_N_SD * sigma).

    Returns one row per stage sorted in STAGE_ORDER.
    """
    rows = []
    rng  = np.random.default_rng(SEED)

    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        if len(sub) == 0:
            continue

        vals_log  = sub[value_col].dropna().values
        vals_lin  = sub["D_median"].dropna().values
        n_cells   = len(vals_log)
        n_exp     = sub["experiment"].nunique()

        # Core spread metrics
        log10D_mean   = np.mean(vals_log)
        log10D_median = np.median(vals_log)
        log10D_SD     = np.std(vals_log, ddof=1) if n_cells > 1 else np.nan
        log10D_IQR    = (np.percentile(vals_log, 75)
                         - np.percentile(vals_log, 25)) if n_cells > 1 else np.nan

        # CV in linear space: std(D_median) / mean(D_median)
        D_cv = (np.std(vals_lin, ddof=1) / np.mean(vals_lin) * 100
                if n_cells > 1 else np.nan)
        D_IQR = (np.percentile(vals_lin, 75)
                 - np.percentile(vals_lin, 25)) if n_cells > 1 else np.nan

        # Fold range following Hubatsch et al.
        fold_range = 10 ** (FOLD_RANGE_N_SD * log10D_SD) if not np.isnan(log10D_SD) else np.nan

        # 2.5th – 97.5th percentile range
        p025 = np.percentile(vals_log, 2.5)
        p975 = np.percentile(vals_log, 97.5)

        # Bootstrap CIs for SD and CV
        if n_cells >= 4:
            ci_sd = bootstrap_ci(vals_log, lambda x: np.std(x, ddof=1), N_BOOT, SEED)
            ci_cv = bootstrap_ci(vals_lin,
                                 lambda x: np.std(x, ddof=1) / np.mean(x) * 100,
                                 N_BOOT, SEED)
        else:
            ci_sd = (np.nan, np.nan)
            ci_cv = (np.nan, np.nan)

        rows.append({
            "stage":           stage,
            "stage_label":     STAGE_LABELS[stage],
            "stage_order_idx": STAGE_ORDER.index(stage),
            "n_cells":         n_cells,
            "n_experiments":   n_exp,
            "log10D_mean":     log10D_mean,
            "log10D_median":   log10D_median,
            "log10D_SD":       log10D_SD,
            "log10D_SD_ci_lo": ci_sd[0],
            "log10D_SD_ci_hi": ci_sd[1],
            "log10D_IQR":      log10D_IQR,
            "D_cv_pct":        D_cv,
            "D_cv_ci_lo":      ci_cv[0],
            "D_cv_ci_hi":      ci_cv[1],
            "D_IQR":           D_IQR,
            "fold_range":      fold_range,
            "log10D_p025":     p025,
            "log10D_p975":     p975,
        })

    return pd.DataFrame(rows).sort_values("stage_order_idx").reset_index(drop=True)


def run_variance_homogeneity_tests(df: pd.DataFrame,
                                    value_col: str = "log10D_norm") -> dict:
    """
    Overall and pairwise tests for equality of variance across stages.

    Overall:
      Levene's test    (center='mean')
      Brown-Forsythe   (center='median')  – more robust to non-normality
      Bartlett's test  (assumes normality) – reported for completeness

    Pairwise:
      Levene (BF variant) for all stage pairs, plus Cliff's delta on
      log10D distributions.
    """
    groups = {s: df.loc[df["stage"] == s, value_col].dropna().values
              for s in STAGE_ORDER if s in df["stage"].values}
    group_list = [v for v in groups.values() if len(v) >= 2]

    def _safe_test(fn, *args, **kwargs):
        try:
            s, p = fn(*args, **kwargs)
            return {"stat": s, "p": p, "sig": sig_stars(p)}
        except Exception as e:
            warnings.warn(f"Test failed: {e}")
            return {"stat": np.nan, "p": np.nan, "sig": "n/a"}

    results = {
        "levene_overall":    _safe_test(stats.levene, *group_list, center="mean"),
        "bf_overall":        _safe_test(stats.levene, *group_list, center="median"),
        "bartlett_overall":  _safe_test(stats.bartlett, *group_list),
        "kruskal_overall":   _safe_test(stats.kruskal, *group_list),
    }

    # Pairwise
    pairwise_rows = []
    stage_keys = [s for s in STAGE_ORDER if s in groups]
    for s1, s2 in combinations(stage_keys, 2):
        g1, g2 = groups[s1], groups[s2]
        if len(g1) < 2 or len(g2) < 2:
            continue
        bf_stat, bf_p = stats.levene(g1, g2, center="median")
        mw_stat, mw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        cd = cliffs_delta(g1, g2)
        pairwise_rows.append({
            "stage_A":        s1,
            "stage_B":        s2,
            "levene_BF_stat": bf_stat,
            "levene_BF_p":    bf_p,
            "levene_BF_sig":  sig_stars(bf_p),
            "mw_stat":        mw_stat,
            "mw_p":           mw_p,
            "mw_sig":         sig_stars(mw_p),
            "cliffs_delta":   cd,
            "effect_size":    interpret_cliffs_delta(cd),
            "n_A":            len(g1),
            "n_B":            len(g2),
        })

    results["pairwise"] = pd.DataFrame(pairwise_rows)
    return results


def run_spearman_trend_test(var_df: pd.DataFrame) -> dict:
    """
    Spearman correlation between stage ordinal index and between-cell
    variability metrics.  Tests the monotonicity hypothesis.
    """
    valid = var_df.dropna(subset=["log10D_SD", "D_cv_pct"])
    if len(valid) < 3:
        return {"rho_SD": np.nan, "p_SD": np.nan,
                "rho_CV": np.nan, "p_CV": np.nan,
                "interpretation": "insufficient data"}

    rho_sd, p_sd = stats.spearmanr(valid["stage_order_idx"], valid["log10D_SD"])
    rho_cv, p_cv = stats.spearmanr(valid["stage_order_idx"], valid["D_cv_pct"])

    if p_sd < ALPHA and rho_sd > 0:
        interp = "heterogeneity increases monotonically with stage"
    elif p_sd < ALPHA and rho_sd < 0:
        interp = "heterogeneity decreases monotonically with stage"
    else:
        interp = "no significant monotonic trend detected"

    return {"rho_SD": rho_sd, "p_SD": p_sd,
            "rho_CV": rho_cv, "p_CV": p_cv,
            "interpretation": interp}


def nested_anova_with_stage(df: pd.DataFrame,
                              value_col: str = "log10D_norm") -> tuple:
    """
    Nested ANOVA: stage → experiment → cell

    Adapted from GEM_mitosis/7_variance_decomposition.py nested_anova()
    to include stage as the top hierarchical level.

    NOTE: Since input is diffusion_per_cell.csv (one D_median per cell),
    the within-cell track-to-track component is not recomputed here.
    That component (~79% in Hubatsch et al.) is already absorbed into
    each cell's D_median estimate.

    Returns (component_df, full_results_dict)
    """
    work = df[["cell_uid", "experiment", "stage",
               "stage_order_idx", value_col]].dropna().copy()
    work.columns = ["cell_uid", "experiment", "stage",
                    "stage_order_idx", "val"]

    grand_mean = work["val"].mean()
    SS_total   = ((work["val"] - grand_mean) ** 2).sum()
    df_total   = len(work) - 1

    # Between-stage
    smeans = work.groupby("stage")["val"].mean()
    sn     = work.groupby("stage")["val"].count()
    SS_stage  = ((smeans - grand_mean) ** 2 * sn).sum()
    df_stage  = len(smeans) - 1

    # Between-experiment within stage
    emeans = work.groupby(["stage", "experiment"])["val"].mean()
    en     = work.groupby(["stage", "experiment"])["val"].count()
    SS_exp = 0.0
    for (stg, exp), n in en.items():
        SS_exp += n * (emeans[(stg, exp)] - smeans[stg]) ** 2
    df_exp = len(emeans) - len(smeans)

    # Between-cell within experiment
    cmeans = work.groupby("cell_uid")["val"].first()    # one value per cell
    cell_exp = work.drop_duplicates("cell_uid").set_index("cell_uid")["experiment"]
    cell_stg = work.drop_duplicates("cell_uid").set_index("cell_uid")["stage"]
    SS_cell = 0.0
    for uid, val in cmeans.items():
        exp = cell_exp[uid]
        stg = cell_stg[uid]
        SS_cell += (val - emeans[(stg, exp)]) ** 2
    df_cell = len(cmeans) - len(emeans)

    eta2_stage = SS_stage / SS_total if SS_total > 0 else np.nan
    eta2_exp   = SS_exp   / SS_total if SS_total > 0 else np.nan
    eta2_cell  = SS_cell  / SS_total if SS_total > 0 else np.nan

    components = pd.DataFrame([
        {"component": "Between-stage",
         "SS": SS_stage, "df": df_stage,
         "eta2": eta2_stage, "pct": eta2_stage * 100},
        {"component": "Between-experiment\n(within stage)",
         "SS": SS_exp,   "df": df_exp,
         "eta2": eta2_exp,   "pct": eta2_exp * 100},
        {"component": "Between-cell\n(within experiment)",
         "SS": SS_cell,  "df": df_cell,
         "eta2": eta2_cell,  "pct": eta2_cell * 100},
    ])

    full = dict(grand_mean=grand_mean, SS_total=SS_total,
                N=len(work), N_cells=len(cmeans),
                N_experiments=work["experiment"].nunique(),
                N_stages=work["stage"].nunique(),
                eta2_stage=eta2_stage, eta2_exp=eta2_exp, eta2_cell=eta2_cell)

    # Per-stage between-cell SD (for Fig 4 panel B)
    per_stage_bc = {}
    for stg in STAGE_ORDER:
        sub = work[work["stage"] == stg]
        if len(sub) < 2:
            per_stage_bc[stg] = np.nan
            continue
        cell_vals = sub.groupby("cell_uid")["val"].first()
        per_stage_bc[stg] = cell_vals.std(ddof=1)

    full["per_stage_between_cell_SD"] = per_stage_bc

    return components, full


# ============================================================
# Figures
# ============================================================

def _stage_positions(stages_present):
    """Map stage names to x-axis integer positions in canonical order."""
    return {s: i for i, s in enumerate(STAGE_ORDER) if s in stages_present}


def plot_violin_panel(df: pd.DataFrame, var_df: pd.DataFrame,
                       pairwise_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 1: Violin + jitter of log10(D_median) per cell, ordered by stage.
    Pairwise significance bars for adjacent stages from Levene BF test.
    """
    stages   = [s for s in STAGE_ORDER if s in df["stage"].values]
    n_stages = len(stages)
    positions = list(range(n_stages))

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_stages), 6), dpi=DPI)

    groups = [df.loc[df["stage"] == s, "log10D_norm"].dropna().values
              for s in stages]

    parts = ax.violinplot(groups, positions=positions,
                          showmedians=True, showextrema=False, widths=0.65)

    for pc, stg in zip(parts["bodies"], stages):
        pc.set_facecolor(STAGE_COLOURS[stg])
        pc.set_alpha(0.60)
        pc.set_edgecolor("none")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2.0)

    rng = np.random.default_rng(SEED)
    for pos, vals, stg in zip(positions, groups, stages):
        jit = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jit, vals, s=22, color=STAGE_COLOURS[stg],
                   alpha=0.80, edgecolors="white", linewidths=0.3, zorder=3)

    # Pairwise significance bars for adjacent stages
    y_max = max(np.nanmax(g) for g in groups if len(g) > 0)
    bar_step = (y_max - ax.get_ylim()[0]) * 0.07 if ax.get_ylim()[0] != 0 else 0.15
    # Recalculate after scatter
    y_top = np.nanmax([np.nanmax(g) for g in groups]) * 1.05

    if len(pairwise_df) > 0:
        drawn = 0
        for i in range(len(stages) - 1):
            s1, s2 = stages[i], stages[i + 1]
            row = pairwise_df[
                ((pairwise_df["stage_A"] == s1) & (pairwise_df["stage_B"] == s2)) |
                ((pairwise_df["stage_A"] == s2) & (pairwise_df["stage_B"] == s1))
            ]
            if len(row) == 0:
                continue
            p = row["levene_BF_p"].values[0]
            sig = sig_stars(p)
            if sig == "ns":
                continue
            y_bar = y_top + bar_step * (drawn + 1)
            ax.plot([i, i + 1], [y_bar, y_bar], "k-", lw=1.0)
            ax.text((i + i + 1) / 2, y_bar * 1.005, sig,
                    ha="center", va="bottom", fontsize=11)
            drawn += 1

    labels = []
    for s in stages:
        n = int(var_df.loc[var_df["stage"] == s, "n_cells"].values[0]
                if s in var_df["stage"].values else 0)
        labels.append(f"{STAGE_LABELS[s]}\n(n={n})")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    y_label = ("Normalised log\u2081\u2080(D\u2098\u2091\u2093) [a.u.]"
               if NORMALIZE_WITHIN_EXPERIMENT
               else "log\u2081\u2080(D\u2098\u2091\u2093) [\u03bcm\u00b2/s]")
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        "GEM diffusion coefficient distributions across developmental stages\n"
        "(per-cell median D, log\u2081\u2080 scale; bars = Brown-Forsythe variance test)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_cv_panel(var_df: pd.DataFrame, trend: dict,
                   out_path: Path) -> None:
    """
    Figure 2: Two-panel bar chart.
    Left: CV (%) of D_median across cells per stage.
    Right: SD of log10(D_median) across cells per stage.
    Both with bootstrap 95% CI error bars and Spearman trend annotation.
    """
    valid = var_df.dropna(subset=["log10D_SD"]).copy()
    if len(valid) == 0:
        print("  Skipping fig2: no valid variability data.")
        return

    x    = np.arange(len(valid))
    cols = [STAGE_COLOURS[s] for s in valid["stage"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)

    for ax, (val_col, err_lo, err_hi, ylabel, title) in zip(axes, [
        ("D_cv_pct", "D_cv_ci_lo", "D_cv_ci_hi",
         "CV of D\u2098\u2091\u2093 across cells (%)",
         "Coefficient of variation"),
        ("log10D_SD", "log10D_SD_ci_lo", "log10D_SD_ci_hi",
         "SD of log\u2081\u2080(D\u2098\u2091\u2093) across cells",
         "Log-scale spread"),
    ]):
        vals = valid[val_col].values
        lo   = vals - valid[err_lo].values
        hi   = valid[err_hi].values - vals
        lo   = np.clip(lo, 0, None)
        hi   = np.clip(hi, 0, None)

        bars = ax.bar(x, vals, color=cols, alpha=0.85,
                      edgecolor="white", width=0.6, zorder=2)
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="none",
                    color="black", capsize=5, lw=1.5, zorder=3)

        for xi, (val, row) in enumerate(zip(vals, valid.itertuples())):
            ax.text(xi, val + hi[xi] + 0.005,
                    f"n={int(row.n_cells)}", ha="center",
                    va="bottom", fontsize=8, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LABELS[s] for s in valid["stage"]],
                           fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, None)

    # Shared Spearman annotation
    rho = trend.get("rho_SD", np.nan)
    p_t = trend.get("p_SD", np.nan)
    if not np.isnan(rho):
        fig.text(0.5, -0.02,
                 f"Spearman \u03c1 (stage order vs SD) = {rho:.2f}, "
                 f"p = {p_t:.4f} {sig_stars(p_t)}",
                 ha="center", fontsize=11)

    plt.suptitle(
        "Between-cell diffusion variability across developmental stages\n"
        "(bootstrap 95% CI error bars)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_distribution_overlay(df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 3: KDE overlay (left) and ECDF overlay (right) of log10(D_median)
    per stage, coloured by stage.
    """
    from scipy.stats import gaussian_kde

    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)

    all_vals = df["log10D_norm"].dropna().values
    x_lo = np.percentile(all_vals, 1)
    x_hi = np.percentile(all_vals, 99)
    x_grid = np.linspace(x_lo - 0.3, x_hi + 0.3, 300)

    ax_kde, ax_cdf = axes

    for stg in stages:
        vals = df.loc[df["stage"] == stg, "log10D_norm"].dropna().values
        col  = STAGE_COLOURS[stg]
        lbl  = f"{STAGE_LABELS[stg]}  (n={len(vals)})"

        if len(vals) >= 4:
            kde = gaussian_kde(vals)
            ax_kde.plot(x_grid, kde(x_grid), color=col, lw=2.5, label=lbl)
            med = np.median(vals)
            ax_kde.axvline(med, color=col, lw=1.2, ls="--", alpha=0.7)

        # ECDF
        sorted_v = np.sort(vals)
        ecdf     = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax_cdf.step(sorted_v, ecdf, where="post", color=col, lw=2.5, label=lbl)
        ax_cdf.axvline(np.median(vals), color=col, lw=1.2, ls="--", alpha=0.7)

    x_label = ("Normalised log\u2081\u2080(D) [a.u.]"
               if NORMALIZE_WITHIN_EXPERIMENT
               else "log\u2081\u2080(D\u2098\u2091\u2093) [\u03bcm\u00b2/s]")
    for ax in axes:
        ax.set_xlabel(x_label, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.set_xlim(x_lo - 0.3, x_hi + 0.3)

    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.set_title("KDE of per-cell log\u2081\u2080(D)\nby developmental stage", fontsize=11)
    ax_cdf.set_ylabel("Cumulative fraction", fontsize=11)
    ax_cdf.set_title("ECDF of per-cell log\u2081\u2080(D)\nby developmental stage", fontsize=11)
    ax_cdf.set_ylim(0, 1.02)

    plt.suptitle(
        "Distribution of GEM diffusion coefficients across developmental stages\n"
        "(dashed lines = stage medians)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_variance_decomposition(anova_components: pd.DataFrame,
                                 anova_full: dict,
                                 out_path: Path) -> None:
    """
    Figure 4: Two-panel nested ANOVA figure.
    Panel A: stacked eta^2 bar (overall decomposition).
    Panel B: between-cell SD per stage (shows growing heterogeneity).
    """
    bar_colours = ["#e64b35", "#8491b4", "#91d1c2"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)

    # --- Panel A: stacked eta^2 ---
    ax = axes[0]
    eta2_vals = anova_components["eta2"].values
    comp_lbls = anova_components["component"].values

    bottom = 0.0
    for j, (comp, col, val) in enumerate(zip(comp_lbls, bar_colours, eta2_vals)):
        if np.isnan(val):
            continue
        ax.bar(0, val, bottom=bottom, color=col, edgecolor="white",
               width=0.5, label=comp)
        if val > 0.04:
            ax.text(0, bottom + val / 2,
                    f"{val * 100:.0f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        bottom += val

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([0])
    ax.set_xticklabels(["All cells"], fontsize=11)
    ax.set_ylabel("\u03b7\u00b2 (fraction of total log\u2081\u2080D variance)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        "Nested ANOVA decomposition\n"
        f"(N={anova_full['N_cells']} cells, "
        f"{anova_full['N_experiments']} exp, "
        f"{anova_full['N_stages']} stages)",
        fontsize=10,
    )

    # --- Panel B: between-cell SD per stage ---
    ax2 = axes[1]
    bc_sd = anova_full["per_stage_between_cell_SD"]
    present = [s for s in STAGE_ORDER if s in bc_sd and not np.isnan(bc_sd[s])]
    x      = np.arange(len(present))
    vals_b = [bc_sd[s] for s in present]
    cols_b = [STAGE_COLOURS[s] for s in present]

    ax2.bar(x, vals_b, color=cols_b, alpha=0.85, edgecolor="white",
            width=0.6, zorder=2)
    for xi, (stg, val) in enumerate(zip(present, vals_b)):
        ax2.text(xi, val + 0.003, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([STAGE_LABELS[s] for s in present], fontsize=11)
    ax2.set_ylabel("Between-cell SD of log\u2081\u2080(D)", fontsize=11)
    ax2.set_ylim(0, None)
    ax2.set_title("Cell-to-cell variability per stage\n"
                  "(SD of cell-mean log\u2081\u2080D within each stage)",
                  fontsize=10)

    plt.suptitle(
        "Nested ANOVA variance decomposition of GEM diffusion heterogeneity\n"
        "(hierarchy: stage \u2192 experiment \u2192 cell)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_fold_range(var_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 5: Fold range of D per stage.
    fold_range = 10^(FOLD_RANGE_N_SD * sigma_log10D)
    following Hubatsch et al. 2023.
    """
    valid = var_df.dropna(subset=["fold_range"]).copy()
    if len(valid) == 0:
        print("  Skipping fig5: no valid fold-range data.")
        return

    x    = np.arange(len(valid))
    cols = [STAGE_COLOURS[s] for s in valid["stage"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)

    bars = ax.bar(x, valid["fold_range"], color=cols, alpha=0.85,
                  edgecolor="white", width=0.55, zorder=2)

    for xi, (val, row) in enumerate(zip(valid["fold_range"], valid.itertuples())):
        ax.text(xi, val + 0.3, f"{val:.1f}\u00d7",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="grey", lw=1.2, ls="--", alpha=0.6,
               label="no heterogeneity (fold range = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[s] for s in valid["stage"]], fontsize=12)
    ax.set_ylabel(
        f"Fold range of D\u2098\u2091\u2093  "
        f"[10^({FOLD_RANGE_N_SD:.0f}\u03c3)]",
        fontsize=11,
    )
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)
    ax.set_title(
        "Cell-to-cell fold range of GEM diffusion coefficient\n"
        f"(fold range = 10^({FOLD_RANGE_N_SD:.0f}\u03c3), following Hubatsch et al. 2023)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ============================================================
# Text report
# ============================================================

def write_text_report(var_df: pd.DataFrame,
                       test_results: dict,
                       trend: dict,
                       anova_components: pd.DataFrame,
                       anova_full: dict,
                       df: pd.DataFrame,
                       out_path: Path) -> None:
    """Write human-readable statistics report matching existing pipeline style."""

    with open(out_path, "w") as fh:
        W = lambda s="": fh.write(s + "\n")

        W("=" * 65)
        W("Stage Heterogeneity Analysis Report")
        W("GEM diffusion variability across developmental stages")
        W("(following Hubatsch et al. 2023, Biophys J PMC10027447)")
        W("=" * 65)
        W()
        W("Dataset summary")
        W("-" * 40)
        W(f"  Total cells:       {len(df):,}")
        W(f"  Total experiments: {df['experiment'].nunique()}")
        W(f"  Stages analysed:   {', '.join(s for s in STAGE_ORDER if s in df['stage'].values)}")
        W(f"  Within-exp normalisation: {NORMALIZE_WITHIN_EXPERIMENT}")
        W(f"  Fold-range formula:       10^({FOLD_RANGE_N_SD}*sigma_log10D)")
        W()

        W("Per-stage variability summary")
        W("-" * 40)
        for _, row in var_df.iterrows():
            W(f"  [{STAGE_LABELS[row['stage']]}]  "
              f"n_cells={int(row['n_cells'])}  n_exp={int(row['n_experiments'])}")
            W(f"    log10D mean  = {row['log10D_mean']:.4f}")
            W(f"    log10D SD    = {row['log10D_SD']:.4f}  "
              f"(95% CI: [{row['log10D_SD_ci_lo']:.4f}, {row['log10D_SD_ci_hi']:.4f}])")
            W(f"    CV (linear)  = {row['D_cv_pct']:.2f}%")
            W(f"    Fold range   = {row['fold_range']:.2f}x")
            W()

        W("Variance homogeneity tests (overall)")
        W("-" * 40)
        for name, key in [
            ("Levene (mean-centred)",   "levene_overall"),
            ("Brown-Forsythe (median)", "bf_overall"),
            ("Bartlett (assumes normal)","bartlett_overall"),
            ("Kruskal-Wallis",          "kruskal_overall"),
        ]:
            r = test_results.get(key, {})
            W(f"  {name:38s}  "
              f"stat={r.get('stat', float('nan')):.4f}  "
              f"p={r.get('p', float('nan')):.4e}  "
              f"[{r.get('sig', 'n/a')}]")
        W()

        W("Monotonic trend test (Spearman)")
        W("-" * 40)
        W(f"  stage order vs log10D SD:  "
          f"rho={trend.get('rho_SD', float('nan')):.3f}  "
          f"p={trend.get('p_SD', float('nan')):.4f}  "
          f"[{sig_stars(trend.get('p_SD', 1.0))}]")
        W(f"  stage order vs CV:         "
          f"rho={trend.get('rho_CV', float('nan')):.3f}  "
          f"p={trend.get('p_CV', float('nan')):.4f}  "
          f"[{sig_stars(trend.get('p_CV', 1.0))}]")
        W(f"  Interpretation: {trend.get('interpretation', 'n/a')}")
        W()

        W("Pairwise Levene (Brown-Forsythe) tests")
        W("-" * 40)
        pw = test_results.get("pairwise", pd.DataFrame())
        if len(pw) > 0:
            for _, r in pw.iterrows():
                W(f"  {STAGE_LABELS[r['stage_A']]:15s} vs "
                  f"{STAGE_LABELS[r['stage_B']]:15s}  "
                  f"F={r['levene_BF_stat']:.3f}  "
                  f"p={r['levene_BF_p']:.4e}  [{r['levene_BF_sig']}]  "
                  f"Cliff d={r['cliffs_delta']:.3f} ({r['effect_size']})")
        else:
            W("  (insufficient data for pairwise tests)")
        W()

        W("Nested ANOVA decomposition (stage -> experiment -> cell)")
        W("-" * 40)
        W(f"  N={anova_full['N_cells']} cells  |  "
          f"{anova_full['N_experiments']} experiments  |  "
          f"{anova_full['N_stages']} stages")
        W(f"  Grand mean log10D = {anova_full['grand_mean']:.4f}")
        W()
        for _, r in anova_components.iterrows():
            comp = r['component'].replace('\n', ' ')
            W(f"  {comp:45s}  "
              f"SS={r['SS']:.4f}  eta2={r['eta2']:.4f}  ({r['pct']:.1f}%)")
        W()
        W("  NOTE: Within-cell (track-to-track) variance is not recomputed")
        W("  here; it is already folded into each cell's D_median estimate.")
        W("  See GEM_mitosis/7_variance_decomposition.py for within-cell")
        W("  decomposition (~79% in Hubatsch et al. 2023).")
        W()
        W("=" * 65)

    print(f"  Saved {out_path.name}")


# ============================================================
# Main execution
# ============================================================

print("=" * 60)
print("Stage Heterogeneity Analysis")
print("GEM diffusion variability across developmental stages")
print("=" * 60 + "\n")

# ------------------------------------------------------------------
# 0. Initialise stage config from manifest (must happen before anything
#    that references STAGE_ORDER / STAGE_LABELS / STAGE_COLOURS)
# ------------------------------------------------------------------
_init_stage_config(MANIFEST_CSV)
print(f"  Stage order (from manifest): {' → '.join(STAGE_ORDER)}\n")

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading data from manifest …")
manifest = load_manifest(MANIFEST_CSV)
print(f"  {len(manifest)} experiment(s) found across "
      f"{len(set(r['stage'] for r in manifest))} stage(s).")

df = build_pooled_df(manifest)
print(f"  Pooled: {len(df):,} cells  |  "
      f"{df['experiment'].nunique()} experiments  |  "
      f"{df['stage'].nunique()} stages")

# Validate minimum cell counts
for stg in STAGE_ORDER:
    n = (df["stage"] == stg).sum()
    if 0 < n < MIN_CELLS_PER_STAGE:
        warnings.warn(
            f"Stage '{stg}' has only {n} cells "
            f"(minimum recommended: {MIN_CELLS_PER_STAGE}). "
            "Results may be unreliable."
        )
    if stg in df["stage"].values:
        print(f"    {STAGE_LABELS[stg]:18s}  {n:4d} cells  "
              f"{df.loc[df['stage']==stg,'experiment'].nunique()} experiment(s)")

# Save pooled cells for auditing / downstream use
pool_out = OUTPUT_DIR / "stage_pooled_cells.csv"
df.to_csv(pool_out, index=False)
print(f"\n  Saved stage_pooled_cells.csv ({len(df)} rows)\n")

# ------------------------------------------------------------------
# 2. Compute variability metrics
# ------------------------------------------------------------------
print("Computing between-cell variability metrics …")
var_df = compute_stage_variability(df)
var_df.to_csv(OUTPUT_DIR / "stage_variability_metrics.csv", index=False)
print("  Saved stage_variability_metrics.csv")
for _, row in var_df.iterrows():
    print(f"  [{STAGE_LABELS[row['stage']]}]  "
          f"SD={row['log10D_SD']:.4f}  "
          f"CV={row['D_cv_pct']:.1f}%  "
          f"fold_range={row['fold_range']:.2f}x  "
          f"(n={int(row['n_cells'])})")

# ------------------------------------------------------------------
# 3. Statistical tests
# ------------------------------------------------------------------
print("\nRunning variance homogeneity tests …")
test_results = run_variance_homogeneity_tests(df)
for name, key in [("Levene", "levene_overall"),
                   ("Brown-Forsythe", "bf_overall"),
                   ("Bartlett", "bartlett_overall"),
                   ("Kruskal-Wallis", "kruskal_overall")]:
    r = test_results[key]
    print(f"  {name:20s}  p={r['p']:.4e}  [{r['sig']}]")

pairwise_df = test_results["pairwise"]
if len(pairwise_df) > 0:
    pairwise_df.to_csv(OUTPUT_DIR / "stage_pairwise_levene.csv", index=False)
    print("  Saved stage_pairwise_levene.csv")

print("\nMonotonic trend test …")
trend = run_spearman_trend_test(var_df)
print(f"  Spearman rho (stage order vs SD) = {trend['rho_SD']:.3f}  "
      f"p={trend['p_SD']:.4f}  [{sig_stars(trend['p_SD'])}]")
print(f"  {trend['interpretation']}")

# Save combined tests
all_tests = []
for name, key in [("Levene_overall", "levene_overall"),
                   ("BrownForsythe_overall", "bf_overall"),
                   ("Bartlett_overall", "bartlett_overall"),
                   ("KruskalWallis_overall", "kruskal_overall")]:
    r = test_results[key]
    all_tests.append({"test": name, "stat": r["stat"], "p": r["p"], "sig": r["sig"]})
all_tests.append({"test": "Spearman_trend_SD",
                   "stat": trend["rho_SD"], "p": trend["p_SD"],
                   "sig": sig_stars(trend["p_SD"])})
all_tests.append({"test": "Spearman_trend_CV",
                   "stat": trend["rho_CV"], "p": trend["p_CV"],
                   "sig": sig_stars(trend["p_CV"])})
pd.DataFrame(all_tests).to_csv(OUTPUT_DIR / "stage_stats_tests.csv", index=False)
print("  Saved stage_stats_tests.csv")

# ------------------------------------------------------------------
# 4. Nested ANOVA
# ------------------------------------------------------------------
print("\nRunning nested ANOVA (stage -> experiment -> cell) …")
anova_components, anova_full = nested_anova_with_stage(df)
anova_components.to_csv(OUTPUT_DIR / "stage_nested_anova.csv", index=False)
print("  Saved stage_nested_anova.csv")
for _, row in anova_components.iterrows():
    comp = row['component'].replace('\n', ' ')
    print(f"  {comp:45s}  eta2={row['eta2']:.3f}  ({row['pct']:.1f}%)")

# ------------------------------------------------------------------
# 5. Figures
# ------------------------------------------------------------------
print("\nGenerating figures …")
plot_violin_panel(df, var_df, pairwise_df,
                  OUTPUT_DIR / "fig1_violin_logD_per_stage.png")
plot_cv_panel(var_df, trend,
              OUTPUT_DIR / "fig2_cv_per_stage.png")
plot_distribution_overlay(df,
                           OUTPUT_DIR / "fig3_distribution_overlay.png")
plot_variance_decomposition(anova_components, anova_full,
                             OUTPUT_DIR / "fig4_variance_decomposition.png")
plot_fold_range(var_df,
                OUTPUT_DIR / "fig5_fold_range.png")

# ------------------------------------------------------------------
# 6. Text report
# ------------------------------------------------------------------
print("\nWriting statistics report …")
write_text_report(var_df, test_results, trend,
                  anova_components, anova_full, df,
                  OUTPUT_DIR / "stage_analysis_report.txt")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE HETEROGENEITY ANALYSIS — COMPLETE")
print("=" * 60)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nKey result:")
for _, row in var_df.iterrows():
    print(f"  {STAGE_LABELS[row['stage']]:18s}  "
          f"SD={row['log10D_SD']:.4f}  "
          f"fold_range={row['fold_range']:.2f}x")
print(f"\nMonotonic trend: rho={trend['rho_SD']:.3f}  "
      f"p={trend['p_SD']:.4f}  [{sig_stars(trend['p_SD'])}]")
print(f"Interpretation:  {trend['interpretation']}")
print("\nDone.")

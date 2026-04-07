#!/usr/bin/env python3
"""
4_pool_experiments.py

Step 4 of the mitosis GEM analysis pipeline.

Pools per-cell diffusion data from multiple experiments (each processed
through Steps 1-3) and re-runs the mitotic vs interphase comparison and
between-mitotic-cell analysis on the combined dataset.

Each experiment contributes one `diffusion_per_cell.csv` produced by
Step 2.  You can also supply pre-classified CSVs from Step 3
(`cell_classification.csv`) — if present, the existing cell_type labels
are used; otherwise the circularity threshold is applied fresh.

Directory layout expected (one sub-folder per experiment):
  experiments/
    exp1/
        diffusion_per_cell.csv          (required)
        cell_classification.csv         (optional – uses Step 3 labels if present)
    exp2/
        diffusion_per_cell.csv
        ...

OR you can list CSV paths explicitly in EXPERIMENT_CSVS below.

Outputs (written to OUTPUT_DIR):
  pooled_cells.csv                    – all cells with experiment + cell_type labels
  pooled_correlation_diffusion.png    – D vs circularity & area (all experiments)
  pooled_mitotic_vs_interphase.png    – D comparison: mitotic vs interphase
  pooled_mitotic_cell_comparison.png  – D per individual mitotic cell
  pooled_stats.txt                    – full statistical report

Usage:
  python 4_pool_experiments.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.mixture import GaussianMixture
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit this section to add your experiments
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

# Option A: auto-discover — point to a parent folder that contains one
# sub-folder per experiment, each with a diffusion_per_cell.csv
EXPERIMENTS_ROOT = SCRIPT_DIR   # change to e.g. Path("/data/mitosis_experiments")

# Option B: explicit list of (label, path_to_diffusion_per_cell.csv) tuples.
# Leave empty ([]) to use Option A auto-discovery.
EXPERIMENT_CSVS = [
    # ("Experiment 1", SCRIPT_DIR / "diffusion_per_cell.csv"),
    # ("Experiment 2", Path("/path/to/exp2/diffusion_per_cell.csv")),
]

# Circularity threshold for mitotic classification.
# Set to None to fit a fresh GMM on the pooled circularity distribution.
MANUAL_CIRCULARITY_THRESHOLD = 0.70

# Minimum valid trajectories per cell (applied after pooling)
MIN_TRAJ_PER_CELL = 5

# Minimum per-trajectory D values a cell needs to enter the
# between-mitotic-cell comparison
MIN_TRAJ_MITOTIC = 5

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "pooled_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Discover experiments
# ---------------------------------------------------------------------------
def find_experiments():
    """Return list of (label, csv_path) tuples."""
    if EXPERIMENT_CSVS:
        return [(lbl, Path(p)) for lbl, p in EXPERIMENT_CSVS]

    results = []
    # Search immediate sub-directories for diffusion_per_cell.csv
    for sub in sorted(EXPERIMENTS_ROOT.iterdir()):
        if sub.is_dir():
            csv = sub / "diffusion_per_cell.csv"
            if csv.exists():
                results.append((sub.name, csv))

    # Also check the root itself (single-experiment case: current folder)
    root_csv = EXPERIMENTS_ROOT / "diffusion_per_cell.csv"
    if root_csv.exists() and not any(p == root_csv for _, p in results):
        results.append(("set1_3em_001", root_csv))

    return results


experiments = find_experiments()
if not experiments:
    raise FileNotFoundError(
        "No diffusion_per_cell.csv files found.\n"
        "Either set EXPERIMENT_CSVS explicitly or organise experiments into\n"
        "sub-folders under EXPERIMENTS_ROOT."
    )

print(f"Found {len(experiments)} experiment(s):")
for lbl, p in experiments:
    print(f"  [{lbl}]  {p}")

# ---------------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------------
frames = []
for exp_label, csv_path in experiments:
    df_exp = pd.read_csv(csv_path)

    # Try to load Step 3 cell_type labels if available
    classif_csv = csv_path.parent / "cell_classification.csv"
    if classif_csv.exists():
        df_classif = pd.read_csv(classif_csv)[["cell_label", "cell_type"]]
        df_exp = df_exp.merge(df_classif, on="cell_label", how="left")
        print(f"  [{exp_label}] loaded {len(df_exp)} cells (with Step 3 labels)")
    else:
        print(f"  [{exp_label}] loaded {len(df_exp)} cells (no Step 3 labels — will classify)")

    df_exp["experiment"] = exp_label
    # Make cell IDs unique across experiments
    df_exp["cell_uid"] = exp_label + "_" + df_exp["cell_label"].astype(str)
    frames.append(df_exp)

df_pool = pd.concat(frames, ignore_index=True)
df_pool = df_pool[df_pool["n_valid_traj"] >= MIN_TRAJ_PER_CELL].copy()
print(f"\nPooled dataset: {len(df_pool)} cells from {df_pool['experiment'].nunique()} experiment(s)")

# ---------------------------------------------------------------------------
# Circularity classification (re-run if any cells lack cell_type)
# ---------------------------------------------------------------------------
need_classif = df_pool["cell_type"].isna() if "cell_type" in df_pool.columns \
               else pd.Series(True, index=df_pool.index)

if need_classif.any() or "cell_type" not in df_pool.columns:
    circ_vals = df_pool["circularity"].dropna().values

    if MANUAL_CIRCULARITY_THRESHOLD is not None:
        threshold = MANUAL_CIRCULARITY_THRESHOLD
        print(f"Using manual circularity threshold: {threshold:.3f}")
    else:
        gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
        gmm.fit(circ_vals.reshape(-1, 1))
        means = gmm.means_.flatten()
        mitotic_comp = int(np.argmax(means))
        mean_low  = means[1 - mitotic_comp]
        mean_high = means[mitotic_comp]
        std_low   = np.sqrt(gmm.covariances_.flatten()[1 - mitotic_comp])
        std_high  = np.sqrt(gmm.covariances_.flatten()[mitotic_comp])
        w_low     = gmm.weights_[1 - mitotic_comp]
        w_high    = gmm.weights_[mitotic_comp]
        x_scan = np.linspace(mean_low, mean_high, 10000)
        pdf_low  = w_low  * stats.norm.pdf(x_scan, mean_low,  std_low)
        pdf_high = w_high * stats.norm.pdf(x_scan, mean_high, std_high)
        threshold = x_scan[np.argmin(np.abs(pdf_low - pdf_high))]
        print(f"GMM circularity threshold: {threshold:.3f}  (means: {mean_low:.3f}, {mean_high:.3f})")

    df_pool["cell_type"] = np.where(
        df_pool["circularity"] >= threshold, "mitotic", "interphase"
    )

n_mit = (df_pool["cell_type"] == "mitotic").sum()
n_int = (df_pool["cell_type"] == "interphase").sum()
print(f"Classification:  {n_mit} mitotic,  {n_int} interphase")
df_pool.to_csv(OUTPUT_DIR / "pooled_cells.csv", index=False)
print(f"Saved pooled_cells.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
colours = {"mitotic": "#e64b35", "interphase": "#4dbbd5"}

# Experiment colour map for multi-experiment scatter
exp_labels  = sorted(df_pool["experiment"].unique())
exp_colours = {e: plt.cm.Set2(i / max(len(exp_labels) - 1, 1))
               for i, e in enumerate(exp_labels)}


def compare_groups(a, b, label_a="A", label_b="B"):
    a, b = np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)]
    u, p_mw  = stats.mannwhitneyu(a, b, alternative="two-sided")
    ks, p_ks = stats.ks_2samp(a, b)
    return dict(
        label_a=label_a, n_a=len(a),
        median_a=np.median(a), iqr_a=np.percentile(a,75)-np.percentile(a,25),
        label_b=label_b, n_b=len(b),
        median_b=np.median(b), iqr_b=np.percentile(b,75)-np.percentile(b,25),
        U=u, p_mw=p_mw, KS=ks, p_ks=p_ks,
        fold_change=np.median(b)/np.median(a) if np.median(a)!=0 else np.nan,
    )


# ---------------------------------------------------------------------------
# Figure 1: Correlation (pooled)
# ---------------------------------------------------------------------------
print("\nGenerating pooled correlation figure …")

df_valid_pool = df_pool.dropna(subset=["D_median", "circularity"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=130)

# --- D vs circularity ---
ax = axes[0]
for ct in ["mitotic", "interphase"]:
    sub = df_valid_pool[df_valid_pool["cell_type"] == ct]
    for exp, grp in sub.groupby("experiment"):
        ax.scatter(grp["circularity"], grp["D_median"],
                   color=colours[ct], marker="o" if ct == "mitotic" else "s",
                   s=50, alpha=0.8, edgecolors=exp_colours[exp][:3],
                   linewidths=1.2, label=f"{ct} ({exp})", zorder=3)

x_c = df_valid_pool["circularity"].values
y_d = df_valid_pool["D_median"].values
slope, intercept, r, p, _ = stats.linregress(x_c, y_d)
x_line = np.array([x_c.min(), x_c.max()])
ax.plot(x_line, slope*x_line + intercept, "k-", lw=1.5,
        label=f"r = {r:.2f},  p = {p:.3f}", zorder=2)
if MANUAL_CIRCULARITY_THRESHOLD is not None:
    ax.axvline(MANUAL_CIRCULARITY_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.7,
               label=f"threshold = {MANUAL_CIRCULARITY_THRESHOLD:.2f}")

ax.set_xlabel("Circularity", fontsize=10)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Diffusion vs Circularity (pooled)", fontsize=10)
# De-duplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2)

# --- D vs area ---
ax = axes[1]
for ct in ["mitotic", "interphase"]:
    sub = df_valid_pool[df_valid_pool["cell_type"] == ct]
    for exp, grp in sub.groupby("experiment"):
        ax.scatter(grp["area_px"], grp["D_median"],
                   color=colours[ct], marker="o" if ct == "mitotic" else "s",
                   s=50, alpha=0.8, edgecolors=exp_colours[exp][:3],
                   linewidths=1.2, zorder=3)

x_a = df_valid_pool["area_px"].values
slope2, intercept2, r2, p2, _ = stats.linregress(x_a, y_d)
ax.plot(np.array([x_a.min(), x_a.max()]),
        slope2*np.array([x_a.min(), x_a.max()]) + intercept2,
        "k-", lw=1.5, label=f"r = {r2:.2f},  p = {p2:.3f}", zorder=2)
ax.set_xlabel("Cell area  (px²)", fontsize=10)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Diffusion vs Cell Area (pooled)", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle(f"Pooled correlations  [{len(df_pool)} cells, "
             f"{df_pool['experiment'].nunique()} experiment(s)]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pooled_correlation_diffusion.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved pooled_correlation_diffusion.png")

# ---------------------------------------------------------------------------
# Figure 2: Mitotic vs Interphase (pooled)
# ---------------------------------------------------------------------------
print("Generating pooled mitotic vs interphase figure …")

mit_D = df_pool.loc[df_pool["cell_type"] == "mitotic",    "D_median"].dropna().values
int_D = df_pool.loc[df_pool["cell_type"] == "interphase", "D_median"].dropna().values
comp  = compare_groups(mit_D, int_D, "mitotic", "interphase")

p_val = comp["p_mw"]
sig   = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
         ("*" if p_val < 0.05 else "ns"))

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), dpi=130)

# --- violin ---
ax = axes2[0]
parts = ax.violinplot([mit_D, int_D], positions=[1, 2],
                      showmedians=True, showextrema=False, widths=0.65)
for pc, col in zip(parts["bodies"], [colours["mitotic"], colours["interphase"]]):
    pc.set_facecolor(col); pc.set_alpha(0.75)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

rng = np.random.default_rng(7)
for pos, vals, col in [(1, mit_D, colours["mitotic"]),
                        (2, int_D, colours["interphase"])]:
    jit = rng.uniform(-0.09, 0.09, len(vals))
    ax.scatter(pos + jit, vals, s=35, color=col, alpha=0.9,
               edgecolors="white", linewidths=0.4, zorder=3)

y_max = max(np.nanmax(mit_D), np.nanmax(int_D)) * 1.06
ax.plot([1, 2], [y_max, y_max], "k-", lw=1)
ax.text(1.5, y_max * 1.02, sig, ha="center", va="bottom", fontsize=13)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Mitotic\n(n={len(mit_D)})", f"Interphase\n(n={len(int_D)})"])
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title(f"Mitotic vs Interphase\nMW p = {p_val:.4f}", fontsize=10)

# --- per-experiment breakdown ---
ax = axes2[1]
x_pos = 0
xtick_pos, xtick_lbl = [], []
for exp in exp_labels:
    for i, ct in enumerate(["mitotic", "interphase"]):
        sub_vals = df_pool.loc[(df_pool["experiment"]==exp) &
                                (df_pool["cell_type"]==ct), "D_median"].dropna().values
        if len(sub_vals) == 0:
            continue
        x_pos += 1
        jit = rng.uniform(-0.1, 0.1, len(sub_vals))
        ax.scatter(x_pos + jit, sub_vals, s=25, color=colours[ct],
                   alpha=0.8, edgecolors="white", linewidths=0.3)
        ax.plot([x_pos - 0.25, x_pos + 0.25],
                [np.median(sub_vals)]*2, color="black", lw=2)
        xtick_pos.append(x_pos)
        xtick_lbl.append(f"{exp[:8]}\n{ct[:3]}")
    x_pos += 0.5  # gap between experiments

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=7)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Per-experiment breakdown", fontsize=10)

# --- CDF ---
ax = axes2[2]
for vals, col, lbl in [(mit_D, colours["mitotic"],    f"Mitotic (n={len(mit_D)})"),
                        (int_D, colours["interphase"], f"Interphase (n={len(int_D)})")]:
    sv = np.sort(vals)
    ax.plot(sv, np.arange(1, len(sv)+1)/len(sv), color=col, lw=2, label=lbl)
ax.set_xlabel("Median D  (µm²/s)", fontsize=10)
ax.set_ylabel("Cumulative fraction", fontsize=10)
ax.set_title("CDF — per-cell median D", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle(f"Pooled mitotic vs interphase GEM diffusion  [{df_pool['experiment'].nunique()} experiments]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "pooled_mitotic_vs_interphase.png", dpi=130, bbox_inches="tight")
plt.close(fig2)
print("  Saved pooled_mitotic_vs_interphase.png")

# ---------------------------------------------------------------------------
# Figure 3: Between-mitotic-cell comparison (pooled)
# ---------------------------------------------------------------------------
print("Generating pooled mitotic cell comparison figure …")

df_mit = df_pool[(df_pool["cell_type"] == "mitotic") &
                 (df_pool["n_valid_traj"] >= MIN_TRAJ_MITOTIC)].copy()
df_mit = df_mit.sort_values("D_median").reset_index(drop=True)
n_mit_cells = len(df_mit)
print(f"  {n_mit_cells} mitotic cells with ≥{MIN_TRAJ_MITOTIC} trajectories")

if n_mit_cells >= 2:
    fig3, ax3 = plt.subplots(figsize=(max(10, n_mit_cells * 0.55), 5), dpi=130)

    positions = np.arange(1, n_mit_cells + 1)
    cmap_vals = plt.cm.Reds(np.linspace(0.35, 0.9, n_mit_cells))

    # Box plot (no individual D values here since we only have cell-level median)
    ax3.bar(positions, df_mit["D_median"], color=cmap_vals, alpha=0.8,
            edgecolor="white", width=0.65, zorder=2)
    ax3.errorbar(positions, df_mit["D_median"],
                 yerr=[df_mit["D_median"] - df_mit["D_p25"],
                       df_mit["D_p75"] - df_mit["D_median"]],
                 fmt="none", color="black", capsize=3, linewidth=1, zorder=3)

    # Colour border by experiment
    for pos, (_, row) in zip(positions, df_mit.iterrows()):
        exp_col = exp_colours[row["experiment"]]
        ax3.bar(pos, row["D_median"], color="none",
                edgecolor=exp_col, linewidth=2.5, width=0.65, zorder=4)

    ax3.set_xticks(positions)
    xticklabels = [
        f"C{int(row['cell_label'])}\n{row['experiment'][:6]}\ncirc={row['circularity']:.2f}"
        for _, row in df_mit.iterrows()
    ]
    ax3.set_xticklabels(xticklabels, fontsize=6, rotation=45, ha="right")
    ax3.set_ylabel("Median D  (µm²/s)", fontsize=10)
    ax3.set_title(
        f"Individual mitotic cell diffusion  [{n_mit_cells} cells, "
        f"{df_mit['experiment'].nunique()} experiment(s)]\n"
        f"Bar = median, error bars = IQR, border colour = experiment",
        fontsize=10,
    )

    # Kruskal-Wallis if ≥3 mitotic cells
    if n_mit_cells >= 3:
        # Use D_median as single value per cell (since per-traj not pooled here)
        kw_stat, kw_p = stats.kruskal(*[
            [row["D_median"]] for _, row in df_mit.iterrows()
        ])
        ax3.text(0.98, 0.97,
                 f"Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f}",
                 transform=ax3.transAxes, fontsize=8,
                 ha="right", va="top", color="darkred",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Legend for experiments
    legend_handles = [plt.Rectangle((0,0),1,1, color=exp_colours[e], label=e)
                      for e in exp_labels]
    ax3.legend(handles=legend_handles, fontsize=8, title="Experiment",
               loc="upper left", framealpha=0.8)

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "pooled_mitotic_cell_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved pooled_mitotic_cell_comparison.png")

# ---------------------------------------------------------------------------
# Stats text report
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "pooled_stats.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("Step 4 — Pooled Experiment Analysis Report\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Experiments included ({len(experiments)}):\n")
    for lbl, p_csv in experiments:
        f.write(f"  {lbl}:  {p_csv}\n")
    f.write(f"\nTotal cells after filtering: {len(df_pool)}\n")
    f.write(f"  Mitotic ({MANUAL_CIRCULARITY_THRESHOLD or 'GMM'}): {n_mit} cells\n")
    f.write(f"  Interphase: {n_int} cells\n\n")
    f.write(f"Circularity threshold: {MANUAL_CIRCULARITY_THRESHOLD or 'GMM'}\n\n")

    f.write("Correlation — D_median vs circularity:\n")
    f.write(f"  Pearson r = {r:.4f},  p = {p:.4e}\n\n")
    f.write("Correlation — D_median vs area:\n")
    f.write(f"  Pearson r = {r2:.4f},  p = {p2:.4e}\n\n")

    f.write("Mitotic vs Interphase:\n")
    f.write(f"  Mitotic    median D = {comp['median_a']:.5f} µm²/s  (IQR {comp['iqr_a']:.5f})\n")
    f.write(f"  Interphase median D = {comp['median_b']:.5f} µm²/s  (IQR {comp['iqr_b']:.5f})\n")
    f.write(f"  Fold change (int/mit) = {comp['fold_change']:.3f}×\n")
    f.write(f"  Mann-Whitney U={comp['U']:.1f},  p={comp['p_mw']:.4e}  [{sig}]\n")
    f.write(f"  KS stat={comp['KS']:.4f},  p={comp['p_ks']:.4e}\n\n")

    f.write("Per-experiment breakdown:\n")
    for exp in exp_labels:
        m = df_pool.loc[(df_pool["experiment"]==exp) & (df_pool["cell_type"]=="mitotic"), "D_median"]
        i = df_pool.loc[(df_pool["experiment"]==exp) & (df_pool["cell_type"]=="interphase"), "D_median"]
        f.write(f"  {exp}: mitotic n={len(m)}, median={np.median(m) if len(m) else np.nan:.5f} | "
                f"interphase n={len(i)}, median={np.median(i) if len(i) else np.nan:.5f}\n")

print(f"Saved pooled_stats.txt")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 4 POOLED RESULTS SUMMARY")
print("=" * 55)
print(f"Experiments: {len(experiments)},  Total cells: {len(df_pool)}")
print(f"Mitotic: {n_mit},  Interphase: {n_int}")
print()
print(f"D vs circularity:  r = {r:.3f},  p = {p:.4f}")
print(f"D vs area:         r = {r2:.3f},  p = {p2:.4f}")
print()
print(f"Mitotic    median D = {comp['median_a']:.5f} µm²/s")
print(f"Interphase median D = {comp['median_b']:.5f} µm²/s")
print(f"Fold change (int/mit) = {comp['fold_change']:.2f}×")
print(f"Mann-Whitney p = {comp['p_mw']:.4f}  [{sig}]")
print()
print(f"Results saved to: {OUTPUT_DIR}/")
print("\nStep 4 complete.")
print("\nTo add more experiments, either:")
print("  1. Add sub-folders with diffusion_per_cell.csv to EXPERIMENTS_ROOT, or")
print("  2. Add entries to EXPERIMENT_CSVS in this script.")

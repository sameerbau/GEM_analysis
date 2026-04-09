#!/usr/bin/env python3
"""
3_circularity_diffusion_correlation.py

Step 3 of the mitosis GEM analysis pipeline.

Classifies cells as mitotic vs interphase based on circularity, then:
  1. Correlates diffusion coefficient with circularity and cell area.
  2. Compares D between mitotic and interphase cells.
  3. Compares D across individual mitotic cells (cell-to-cell viscosity variation).

Classification strategy
-----------------------
A 2-component Gaussian Mixture Model is fitted to the circularity distribution.
The component with higher mean circularity defines the "mitotic" population.
The decision boundary (crossover point) is used as the threshold.
A MANUAL_CIRCULARITY_THRESHOLD override is also available (set to None to use GMM).

Outputs
-------
  cell_classification.csv         – per-cell label with mitotic/interphase assignment
  correlation_diffusion.png       – scatter plots: D vs circularity & D vs area
  mitotic_vs_interphase.png       – D comparison: mitotic vs interphase
  mitotic_cell_comparison.png     – D per individual mitotic cell (violin + strip)
  step3_stats.txt                 – statistical test results

Requires: diffusion_per_cell.pkl  (output of 2_diffusion_per_cell.py)

Usage:
  python 3_circularity_diffusion_correlation.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.mixture import GaussianMixture
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

def _ask_folder(prompt, default=None):
    """Prompt for a folder. Accepts sys.argv[1] if provided."""
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        p = Path(sys.argv[1]).expanduser().resolve()
        print(f"  Using folder from command line: {p}")
        return p
    while True:
        hint = f"  [Enter = {default}]" if default else ""
        raw = input(f"\n  {prompt}{hint}: ").strip().strip("'\"")
        if not raw and default:
            return Path(default).expanduser().resolve()
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            return p
        print(f"  Not found: '{raw}'  —  please try again.")

import subprocess

print("=" * 60)
print("Step 3 — Circularity / diffusion correlation")
print("=" * 60)
WORK_DIR = _ask_folder(
    "Folder containing diffusion_per_cell.pkl  (or megafolder with embryo subfolders)",
    default=str(SCRIPT_DIR),
)
print(f"  Working folder: {WORK_DIR}\n")

# ---------------------------------------------------------------------------
# Megafolder mode: if diffusion_per_cell.pkl is not directly here, look for
# subfolders that each contain it and process them all.
# ---------------------------------------------------------------------------
if not (WORK_DIR / "diffusion_per_cell.pkl").exists():
    _subdirs = sorted(d for d in WORK_DIR.iterdir()
                      if d.is_dir() and (d / "diffusion_per_cell.pkl").exists())
    if _subdirs:
        print(f"  Megafolder mode — found {len(_subdirs)} embryo subfolder(s):")
        for _d in _subdirs:
            print(f"    {_d.name}")
        _resp = input("\n  Process all? [Y/n]: ").strip().lower()
        if _resp not in ("", "y", "yes"):
            print("  Aborted.")
            sys.exit(0)
        for _d in _subdirs:
            print(f"\n{'='*60}\n  Processing: {_d.name}\n{'='*60}")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), str(_d)],
                           check=False)
        print("\nStep 3 — Megafolder batch complete.")
        sys.exit(0)
    print(f"  ERROR: diffusion_per_cell.pkl not found in {WORK_DIR}")
    print("  Run Step 2 (2_diffusion_per_cell.py) first.")
    sys.exit(1)

IN_PKL     = WORK_DIR / "diffusion_per_cell.pkl"

# Set a fixed circularity threshold to override GMM (e.g. 0.72), or None to use GMM.
MANUAL_CIRCULARITY_THRESHOLD = 0.70

# Minimum number of valid trajectories a cell needs to enter the
# mitotic-vs-interphase comparison (cells with very few D values are noisy)
MIN_TRAJ_FOR_COMPARISON = 5

# For between-mitotic-cell comparison: minimum trajectories per mitotic cell
MIN_TRAJ_MITOTIC = 5

# Output files
OUT_CSV       = WORK_DIR / "cell_classification.csv"
OUT_FIG_CORR  = WORK_DIR / "correlation_diffusion.png"
OUT_FIG_COMP  = WORK_DIR / "mitotic_vs_interphase.png"
OUT_FIG_MIT   = WORK_DIR / "mitotic_cell_comparison.png"
OUT_STATS_TXT = WORK_DIR / "step3_stats.txt"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading per-cell diffusion data …")
with open(IN_PKL, "rb") as f:
    data = pickle.load(f)

df        = data["cell_stats"].copy()
df_valid  = data["valid_traj"].copy()   # per-trajectory D values

print(f"  {len(df)} cells loaded")
df = df.dropna(subset=["D_median", "circularity"])
print(f"  {len(df)} cells with valid D and circularity")

# ---------------------------------------------------------------------------
# GMM-based circularity threshold
# ---------------------------------------------------------------------------
circ_vals = df["circularity"].values.reshape(-1, 1)

gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
gmm.fit(circ_vals)
means = gmm.means_.flatten()
# Label component with higher mean as "mitotic"
mitotic_component = int(np.argmax(means))
interphase_component = 1 - mitotic_component

print(f"\nGMM circularity fit:")
print(f"  Component 0: mean = {means[0]:.3f},  weight = {gmm.weights_[0]:.2f}")
print(f"  Component 1: mean = {means[1]:.3f},  weight = {gmm.weights_[1]:.2f}")
print(f"  → Mitotic component: {mitotic_component} (higher mean circularity)")

# Find the decision boundary (crossover of the two Gaussian PDFs)
# between the two component means
mean_low  = means[1 - mitotic_component]
mean_high = means[mitotic_component]
std_low   = np.sqrt(gmm.covariances_.flatten()[1 - mitotic_component])
std_high  = np.sqrt(gmm.covariances_.flatten()[mitotic_component])
w_low     = gmm.weights_[1 - mitotic_component]
w_high    = gmm.weights_[mitotic_component]

# Scan for crossover between the two component means
x_scan = np.linspace(mean_low, mean_high, 10000)
pdf_low  = w_low  * stats.norm.pdf(x_scan, mean_low,  std_low)
pdf_high = w_high * stats.norm.pdf(x_scan, mean_high, std_high)
crossover_idx = np.argmin(np.abs(pdf_low - pdf_high))
gmm_threshold = x_scan[crossover_idx]

if MANUAL_CIRCULARITY_THRESHOLD is not None:
    threshold = MANUAL_CIRCULARITY_THRESHOLD
    print(f"  Using MANUAL threshold: {threshold:.3f}  (GMM would give {gmm_threshold:.3f})")
else:
    threshold = gmm_threshold
    print(f"  GMM threshold (crossover): {threshold:.3f}")

# ---------------------------------------------------------------------------
# Classify cells
# ---------------------------------------------------------------------------
df["cell_type"] = np.where(df["circularity"] >= threshold, "mitotic", "interphase")
n_mit  = (df["cell_type"] == "mitotic").sum()
n_int  = (df["cell_type"] == "interphase").sum()
print(f"\nClassification at threshold = {threshold:.3f}:")
print(f"  Mitotic:    {n_mit} cells")
print(f"  Interphase: {n_int} cells")

df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV.name}")

# Attach cell_type to per-trajectory D values
df_valid2 = df_valid.merge(df[["cell_label", "cell_type"]], on="cell_label", how="left")

# ---------------------------------------------------------------------------
# Helper: statistical comparison between two groups
# ---------------------------------------------------------------------------
def compare_groups(a, b, label_a, label_b):
    """Mann-Whitney U + Kolmogorov-Smirnov + summary stats. Returns dict."""
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]

    u_stat, p_mw  = stats.mannwhitneyu(a, b, alternative="two-sided")
    ks_stat, p_ks = stats.ks_2samp(a, b)

    result = {
        "label_a": label_a, "n_a": len(a),
        "median_a": np.median(a), "iqr_a": np.percentile(a,75) - np.percentile(a,25),
        "label_b": label_b, "n_b": len(b),
        "median_b": np.median(b), "iqr_b": np.percentile(b,75) - np.percentile(b,25),
        "U_stat": u_stat, "p_mannwhitney": p_mw,
        "KS_stat": ks_stat, "p_ks": p_ks,
        "fold_change": np.median(b) / np.median(a) if np.median(a) != 0 else np.nan,
    }
    return result


# ---------------------------------------------------------------------------
# Figure 1: Correlation – D vs circularity, D vs area
# ---------------------------------------------------------------------------
print("\nGenerating correlation figure …")

colours = {"mitotic": "#e64b35", "interphase": "#4dbbd5"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=130)

# --- (a) GMM fit on circularity distribution ---
ax = axes[0]
x_plot = np.linspace(df["circularity"].min() - 0.05,
                     df["circularity"].max() + 0.05, 500)
ax.hist(df["circularity"], bins=20, density=True,
        color="lightgray", edgecolor="white", label="data")

for comp_idx, (col, lbl) in enumerate(
        zip(["#4dbbd5","#e64b35"],
            ["interphase component","mitotic component"])):
    m_c = gmm.means_[comp_idx, 0]
    s_c = np.sqrt(gmm.covariances_[comp_idx, 0])
    w_c = gmm.weights_[comp_idx]
    ax.plot(x_plot, w_c * stats.norm.pdf(x_plot, m_c, s_c),
            color=col, lw=2, label=f"{lbl}\nμ={m_c:.2f}")

ax.axvline(threshold, color="black", ls="--", lw=1.5,
           label=f"threshold = {threshold:.3f}")
ax.set_xlabel("Circularity", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("GMM circularity classification", fontsize=10)
ax.legend(fontsize=7)

# --- (b) D vs circularity ---
ax = axes[1]
for ct, grp in df.groupby("cell_type"):
    ax.scatter(grp["circularity"], grp["D_median"],
               color=colours[ct], s=50, alpha=0.85,
               edgecolors="white", linewidths=0.5, label=ct, zorder=3)

# Regression line (all cells)
x_c = df["circularity"].values
y_d = df["D_median"].values
slope, intercept, r, p, se = stats.linregress(x_c, y_d)
x_line = np.array([x_c.min(), x_c.max()])
ax.plot(x_line, slope * x_line + intercept, color="black",
        lw=1.5, ls="-", zorder=2, label=f"r = {r:.2f},  p = {p:.3f}")
ax.axvline(threshold, color="gray", ls="--", lw=1, alpha=0.6)

ax.set_xlabel("Circularity", fontsize=10)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Diffusion vs Circularity", fontsize=10)
ax.legend(fontsize=8)

# --- (c) D vs area ---
ax = axes[2]
for ct, grp in df.groupby("cell_type"):
    ax.scatter(grp["area_px"], grp["D_median"],
               color=colours[ct], s=50, alpha=0.85,
               edgecolors="white", linewidths=0.5, label=ct, zorder=3)

x_a = df["area_px"].values
slope2, intercept2, r2, p2, _ = stats.linregress(x_a, y_d)
x_line2 = np.array([x_a.min(), x_a.max()])
ax.plot(x_line2, slope2 * x_line2 + intercept2, color="black",
        lw=1.5, ls="-", zorder=2, label=f"r = {r2:.2f},  p = {p2:.3f}")

ax.set_xlabel("Cell area  (px²)", fontsize=10)
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title("Diffusion vs Cell Area", fontsize=10)
ax.legend(fontsize=8)

plt.suptitle("GEM diffusion correlations with cell morphology",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_FIG_CORR, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_FIG_CORR.name}")

# ---------------------------------------------------------------------------
# Figure 2: Mitotic vs Interphase D comparison
# ---------------------------------------------------------------------------
print("Generating mitotic vs interphase comparison figure …")

df_comp = df[df["n_valid_traj"] >= MIN_TRAJ_FOR_COMPARISON].copy()
mit_D   = df_comp.loc[df_comp["cell_type"] == "mitotic",   "D_median"].dropna().values
int_D   = df_comp.loc[df_comp["cell_type"] == "interphase","D_median"].dropna().values

# Per-trajectory D for CDF
mit_traj_D = df_valid2.loc[
    (df_valid2["cell_type"] == "mitotic")  & df_valid2["D"].notna(), "D"].values
int_traj_D = df_valid2.loc[
    (df_valid2["cell_type"] == "interphase") & df_valid2["D"].notna(), "D"].values

comp = compare_groups(mit_D, int_D, "mitotic", "interphase")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), dpi=130)

# --- (a) Violin ---
ax = axes2[0]
parts = ax.violinplot([mit_D, int_D], positions=[1, 2],
                      showmedians=True, showextrema=False, widths=0.6)
for pc, col in zip(parts["bodies"], [colours["mitotic"], colours["interphase"]]):
    pc.set_facecolor(col); pc.set_alpha(0.75)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

# Overlay individual points with jitter
rng = np.random.default_rng(7)
for pos, vals, col in [(1, mit_D, colours["mitotic"]),
                        (2, int_D, colours["interphase"])]:
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(pos + jitter, vals, s=30, color=col, alpha=0.9,
               edgecolors="white", linewidths=0.4, zorder=3)

p_val = comp["p_mannwhitney"]
sig   = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
         ("*" if p_val < 0.05 else "ns"))
y_max = max(np.max(mit_D), np.max(int_D)) * 1.05
ax.plot([1, 2], [y_max, y_max], color="black", lw=1)
ax.text(1.5, y_max * 1.02, sig, ha="center", va="bottom", fontsize=12)

ax.set_xticks([1, 2])
ax.set_xticklabels([f"Mitotic\n(n={len(mit_D)})", f"Interphase\n(n={len(int_D)})"])
ax.set_ylabel("Median D  (µm²/s)", fontsize=10)
ax.set_title(f"Mitotic vs Interphase\nMW p = {p_val:.4f}", fontsize=10)

# --- (b) CDF ---
ax = axes2[1]
for vals, col, lbl in [(mit_traj_D,  colours["mitotic"],    "Mitotic"),
                        (int_traj_D, colours["interphase"], "Interphase")]:
    sorted_v = np.sort(vals)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax.plot(sorted_v, cdf, color=col, lw=2, label=f"{lbl} (n={len(vals)})")
ax.set_xlabel("D  (µm²/s)", fontsize=10)
ax.set_ylabel("Cumulative fraction", fontsize=10)
ax.set_title("CDF — per-trajectory D", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(left=0)

# --- (c) Summary table as text ---
ax = axes2[2]
ax.axis("off")
table_data = [
    ["", "Mitotic", "Interphase"],
    ["n (cells)", str(len(mit_D)), str(len(int_D))],
    ["Median D", f"{comp['median_a']:.4f}", f"{comp['median_b']:.4f}"],
    ["IQR D", f"{comp['iqr_a']:.4f}", f"{comp['iqr_b']:.4f}"],
    ["Fold change", f"{comp['fold_change']:.2f}×", "—"],
    ["Mann-Whitney p", f"{comp['p_mannwhitney']:.4f}", ""],
    ["KS p", f"{comp['p_ks']:.4f}", ""],
    ["Significance", sig, ""],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.3, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2b2b2b"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#ffe0dc")
    elif c == 2:
        cell.set_facecolor("#d6f0f5")
ax.set_title("Statistical summary", fontsize=10, pad=10)

plt.suptitle("Mitotic vs Interphase GEM diffusion",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig2.savefig(OUT_FIG_COMP, dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved {OUT_FIG_COMP.name}")

# ---------------------------------------------------------------------------
# Figure 3: Between-mitotic-cell comparison
# ---------------------------------------------------------------------------
print("Generating between-mitotic-cell comparison figure …")

df_mit = df_comp[df_comp["cell_type"] == "mitotic"].copy()
df_mit = df_mit[df_mit["n_valid_traj"] >= MIN_TRAJ_MITOTIC].sort_values("D_median")
mitotic_cells = df_mit["cell_label"].tolist()
n_mit_cells   = len(mitotic_cells)

print(f"  {n_mit_cells} mitotic cells with ≥{MIN_TRAJ_MITOTIC} trajectories")

if n_mit_cells >= 2:
    # Collect per-trajectory D for each mitotic cell
    per_cell_D = {}
    for lbl in mitotic_cells:
        vals = df_valid2.loc[(df_valid2["cell_label"] == lbl) &
                              df_valid2["D"].notna(), "D"].values
        per_cell_D[lbl] = vals

    fig3, axes3 = plt.subplots(1, 2, figsize=(max(10, n_mit_cells * 1.2), 5),
                                dpi=130)

    # --- (a) Violin per mitotic cell ---
    ax = axes3[0]
    positions = np.arange(1, n_mit_cells + 1)
    cmap_mit = plt.cm.Reds(np.linspace(0.4, 0.9, n_mit_cells))

    parts = ax.violinplot(
        [per_cell_D[lbl] for lbl in mitotic_cells],
        positions=positions, showmedians=True, showextrema=False, widths=0.65,
    )
    for pc, col in zip(parts["bodies"], cmap_mit):
        pc.set_facecolor(col); pc.set_alpha(0.8)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)

    # Strip plot overlay
    rng2 = np.random.default_rng(99)
    for pos, lbl, col in zip(positions, mitotic_cells, cmap_mit):
        vals = per_cell_D[lbl]
        jit  = rng2.uniform(-0.1, 0.1, len(vals))
        ax.scatter(pos + jit, vals, s=20, color=col, alpha=0.9,
                   edgecolors="white", linewidths=0.3, zorder=3)

    # Annotate with n and circularity
    for pos, lbl in zip(positions, mitotic_cells):
        row   = df_mit[df_mit["cell_label"] == lbl].iloc[0]
        label = f"Cell {lbl}\n(n={int(row['n_valid_traj'])}, circ={row['circularity']:.2f})"
        ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                label, ha="center", va="top", fontsize=6, rotation=45)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{lbl}" for lbl in mitotic_cells], fontsize=8)
    ax.set_ylabel("D  (µm²/s)", fontsize=10)
    ax.set_title(f"D per mitotic cell  (sorted by median D)", fontsize=10)

    # --- (b) Kruskal-Wallis + pairwise summary ---
    ax2 = axes3[1]
    ax2.axis("off")

    all_groups = [per_cell_D[lbl] for lbl in mitotic_cells]
    if n_mit_cells >= 3:
        kw_stat, kw_p = stats.kruskal(*all_groups)
    else:
        kw_stat, kw_p = np.nan, np.nan

    # Cell-level D stats
    rows_table = [["Cell", "n traj", "Median D", "IQR D", "Circularity"]]
    for lbl in mitotic_cells:
        row_d = df_mit[df_mit["cell_label"] == lbl].iloc[0]
        vals  = per_cell_D[lbl]
        iqr_v = np.percentile(vals, 75) - np.percentile(vals, 25)
        rows_table.append([
            str(lbl),
            str(int(row_d["n_valid_traj"])),
            f"{np.median(vals):.4f}",
            f"{iqr_v:.4f}",
            f"{row_d['circularity']:.3f}",
        ])

    if n_mit_cells >= 3:
        rows_table.append(["", "", "", "", ""])
        rows_table.append([f"Kruskal-Wallis", f"H={kw_stat:.2f}", f"p={kw_p:.4f}", "", ""])

    tbl2 = ax2.table(cellText=rows_table[1:], colLabels=rows_table[0],
                     loc="center", cellLoc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8)
    tbl2.scale(1.2, 1.5)
    for (r, c), cell in tbl2.get_celld().items():
        if r == 0:
            cell.set_facecolor("#8b0000"); cell.set_text_props(color="white", fontweight="bold")
    ax2.set_title("Mitotic cell statistics", fontsize=10, pad=10)

    plt.suptitle("Between-mitotic-cell diffusion comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(OUT_FIG_MIT, dpi=130, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {OUT_FIG_MIT.name}")
else:
    print(f"  Only {n_mit_cells} mitotic cell(s) — skipping between-cell comparison.")
    kw_stat, kw_p = np.nan, np.nan

# ---------------------------------------------------------------------------
# Save statistics text report
# ---------------------------------------------------------------------------
with open(OUT_STATS_TXT, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Step 3 — Circularity / Diffusion Analysis Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Circularity threshold (GMM): {gmm_threshold:.4f}\n")
    f.write(f"Threshold used:              {threshold:.4f}\n")
    f.write(f"  (GMM component means: {means[0]:.3f}, {means[1]:.3f})\n\n")

    f.write(f"Cell classification:\n")
    f.write(f"  Mitotic:    {n_mit} cells\n")
    f.write(f"  Interphase: {n_int} cells\n\n")

    f.write("Correlation — D_median vs circularity:\n")
    f.write(f"  Pearson r = {r:.4f},  p = {p:.4e}\n\n")
    f.write("Correlation — D_median vs area:\n")
    f.write(f"  Pearson r = {r2:.4f},  p = {p2:.4e}\n\n")

    f.write("Mitotic vs Interphase (cell-level median D):\n")
    f.write(f"  Mitotic    median D = {comp['median_a']:.5f} µm²/s  (IQR {comp['iqr_a']:.5f})\n")
    f.write(f"  Interphase median D = {comp['median_b']:.5f} µm²/s  (IQR {comp['iqr_b']:.5f})\n")
    f.write(f"  Fold change (interphase/mitotic) = {comp['fold_change']:.3f}×\n")
    f.write(f"  Mann-Whitney U = {comp['U_stat']:.1f},  p = {comp['p_mannwhitney']:.4e}\n")
    f.write(f"  Kolmogorov-Smirnov stat = {comp['KS_stat']:.4f},  p = {comp['p_ks']:.4e}\n\n")

    if n_mit_cells >= 3:
        f.write("Between-mitotic-cell (Kruskal-Wallis):\n")
        f.write(f"  H = {kw_stat:.3f},  p = {kw_p:.4e}\n\n")

    f.write("Per-cell D values (mitotic cells, sorted by D_median):\n")
    for lbl in mitotic_cells:
        row_d = df_mit[df_mit["cell_label"] == lbl].iloc[0]
        f.write(f"  Cell {lbl:3d}  circ={row_d['circularity']:.3f}  "
                f"n={int(row_d['n_valid_traj'])}  "
                f"D_median={row_d['D_median']:.5f} µm²/s\n")

print(f"\nSaved {OUT_STATS_TXT.name}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 3 RESULTS SUMMARY")
print("=" * 55)
print(f"Circularity threshold: {threshold:.3f}  (GMM: {gmm_threshold:.3f})")
print(f"Mitotic cells: {n_mit}   Interphase cells: {n_int}")
print()
print(f"D vs circularity:  r = {r:.3f},  p = {p:.4f}")
print(f"D vs area:         r = {r2:.3f},  p = {p2:.4f}")
print()
print(f"Mitotic    median D = {comp['median_a']:.5f} µm²/s")
print(f"Interphase median D = {comp['median_b']:.5f} µm²/s")
print(f"Fold change (int/mit) = {comp['fold_change']:.2f}×")
print(f"Mann-Whitney p = {comp['p_mannwhitney']:.4f}  [{sig}]")
print()
print("Step 3 complete.")

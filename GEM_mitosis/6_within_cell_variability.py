#!/usr/bin/env python3
"""
6_within_cell_variability.py

Analyses the *within-cell* variability of GEM diffusion coefficients and
correlates it with cell morphology (circularity, area) for mitotic and
interphase cells separately.

Within-cell variability metrics (all computed per cell from its per-trajectory D values):
  D_iqr   – interquartile range  (robust spread)
  D_std   – standard deviation
  D_cv    – coefficient of variation  (D_std / D_mean, dimensionless)

Outputs (written to OUTPUT_DIR):
  variability_vs_shape.png          – scatter: variability metrics vs circularity & area
  variability_mitotic_vs_interphase.png  – violin/strip: are mitotic cells more variable?
  variability_mitotic_cells.png     – per-mitotic-cell bar: variability ordered by circularity/area
  variability_stats.txt             – full correlation report

Input:
  pooled_results_v2/pooled_cells.csv  (output of 5_batch_pipeline.py)

Usage:
  python 6_within_cell_variability.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV  = SCRIPT_DIR / "pooled_results_v2" / "pooled_cells.csv"
OUTPUT_DIR = SCRIPT_DIR / "pooled_results_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colours
COLOURS = {"mitotic": "#e64b35", "interphase": "#4dbbd5"}

# ---------------------------------------------------------------------------
# Load data and add D_cv
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df["D_cv"] = df["D_std"] / df["D_mean"]   # coefficient of variation

n_total = len(df)
n_mit   = (df["cell_type"] == "mitotic").sum()
n_int   = (df["cell_type"] == "interphase").sum()
print(f"Loaded {n_total} cells — {n_mit} mitotic, {n_int} interphase")
print(f"Experiments: {sorted(df['experiment'].unique())}\n")

df_mit = df[df["cell_type"] == "mitotic"].copy()
df_int = df[df["cell_type"] == "interphase"].copy()

# ---------------------------------------------------------------------------
# Helper: correlation stats (Pearson + Spearman)
# ---------------------------------------------------------------------------
def corr_stats(x, y, label_x, label_y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return dict(x=label_x, y=label_y, n=n,
                pearson_r=r_p, pearson_p=p_p,
                spearman_r=r_s, spearman_p=p_s)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def add_regression(ax, x, y, color="black", lw=1.5):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xl = np.array([x[mask].min(), x[mask].max()])
    ax.plot(xl, slope * xl + intercept, color=color, lw=lw, zorder=5)
    return r, p

# ---------------------------------------------------------------------------
# Figure 1: Variability vs circularity and area (mitotic / interphase side-by-side)
# ---------------------------------------------------------------------------
print("Generating variability vs shape figure …")

variability_metrics = [
    ("D_iqr", "D IQR  (µm²/s)"),
    ("D_std", "D std  (µm²/s)"),
    ("D_cv",  "D CV  (std/mean)"),
]
shape_metrics = [
    ("circularity", "Circularity"),
    ("area_px",     "Cell area  (px²)"),
]

n_rows = len(variability_metrics)
n_cols = len(shape_metrics) * 2   # mitotic + interphase for each shape metric

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5 * n_cols, 4 * n_rows), dpi=120)

results_corr = []

for row_i, (var_col, var_label) in enumerate(variability_metrics):
    for col_i, (shp_col, shp_label) in enumerate(shape_metrics):
        for k, (ct, sub, lbl_ct) in enumerate([
            ("mitotic",    df_mit, f"Mitotic (n={len(df_mit)})"),
            ("interphase", df_int, f"Interphase (n={len(df_int)})"),
        ]):
            ax = axes[row_i, col_i * 2 + k]

            x = sub[shp_col].values.astype(float)
            y = sub[var_col].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)

            # scatter coloured by experiment
            exp_list = sorted(sub["experiment"].unique())
            exp_cmap = {e: plt.cm.Set2(i / max(len(exp_list) - 1, 1))
                        for i, e in enumerate(exp_list)}
            for exp, grp in sub.groupby("experiment"):
                xe = grp[shp_col].values.astype(float)
                ye = grp[var_col].values.astype(float)
                ax.scatter(xe, ye, s=45, color=COLOURS[ct],
                           edgecolors=exp_cmap[exp][:3], linewidths=1.2,
                           alpha=0.85, label=exp, zorder=3)

            if mask.sum() >= 4:
                res = add_regression(ax, x, y, color="black")
                if res:
                    r, p = res
                    ax.text(0.05, 0.95,
                            f"r={r:.2f} {sig_stars(p)}\np={p:.3f}",
                            transform=ax.transAxes, fontsize=8,
                            va="top", color="black",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
                c = corr_stats(x, y, shp_col, var_col)
                c["cell_type"] = ct
                results_corr.append(c)

            ax.set_xlabel(shp_label, fontsize=9)
            ax.set_ylabel(var_label if (col_i * 2 + k) == 0 else "", fontsize=9)
            ax.set_title(f"{var_label}\nvs {shp_label} — {lbl_ct}", fontsize=8)

plt.suptitle(
    "Within-cell GEM diffusion variability vs cell morphology\n"
    f"(pooled {df['experiment'].nunique()} experiments, {n_total} cells)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "variability_vs_shape.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved variability_vs_shape.png")

# ---------------------------------------------------------------------------
# Figure 2: Mitotic vs Interphase — is one more variable?
# ---------------------------------------------------------------------------
print("Generating mitotic vs interphase variability figure …")

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5), dpi=120)

for ax, (var_col, var_label) in zip(axes2, variability_metrics):
    mit_vals = df_mit[var_col].dropna().values
    int_vals = df_int[var_col].dropna().values

    u, p_mw = stats.mannwhitneyu(mit_vals, int_vals, alternative="two-sided")
    sig = sig_stars(p_mw)

    # violin
    parts = ax.violinplot([mit_vals, int_vals], positions=[1, 2],
                          showmedians=True, showextrema=False, widths=0.6)
    for pc, col in zip(parts["bodies"], [COLOURS["mitotic"], COLOURS["interphase"]]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(42)
    for pos, vals, col in [(1, mit_vals, COLOURS["mitotic"]),
                           (2, int_vals, COLOURS["interphase"])]:
        jit = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(pos + jit, vals, s=30, color=col, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)

    y_max = max(np.nanmax(mit_vals), np.nanmax(int_vals)) * 1.05
    ax.plot([1, 2], [y_max, y_max], "k-", lw=1)
    ax.text(1.5, y_max * 1.02, sig, ha="center", va="bottom", fontsize=13)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Mitotic\n(n={len(mit_vals)})",
                        f"Interphase\n(n={len(int_vals)})"])
    ax.set_ylabel(var_label, fontsize=10)
    ax.set_title(f"{var_label}\nMW p={p_mw:.4f}", fontsize=10)

plt.suptitle(
    "Within-cell diffusion variability: mitotic vs interphase",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "variability_mitotic_vs_interphase.png",
             dpi=130, bbox_inches="tight")
plt.close(fig2)
print("  Saved variability_mitotic_vs_interphase.png")

# ---------------------------------------------------------------------------
# Figure 3: Per-mitotic-cell variability ordered by circularity and area
# ---------------------------------------------------------------------------
print("Generating per-mitotic-cell variability figure …")

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 9), dpi=120)

for row_i, sort_col in enumerate(["circularity", "area_px"]):
    sort_label = "Circularity" if sort_col == "circularity" else "Cell area (px²)"
    df_mit_s = df_mit.dropna(subset=["D_iqr", "D_cv", sort_col]).sort_values(sort_col)
    n_m = len(df_mit_s)
    positions = np.arange(n_m)
    cmap_vals  = plt.cm.coolwarm(np.linspace(0, 1, n_m))

    for col_i, (var_col, var_label) in enumerate([("D_iqr", "D IQR  (µm²/s)"),
                                                   ("D_cv",  "D CV  (std/mean)")]):
        ax = axes3[row_i, col_i]
        vals = df_mit_s[var_col].values

        bars = ax.bar(positions, vals, color=cmap_vals, alpha=0.85,
                      edgecolor="white", width=0.7, zorder=2)

        # colour border by experiment
        exp_list2 = sorted(df_mit_s["experiment"].unique())
        exp_cmap2 = {e: plt.cm.Set1(i / max(len(exp_list2) - 1, 1))
                     for i, e in enumerate(exp_list2)}
        for pos, (_, row) in zip(positions, df_mit_s.iterrows()):
            ax.bar(pos, row[var_col], color="none",
                   edgecolor=exp_cmap2[row["experiment"]],
                   linewidth=2, width=0.7, zorder=3)

        # colour bar = sort value
        sm = plt.cm.ScalarMappable(cmap="coolwarm",
             norm=plt.Normalize(df_mit_s[sort_col].min(),
                                df_mit_s[sort_col].max()))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label(sort_label, fontsize=8)

        labels = [f"C{int(r['cell_label'])}\n{r['experiment'][:6]}"
                  for _, r in df_mit_s.iterrows()]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
        ax.set_ylabel(var_label, fontsize=9)
        ax.set_title(
            f"Mitotic cells — {var_label}  sorted by {sort_label}\n"
            f"(bar colour = {sort_col}, border = experiment)",
            fontsize=9,
        )

        # Spearman rho annotation
        r_s, p_s = stats.spearmanr(df_mit_s[sort_col].values,
                                    df_mit_s[var_col].values)
        ax.text(0.98, 0.97, f"Spearman ρ={r_s:.2f}  {sig_stars(p_s)}  p={p_s:.3f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Legend for experiments
        handles = [plt.Rectangle((0, 0), 1, 1, color=exp_cmap2[e], label=e)
                   for e in exp_list2]
        ax.legend(handles=handles, fontsize=7, title="Experiment",
                  loc="upper left", framealpha=0.8)

plt.suptitle(
    f"Individual mitotic cell diffusion variability  [{len(df_mit)} cells]",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig3.savefig(OUTPUT_DIR / "variability_mitotic_cells.png",
             dpi=130, bbox_inches="tight")
plt.close(fig3)
print("  Saved variability_mitotic_cells.png")

# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
print("\nWriting stats report …")

mit_iqr = df_mit["D_iqr"].dropna().values
int_iqr = df_int["D_iqr"].dropna().values
mit_cv  = df_mit["D_cv"].dropna().values
int_cv  = df_int["D_cv"].dropna().values

_, p_iqr = stats.mannwhitneyu(mit_iqr, int_iqr, alternative="two-sided")
_, p_cv  = stats.mannwhitneyu(mit_cv,  int_cv,  alternative="two-sided")

with open(OUTPUT_DIR / "variability_stats.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("Step 6 — Within-cell Variability Report\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Input: {INPUT_CSV}\n")
    f.write(f"Total cells: {n_total}  (mitotic: {n_mit}, interphase: {n_int})\n")
    f.write(f"Experiments: {', '.join(sorted(df['experiment'].unique()))}\n\n")

    f.write("--- Variability summary ---\n")
    for label, vals_m, vals_i in [
        ("D_iqr", mit_iqr, int_iqr),
        ("D_std", df_mit['D_std'].dropna().values, df_int['D_std'].dropna().values),
        ("D_cv",  mit_cv,  int_cv),
    ]:
        _, pv = stats.mannwhitneyu(vals_m, vals_i, alternative="two-sided")
        f.write(f"\n  {label}:\n")
        f.write(f"    Mitotic    median={np.median(vals_m):.5f}  IQR={np.percentile(vals_m,75)-np.percentile(vals_m,25):.5f}  n={len(vals_m)}\n")
        f.write(f"    Interphase median={np.median(vals_i):.5f}  IQR={np.percentile(vals_i,75)-np.percentile(vals_i,25):.5f}  n={len(vals_i)}\n")
        f.write(f"    Mann-Whitney p = {pv:.4e}  [{sig_stars(pv)}]\n")

    f.write("\n--- Correlations: variability vs morphology ---\n")
    for c in results_corr:
        f.write(f"\n  [{c['cell_type']}]  {c['y']}  vs  {c['x']}  (n={c['n']}):\n")
        f.write(f"    Pearson  r={c['pearson_r']:.3f}   p={c['pearson_p']:.4e}  [{sig_stars(c['pearson_p'])}]\n")
        f.write(f"    Spearman r={c['spearman_r']:.3f}  p={c['spearman_p']:.4e}  [{sig_stars(c['spearman_p'])}]\n")

    f.write("\n--- Per-mitotic-cell variability vs circularity (Spearman) ---\n")
    df_m2 = df_mit.dropna(subset=["D_iqr", "D_cv", "circularity", "area_px"])
    for vc in ["D_iqr", "D_cv"]:
        for sc in ["circularity", "area_px"]:
            rs, ps = stats.spearmanr(df_m2[sc], df_m2[vc])
            f.write(f"  {vc} vs {sc}: rho={rs:.3f}  p={ps:.4e}  [{sig_stars(ps)}]\n")

print("  Saved variability_stats.txt")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 6 — WITHIN-CELL VARIABILITY SUMMARY")
print("=" * 55)
print(f"\nMitotic  D_iqr median = {np.median(mit_iqr):.5f} µm²/s")
print(f"Interphase D_iqr median = {np.median(int_iqr):.5f} µm²/s")
print(f"Mitotic vs Interphase D_iqr: MW p = {p_iqr:.4f}  [{sig_stars(p_iqr)}]")
print()
print(f"Mitotic  D_cv median = {np.median(mit_cv):.3f}")
print(f"Interphase D_cv median = {np.median(int_cv):.3f}")
print(f"Mitotic vs Interphase D_cv: MW p = {p_cv:.4f}  [{sig_stars(p_cv)}]")
print()
print("Correlations (mitotic cells only):")
df_m2 = df_mit.dropna(subset=["D_iqr", "D_cv", "circularity", "area_px"])
for vc in ["D_iqr", "D_cv"]:
    for sc in ["circularity", "area_px"]:
        rs, ps = stats.spearmanr(df_m2[sc], df_m2[vc])
        print(f"  {vc} vs {sc}: rho={rs:.3f}  p={ps:.4f}  [{sig_stars(ps)}]")
print()
print(f"Results saved to: {OUTPUT_DIR}/")
print("\nStep 6 complete.")

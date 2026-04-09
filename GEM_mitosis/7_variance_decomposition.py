#!/usr/bin/env python3
"""
7_variance_decomposition.py

Nested ANOVA variance decomposition of GEM diffusion heterogeneity,
following the approach in Hubatsch et al. (2023) Biophys J (PMC10027447).

Partitions total variance in log(D) into three hierarchical components:
  1. Between-experiment  (experiment-to-experiment)
  2. Between-cell        (cell-to-cell, within experiments)
  3. Within-cell         (track-to-track, within cells)

Uses eta-squared (η²) = SS_component / SS_total as the main metric —
the fraction of total log(D) variance attributable to each level.

Analysis is performed:
  - On all filtered cells combined
  - Separately for mitotic and interphase cells

Additionally correlates per-cell within-cell SD and between-cell SD
with cell morphology (circularity, area).

Outputs → pooled_results_v2/
  variance_decomposition.png        – η² bar chart + per-cell SD scatter
  variance_within_cell_vs_shape.png – within-cell SD vs circularity/area
  variance_decomposition_stats.txt  – full numerical report

Usage:
  python 7_variance_decomposition.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent

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

print("=" * 60)
print("Step 7 — Variance decomposition")
print("=" * 60)
_default_pooled = str(SCRIPT_DIR / "pooled_results_v2")
OUTPUT_DIR = _ask_folder(
    "Folder containing pooled_cells.csv  (pooled_results_v2/)",
    default=_default_pooled,
)
print(f"  Pooled results folder: {OUTPUT_DIR}\n")
OUTPUT_DIR.mkdir(exist_ok=True)

# Experiments folder: auto-detected as <pooled_dir>/../experiments or prompted
_default_exp = str(OUTPUT_DIR.parent / "experiments")
if Path(_default_exp).is_dir():
    EXP_DIR = Path(_default_exp)
else:
    EXP_DIR = _ask_folder(
        "Folder containing per-experiment subfolders (experiments/)",
        default=_default_exp,
    )

COLOURS = {"mitotic": "#e64b35", "interphase": "#4dbbd5", "all": "#7e6ebf"}

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ---------------------------------------------------------------------------
# Load per-trajectory data from all experiments
# Restrict to cells that passed the pooled filter (in pooled_cells.csv)
# ---------------------------------------------------------------------------
print("Loading data …")

pooled = pd.read_csv(OUTPUT_DIR / "pooled_cells.csv")
# Build a lookup: (experiment, cell_label) → (cell_type, circularity, area_px)
pooled["_key"] = pooled["experiment"] + "_" + pooled["cell_label"].astype(str)
cell_meta = pooled.set_index("_key")[["cell_type","circularity","area_px","n_valid_traj"]]

# Map experiment label → path to diffusion_per_traj.csv
# Scan all subfolders of EXP_DIR for diffusion_per_traj.csv
traj_files = {}
if EXP_DIR.is_dir():
    for sub in sorted(EXP_DIR.iterdir()):
        f = sub / "diffusion_per_traj.csv"
        if f.exists():
            traj_files[sub.name] = f
if not traj_files:
    print(f"  WARNING: no diffusion_per_traj.csv files found under {EXP_DIR}")
    print("  Run 5_batch_pipeline.py first to generate per-experiment outputs.")

frames = []
for exp, fpath in traj_files.items():
    df = pd.read_csv(fpath)
    df = df[df["filtered"] == "ok"].copy()
    df["experiment"] = exp
    df["_key"] = exp + "_" + df["cell_label"].astype(str)
    # keep only cells that are in pooled_cells (passed quality filter)
    df = df[df["_key"].isin(cell_meta.index)].copy()
    df = df.join(cell_meta, on="_key")
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)
df_all = df_all[df_all["D"] > 0].copy()
df_all["logD"] = np.log(df_all["D"])

print(f"  {len(df_all):,} trajectories in {df_all['_key'].nunique()} cells "
      f"({df_all['experiment'].nunique()} experiments)")
print(f"  Mitotic trajectories:    {(df_all['cell_type']=='mitotic').sum():,}")
print(f"  Interphase trajectories: {(df_all['cell_type']=='interphase').sum():,}\n")

# ---------------------------------------------------------------------------
# Nested ANOVA variance decomposition
# Levels: experiment → cell → trajectory
# ---------------------------------------------------------------------------

def nested_anova(df, label="all"):
    """
    Partition SS_total into between-experiment, between-cell, within-cell.
    Returns dict with SS, MS, df, eta-squared for each component.
    """
    grand_mean = df["logD"].mean()
    SS_total   = ((df["logD"] - grand_mean)**2).sum()
    df_total   = len(df) - 1

    # Between-experiment SS
    exp_means = df.groupby("experiment")["logD"].mean()
    exp_n     = df.groupby("experiment")["logD"].count()
    SS_exp    = ((exp_means - grand_mean)**2 * exp_n).sum()
    df_exp    = len(exp_means) - 1

    # Between-cell SS (within experiments)
    cell_means = df.groupby("_key")["logD"].mean()
    cell_n     = df.groupby("_key")["logD"].count()
    # each cell's contribution relative to its experiment mean
    cell_exp   = df.drop_duplicates("_key").set_index("_key")["experiment"]
    SS_cell = 0.0
    for key, n in cell_n.items():
        exp = cell_exp[key]
        SS_cell += n * (cell_means[key] - exp_means[exp])**2
    df_cell = len(cell_means) - len(exp_means)

    # Within-cell SS (track-to-track)
    df["_cell_mean"] = df["_key"].map(cell_means)
    SS_within = ((df["logD"] - df["_cell_mean"])**2).sum()
    df_within  = len(df) - len(cell_means)

    eta2_exp    = SS_exp    / SS_total
    eta2_cell   = SS_cell   / SS_total
    eta2_within = SS_within / SS_total

    print(f"[{label}]  N_traj={len(df)}  N_cells={df['_key'].nunique()}  N_exp={df['experiment'].nunique()}")
    print(f"  SS_total={SS_total:.3f}  (grand mean log D={grand_mean:.3f})")
    print(f"  η² between-experiment = {eta2_exp:.3f}  ({eta2_exp*100:.1f}%)")
    print(f"  η² between-cell       = {eta2_cell:.3f}  ({eta2_cell*100:.1f}%)")
    print(f"  η² within-cell        = {eta2_within:.3f}  ({eta2_within*100:.1f}%)")
    print()

    return dict(
        label=label, N=len(df), N_cells=df["_key"].nunique(),
        grand_mean_logD=grand_mean,
        SS_total=SS_total,
        SS_exp=SS_exp,    eta2_exp=eta2_exp,    df_exp=df_exp,
        SS_cell=SS_cell,  eta2_cell=eta2_cell,  df_cell=df_cell,
        SS_within=SS_within, eta2_within=eta2_within, df_within=df_within,
    )


print("=" * 60)
print("NESTED ANOVA — log(D) variance decomposition")
print("=" * 60 + "\n")

res_all = nested_anova(df_all.copy(), "All cells")
res_mit = nested_anova(df_all[df_all["cell_type"]=="mitotic"].copy(), "Mitotic")
res_int = nested_anova(df_all[df_all["cell_type"]=="interphase"].copy(), "Interphase")

# ---------------------------------------------------------------------------
# Per-cell within-cell statistics (SD, CV of log D)
# ---------------------------------------------------------------------------
def per_cell_stats(df):
    """Compute within-cell SD, CV of log(D) per cell."""
    g = df.groupby("_key")["logD"]
    stats_df = pd.DataFrame({
        "logD_mean": g.mean(),
        "logD_std":  g.std(),
        "n_traj":    g.count(),
    }).reset_index()
    # attach morphology
    stats_df = stats_df.join(cell_meta, on="_key")
    stats_df["logD_cv"] = stats_df["logD_std"] / stats_df["logD_mean"].abs()
    return stats_df

cs_all = per_cell_stats(df_all)
cs_mit = per_cell_stats(df_all[df_all["cell_type"]=="mitotic"])
cs_int = per_cell_stats(df_all[df_all["cell_type"]=="interphase"])

# ---------------------------------------------------------------------------
# Figure 1: η² bar chart + per-cell SD distributions
# ---------------------------------------------------------------------------
print("Generating variance decomposition figure …")

components = ["Between-experiment", "Between-cell\n(within exp)", "Within-cell\n(track-to-track)"]
bar_colours = ["#f39b7f", "#8491b4", "#91d1c2"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=130)

# ---- Panel A: η² stacked bar for all/mitotic/interphase ----
ax = axes[0]
groups   = [res_all, res_mit, res_int]
glabels  = ["All cells", "Mitotic", "Interphase"]
x        = np.arange(len(groups))
bottoms  = np.zeros(len(groups))

eta2_matrix = np.array([
    [r["eta2_exp"], r["eta2_cell"], r["eta2_within"]]
    for r in groups
])

bars_list = []
for j, (comp, col) in enumerate(zip(components, bar_colours)):
    b = ax.bar(x, eta2_matrix[:, j], bottom=bottoms, color=col,
               edgecolor="white", width=0.5, label=comp)
    bars_list.append(b)
    # label inside bar if big enough
    for xi, (bot, val) in enumerate(zip(bottoms, eta2_matrix[:, j])):
        if val > 0.04:
            ax.text(xi, bot + val/2, f"{val*100:.0f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    bottoms += eta2_matrix[:, j]

ax.set_xticks(x)
ax.set_xticklabels(glabels, fontsize=10)
ax.set_ylabel("η²  (fraction of total log D variance)", fontsize=10)
ax.set_title("Nested ANOVA variance decomposition\nlog(D)", fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="upper right")

# ---- Panel B: within-cell SD per cell, mitotic vs interphase ----
ax = axes[1]
mit_wsd = cs_mit["logD_std"].dropna().values
int_wsd = cs_int["logD_std"].dropna().values
_, p_wsd = stats.mannwhitneyu(mit_wsd, int_wsd, alternative="two-sided")

parts = ax.violinplot([mit_wsd, int_wsd], positions=[1, 2],
                      showmedians=True, showextrema=False, widths=0.6)
for pc, col in zip(parts["bodies"], [COLOURS["mitotic"], COLOURS["interphase"]]):
    pc.set_facecolor(col); pc.set_alpha(0.75)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

rng = np.random.default_rng(42)
for pos, vals, col in [(1, mit_wsd, COLOURS["mitotic"]),
                        (2, int_wsd, COLOURS["interphase"])]:
    jit = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(pos + jit, vals, s=30, color=col, alpha=0.85,
               edgecolors="white", linewidths=0.4, zorder=3)

y_max = max(np.nanmax(mit_wsd), np.nanmax(int_wsd)) * 1.05
ax.plot([1, 2], [y_max, y_max], "k-", lw=1)
ax.text(1.5, y_max * 1.02, sig_stars(p_wsd), ha="center", va="bottom", fontsize=13)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Mitotic\n(n={len(mit_wsd)})", f"Interphase\n(n={len(int_wsd)})"])
ax.set_ylabel("Within-cell SD of log(D)", fontsize=10)
ax.set_title(f"Within-cell variability\n(SD of log D per cell)\nMW p={p_wsd:.4f}", fontsize=10)

# ---- Panel C: between-cell SD per experiment, mitotic vs interphase ----
ax = axes[2]
# between-cell SD = SD of per-cell mean log(D) within each experiment×cell_type group
bc_rows = []
for exp in sorted(df_all["experiment"].unique()):
    for ct in ["mitotic", "interphase"]:
        sub = cs_all[(cs_all["_key"].str.startswith(exp)) & (cs_all["cell_type"]==ct)]
        if len(sub) >= 2:
            bc_rows.append({"experiment": exp, "cell_type": ct,
                            "between_cell_SD": sub["logD_mean"].std()})
bc_df = pd.DataFrame(bc_rows)

x_pos = 0
xtick_pos, xtick_lbl = [], []
for exp in sorted(bc_df["experiment"].unique()):
    for ct in ["mitotic", "interphase"]:
        row = bc_df[(bc_df["experiment"]==exp) & (bc_df["cell_type"]==ct)]
        if len(row) == 0: continue
        x_pos += 1
        val = row["between_cell_SD"].values[0]
        ax.bar(x_pos, val, color=COLOURS[ct], alpha=0.8, edgecolor="white", width=0.7)
        xtick_pos.append(x_pos)
        xtick_lbl.append(f"{exp[:8]}\n{ct[:3]}")
    x_pos += 0.4

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=7)
ax.set_ylabel("Between-cell SD of mean log(D)", fontsize=10)
ax.set_title("Cell-to-cell variability\nper experiment", fontsize=10)

plt.suptitle("Variance decomposition of GEM diffusion heterogeneity\n"
             f"(nested ANOVA on log D, {df_all['_key'].nunique()} cells, "
             f"{len(df_all):,} trajectories, {df_all['experiment'].nunique()} experiments)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "variance_decomposition.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved variance_decomposition.png")

# ---------------------------------------------------------------------------
# Figure 2: Within-cell SD vs morphology (mitotic focus)
# ---------------------------------------------------------------------------
print("Generating within-cell SD vs morphology figure …")

shape_cols = [("circularity","Circularity"), ("area_px","Cell area (px²)")]
metrics    = [("logD_std","Within-cell SD of log(D)"),
              ("logD_mean","Mean log(D)  (cell average diffusivity)")]

fig2, axes2 = plt.subplots(len(metrics), len(shape_cols)*2,
                            figsize=(5*len(shape_cols)*2, 4*len(metrics)), dpi=120)

for row_i, (met_col, met_label) in enumerate(metrics):
    for col_i, (shp_col, shp_label) in enumerate(shape_cols):
        for k, (ct, cs) in enumerate([("mitotic", cs_mit), ("interphase", cs_int)]):
            ax = axes2[row_i, col_i*2 + k]
            sub = cs.dropna(subset=[met_col, shp_col])
            x = sub[shp_col].values.astype(float)
            y = sub[met_col].values.astype(float)

            # per-experiment colouring
            exp_list = sorted(sub["experiment"].unique()) if "experiment" in sub.columns else []
            exp_cmap = {e: plt.cm.Set2(i / max(len(exp_list)-1, 1))
                        for i, e in enumerate(exp_list)}

            if "experiment" in sub.columns:
                for exp, grp in sub.groupby("experiment"):
                    ax.scatter(grp[shp_col], grp[met_col], s=50,
                               color=COLOURS[ct], edgecolors=exp_cmap[exp][:3],
                               linewidths=1.2, alpha=0.85, label=exp, zorder=3)
            else:
                ax.scatter(x, y, s=50, color=COLOURS[ct], alpha=0.85, zorder=3)

            if len(x) >= 4:
                slope, intercept, r, p, _ = stats.linregress(x, y)
                xl = np.array([x.min(), x.max()])
                ax.plot(xl, slope*xl + intercept, "k-", lw=1.5, zorder=5)
                rs, ps = stats.spearmanr(x, y)
                ax.text(0.05, 0.95,
                        f"Pearson r={r:.2f} {sig_stars(p)}\nSpearman ρ={rs:.2f} {sig_stars(ps)}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            ax.set_xlabel(shp_label, fontsize=9)
            ax.set_ylabel(met_label if (col_i*2+k)==0 else "", fontsize=9)
            ct_label = "Mitotic" if ct=="mitotic" else "Interphase"
            ax.set_title(f"{met_label[:30]}\nvs {shp_label} — {ct_label} (n={len(sub)})",
                         fontsize=8)

plt.suptitle("Within-cell and mean diffusivity vs cell morphology\n"
             "(per-cell statistics from nested log D distribution)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "variance_within_cell_vs_shape.png", dpi=130, bbox_inches="tight")
plt.close(fig2)
print("  Saved variance_within_cell_vs_shape.png")

# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
print("Writing stats report …")

with open(OUTPUT_DIR / "variance_decomposition_stats.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("Step 7 — Nested ANOVA Variance Decomposition Report\n")
    f.write("(following Hubatsch et al. 2023, Biophys J, PMC10027447)\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Total trajectories: {len(df_all):,}  |  Cells: {df_all['_key'].nunique()}  |  "
            f"Experiments: {df_all['experiment'].nunique()}\n\n")

    for res in [res_all, res_mit, res_int]:
        f.write(f"--- [{res['label']}]  N_traj={res['N']}, N_cells={res['N_cells']} ---\n")
        f.write(f"  Grand mean log(D) = {res['grand_mean_logD']:.4f}\n")
        f.write(f"  SS_total = {res['SS_total']:.4f}\n")
        f.write(f"  η² between-experiment  = {res['eta2_exp']:.4f}  ({res['eta2_exp']*100:.1f}%)\n")
        f.write(f"  η² between-cell        = {res['eta2_cell']:.4f}  ({res['eta2_cell']*100:.1f}%)\n")
        f.write(f"  η² within-cell         = {res['eta2_within']:.4f}  ({res['eta2_within']*100:.1f}%)\n\n")

    f.write("--- Within-cell SD of log(D): mitotic vs interphase ---\n")
    f.write(f"  Mitotic    median={np.median(mit_wsd):.4f}  IQR={np.percentile(mit_wsd,75)-np.percentile(mit_wsd,25):.4f}\n")
    f.write(f"  Interphase median={np.median(int_wsd):.4f}  IQR={np.percentile(int_wsd,75)-np.percentile(int_wsd,25):.4f}\n")
    f.write(f"  Mann-Whitney p = {p_wsd:.4e}  [{sig_stars(p_wsd)}]\n\n")

    f.write("--- Correlations: within-cell SD of log(D) vs morphology ---\n")
    for ct, cs in [("Mitotic", cs_mit), ("Interphase", cs_int)]:
        for shp_col, shp_label in shape_cols:
            sub = cs.dropna(subset=["logD_std", shp_col])
            if len(sub) < 4: continue
            r, p = stats.pearsonr(sub[shp_col], sub["logD_std"])
            rs, ps = stats.spearmanr(sub[shp_col], sub["logD_std"])
            f.write(f"  [{ct}] within-cell SD vs {shp_col} (n={len(sub)}):\n")
            f.write(f"    Pearson  r={r:.3f}   p={p:.4e}  [{sig_stars(p)}]\n")
            f.write(f"    Spearman ρ={rs:.3f}  p={ps:.4e}  [{sig_stars(ps)}]\n")

    f.write("\n--- Correlations: mean log(D) per cell vs morphology ---\n")
    for ct, cs in [("Mitotic", cs_mit), ("Interphase", cs_int)]:
        for shp_col, shp_label in shape_cols:
            sub = cs.dropna(subset=["logD_mean", shp_col])
            if len(sub) < 4: continue
            r, p = stats.pearsonr(sub[shp_col], sub["logD_mean"])
            rs, ps = stats.spearmanr(sub[shp_col], sub["logD_mean"])
            f.write(f"  [{ct}] mean log(D) vs {shp_col} (n={len(sub)}):\n")
            f.write(f"    Pearson  r={r:.3f}   p={p:.4e}  [{sig_stars(p)}]\n")
            f.write(f"    Spearman ρ={rs:.3f}  p={ps:.4e}  [{sig_stars(ps)}]\n")

print("  Saved variance_decomposition_stats.txt")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 — NESTED ANOVA SUMMARY")
print("=" * 60)
for res in [res_all, res_mit, res_int]:
    print(f"\n[{res['label']}]")
    print(f"  η² between-experiment = {res['eta2_exp']*100:.1f}%")
    print(f"  η² between-cell       = {res['eta2_cell']*100:.1f}%")
    print(f"  η² within-cell        = {res['eta2_within']*100:.1f}%")
print(f"\nWithin-cell SD mitotic vs interphase: MW p={p_wsd:.4f} [{sig_stars(p_wsd)}]")
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nStep 7 complete.")

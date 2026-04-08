#!/usr/bin/env python3
"""
8_confound_check.py

Rigorously tests whether the cell-area vs mean-diffusivity correlation
is an artefact of trajectory-count variability per cell.

Three complementary approaches:
  1. Confound audit    – Spearman correlations among area, n_traj, D_median
  2. Bootstrap downsample – subsample k_min trajectories per cell 1000×,
                           recompute mean log(D), retest correlation with area
  3. Partial correlation  – area vs mean log(D) controlling for n_traj
                           (Spearman rank-based partial correlation)

Analysis is done for all cells, mitotic, and interphase separately.

Outputs → pooled_results_v2/
  confound_check.png
  confound_check_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
EXP_DIR    = SCRIPT_DIR / "experiments"
OUTPUT_DIR = SCRIPT_DIR / "pooled_results_v2"
N_BOOT     = 2000          # bootstrap iterations
SEED       = 42
RNG        = np.random.default_rng(SEED)
COLOURS    = {"mitotic": "#e64b35", "interphase": "#4dbbd5", "all": "#7e6ebf"}

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
pooled = pd.read_csv(OUTPUT_DIR / "pooled_cells.csv")
pooled["_key"] = pooled["experiment"] + "_" + pooled["cell_label"].astype(str)

traj_files = {"set1_3em_001": SCRIPT_DIR / "diffusion_per_traj.csv"}
for sub in sorted(EXP_DIR.iterdir()):
    f = sub / "diffusion_per_traj.csv"
    if f.exists() and sub.name != "set1_3em_001":
        traj_files[sub.name] = f

frames = []
for exp, fpath in traj_files.items():
    df = pd.read_csv(fpath)
    df = df[df["filtered"] == "ok"].copy()
    df["experiment"] = exp
    df["_key"] = exp + "_" + df["cell_label"].astype(str)
    frames.append(df)

df_traj = pd.concat(frames, ignore_index=True)
df_traj = df_traj[df_traj["D"] > 0].copy()
df_traj["logD"] = np.log(df_traj["D"])

# restrict to pooled (quality-filtered) cells
df_traj = df_traj[df_traj["_key"].isin(pooled["_key"])].copy()

# attach morphology + cell_type
meta = pooled.set_index("_key")[["cell_type", "circularity", "area_px", "n_valid_traj"]]
df_traj = df_traj.join(meta, on="_key")

# per-cell summary (actual traj count from per-traj table)
cell_df = (
    df_traj.groupby("_key")["logD"]
    .agg(n_traj="count", mean_logD="mean", std_logD="std")
    .reset_index()
    .join(meta, on="_key")
)

print(f"  {len(df_traj):,} trajectories, {len(cell_df)} cells")
print(f"  n_traj per cell: min={cell_df['n_traj'].min()}, "
      f"median={cell_df['n_traj'].median():.0f}, max={cell_df['n_traj'].max()}")

K_MIN = int(cell_df["n_traj"].min())   # subsample to this many trajs per cell
print(f"  Subsampling to k_min = {K_MIN} trajectories per cell\n")


# ---------------------------------------------------------------------------
# Helper: Spearman partial correlation  r(X,Y | Z)
# Regress out Z from both X and Y using rank residuals, correlate residuals
# ---------------------------------------------------------------------------
def partial_spearman(x, y, z):
    """Spearman partial correlation r(x,y|z) via rank residuals."""
    rx  = stats.rankdata(x)
    ry  = stats.rankdata(y)
    rz  = stats.rankdata(z)
    # regress rz out of rx and ry
    def residuals(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    res_x = residuals(rx, rz)
    res_y = residuals(ry, rz)
    r, p = stats.pearsonr(res_x, res_y)   # Pearson on rank residuals ≈ partial Spearman
    return r, p


# ---------------------------------------------------------------------------
# Analysis function (run on a cell_df subset + matching traj rows)
# ---------------------------------------------------------------------------
def analyse(cell_sub, traj_sub, label):
    print(f"[{label}]  {len(cell_sub)} cells, {len(traj_sub)} trajectories")

    area  = cell_sub["area_px"].values.astype(float)
    ntraj = cell_sub["n_traj"].values.astype(float)
    mlogD = cell_sub["mean_logD"].values.astype(float)

    # --- 1. Confound audit ---
    r_an, p_an = stats.spearmanr(area, ntraj)
    r_ad, p_ad = stats.spearmanr(area, mlogD)
    r_nd, p_nd = stats.spearmanr(ntraj, mlogD)
    print(f"  Spearman: area vs n_traj  rho={r_an:.3f} p={p_an:.4f} {sig_stars(p_an)}")
    print(f"  Spearman: area vs logD    rho={r_ad:.3f} p={p_ad:.4f} {sig_stars(p_ad)}")
    print(f"  Spearman: n_traj vs logD  rho={r_nd:.3f} p={p_nd:.4f} {sig_stars(p_nd)}")

    # --- 2. Bootstrap downsample ---
    keys = cell_sub["_key"].values
    boot_r = []
    for _ in range(N_BOOT):
        sampled_means = []
        for key in keys:
            pool = traj_sub.loc[traj_sub["_key"] == key, "logD"].values
            s = RNG.choice(pool, size=K_MIN, replace=False)
            sampled_means.append(s.mean())
        r_boot, _ = stats.spearmanr(area, sampled_means)
        boot_r.append(r_boot)

    boot_r = np.array(boot_r)
    r_med  = np.median(boot_r)
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    # p-value: fraction of bootstrap samples with r <= 0
    p_boot = 2 * min((boot_r <= 0).mean(), (boot_r >= 0).mean())
    p_boot = max(p_boot, 1 / N_BOOT)   # floor at 1/N_BOOT
    print(f"  Bootstrap (k={K_MIN}, n={N_BOOT}): "
          f"median rho={r_med:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"p≈{p_boot:.4f} {sig_stars(p_boot)}")

    # --- 3. Partial correlation ---
    rp, pp = partial_spearman(mlogD, area, ntraj)
    print(f"  Partial Spearman r(area, logD | n_traj) = {rp:.3f}  p={pp:.4f} {sig_stars(pp)}")
    print()

    return dict(
        label=label, n_cells=len(cell_sub),
        r_area_ntraj=r_an, p_area_ntraj=p_an,
        r_area_logD=r_ad, p_area_logD=p_ad,
        r_ntraj_logD=r_nd, p_ntraj_logD=p_nd,
        boot_r=boot_r, boot_r_med=r_med, boot_ci_lo=ci_lo, boot_ci_hi=ci_hi, p_boot=p_boot,
        r_partial=rp, p_partial=pp,
    )


print("=" * 60)
subsets = [
    ("All cells",   cell_df,                                  df_traj),
    ("Mitotic",     cell_df[cell_df["cell_type"]=="mitotic"], df_traj[df_traj["cell_type"]=="mitotic"]),
    ("Interphase",  cell_df[cell_df["cell_type"]=="interphase"], df_traj[df_traj["cell_type"]=="interphase"]),
]
results = []
for lbl, cs, ts in subsets:
    results.append(analyse(cs.reset_index(drop=True), ts, lbl))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print("Generating figures …")

fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=130)

for col_i, (res, cs, ts) in enumerate(zip(results, [s[1] for s in subsets], [s[2] for s in subsets])):
    lbl = res["label"]
    ct_key = "mitotic" if "Mitotic" in lbl else ("interphase" if "Interphase" in lbl else "all")
    col = COLOURS[ct_key]

    area  = cs["area_px"].values.astype(float)
    ntraj = cs["n_traj"].values.astype(float)
    mlogD = cs["mean_logD"].values.astype(float)

    # ---- Top row: raw scatter area vs mean log(D), sized by n_traj ----
    ax = axes[0, col_i]
    sc = ax.scatter(area, mlogD, s=30 + ntraj * 8, c=ntraj,
                    cmap="YlOrRd", alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="n trajectories")

    slope, intercept, r_raw, p_raw, _ = stats.linregress(area, mlogD)
    xl = np.array([area.min(), area.max()])
    ax.plot(xl, slope * xl + intercept, "k-", lw=1.5, zorder=5)

    ax.text(0.05, 0.95,
            f"Raw Spearman ρ={res['r_area_logD']:.2f} {sig_stars(res['p_area_logD'])}\n"
            f"Partial ρ={res['r_partial']:.2f} {sig_stars(res['p_partial'])}\n"
            f"(controlling for n_traj)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("Cell area  (px²)", fontsize=10)
    ax.set_ylabel("Mean log(D)  per cell", fontsize=10)
    ax.set_title(f"{lbl}  (n={len(cs)})\nRaw correlation + partial", fontsize=10)

    # ---- Bottom row: bootstrap distribution of rho ----
    ax = axes[1, col_i]
    boot_r = res["boot_r"]
    ax.hist(boot_r, bins=60, color=col, alpha=0.75, edgecolor="white", density=True)
    ax.axvline(res["boot_r_med"], color="black",  lw=2,   ls="-",  label=f"median ρ={res['boot_r_med']:.2f}")
    ax.axvline(res["boot_ci_lo"], color="black",  lw=1.5, ls="--", label=f"95% CI [{res['boot_ci_lo']:.2f}, {res['boot_ci_hi']:.2f}]")
    ax.axvline(res["boot_ci_hi"], color="black",  lw=1.5, ls="--")
    ax.axvline(0,                 color="dimgray", lw=1,   ls=":",  label="ρ=0")
    ax.set_xlabel("Bootstrap Spearman ρ  (area vs mean log D)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"{lbl} — Bootstrap downsample\n"
        f"k={K_MIN} traj/cell, n={N_BOOT} iterations  |  p≈{res['p_boot']:.4f} {sig_stars(res['p_boot'])}",
        fontsize=9,
    )
    ax.legend(fontsize=8)

plt.suptitle(
    "Confound check: is area → diffusivity correlation\n"
    "an artefact of unequal trajectory counts per cell?",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "confound_check.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved confound_check.png")

# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "confound_check_stats.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("Step 8 — Confound Check: area vs diffusivity\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Bootstrap iterations: {N_BOOT}   k_min (subsample size): {K_MIN}\n\n")

    for res in results:
        f.write(f"--- [{res['label']}]  n_cells={res['n_cells']} ---\n")
        f.write(f"  1. Confound audit (Spearman):\n")
        f.write(f"     area vs n_traj:    rho={res['r_area_ntraj']:.3f}  p={res['p_area_ntraj']:.4e}  [{sig_stars(res['p_area_ntraj'])}]\n")
        f.write(f"     area vs mean logD: rho={res['r_area_logD']:.3f}  p={res['p_area_logD']:.4e}  [{sig_stars(res['p_area_logD'])}]\n")
        f.write(f"     n_traj vs mean logD: rho={res['r_ntraj_logD']:.3f}  p={res['p_ntraj_logD']:.4e}  [{sig_stars(res['p_ntraj_logD'])}]\n")
        f.write(f"  2. Bootstrap downsample to k={K_MIN} traj/cell:\n")
        f.write(f"     median rho={res['boot_r_med']:.3f}  95% CI [{res['boot_ci_lo']:.3f}, {res['boot_ci_hi']:.3f}]  p≈{res['p_boot']:.4e}  [{sig_stars(res['p_boot'])}]\n")
        f.write(f"  3. Partial Spearman r(area, logD | n_traj):\n")
        f.write(f"     rho={res['r_partial']:.3f}  p={res['p_partial']:.4e}  [{sig_stars(res['p_partial'])}]\n\n")

print("  Saved confound_check_stats.txt")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 — CONFOUND CHECK SUMMARY")
print("=" * 60)
for res in results:
    print(f"\n[{res['label']}]")
    print(f"  Raw Spearman (area vs logD):         rho={res['r_area_logD']:.3f}  {sig_stars(res['p_area_logD'])}")
    print(f"  Bootstrap k={K_MIN} (area vs logD):      rho={res['boot_r_med']:.3f}  CI[{res['boot_ci_lo']:.3f},{res['boot_ci_hi']:.3f}]  {sig_stars(res['p_boot'])}")
    print(f"  Partial Spearman (area|n_traj):      rho={res['r_partial']:.3f}  {sig_stars(res['p_partial'])}")
print("\nStep 8 complete.")

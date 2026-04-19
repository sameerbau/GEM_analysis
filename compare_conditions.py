#!/usr/bin/env python3
"""
compare_conditions.py
Compare GEM diffusion (D) across WT, Rab1 RNAi, and Gnu RNAi conditions.
Flags over-linked embryos (mean track length > MAX_MEAN_TRACK_LEN).

Usage:
    python compare_conditions.py <analysed_data_dir>

    <analysed_data_dir> is the "Analysed data" folder inside the repo.

Outputs (written next to the input dir):
    condition_comparison.csv          per-embryo table with flags
    condition_comparison.png          4-panel figure
    roi_inside_outside.png            WT inside/outside ER-ROI panel
                                      (+ Gnu overlay when files are present)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ── constants ─────────────────────────────────────────────────────────────────
MAX_MEAN_TRACK_LEN = 20.0   # frames; normal ~15–17; >20 = likely over-linking
HARD_EXCLUDE = {"tracked_Traj_Em002.nd2_crop"}   # confirmed zig-zag over-linker

# ── group registry ────────────────────────────────────────────────────────────
# (key, display label, hex colour, relative diff csv, relative counts csv)
GROUPS = [
    ("WT",
     "WT",
     "#4878CF",
     "PB diffusion all/diffusion_summary.csv",
     "Particle counts/Full emrbyo/particle_counts_summary.csv"),

    ("Rab1_R1",
     "Rab1\nR1",
     "#6ACC65",
     "Rab1 RNAi/Rab1 RNAi diffusioin round1/diffusion_summary.csv",
     "Rab1 RNAi/Rab1 RNAi diffusioin round1/particle_counts_summary.csv"),

    ("Rab1_R2",
     "Rab1\nR2",
     "#1d9e45",
     "Rab1 RNAi/Rab1 round2 data/diffusion_summary.csv",
     "Rab1 RNAi/Rab1 round2 data/particle_counts_summary.csv"),

    ("Gnu_Early_R1",
     "Gnu\nEarly R1",
     "#C77CFF",
     "Gnu RNAi/Gnu RNAi Early Round1/diffusion_summary.csv",
     "Gnu RNAi/Gnu RNAi Early Round1/particle_counts_summary.csv"),

    ("Gnu_Later_R1",
     "Gnu\nLater R1",
     "#8B00FF",
     "Gnu RNAi/Gnu RNAi later round1/diffusion_summary.csv",
     "Gnu RNAi/Gnu RNAi later round1/particle_counts_summary.csv"),

    ("Gnu_Early_R2",
     "Gnu\nEarly R2",
     "#FF9A3C",
     "Gnu RNAi/Gnu RNAI Early round2/diffusion_summary.csv",
     "Gnu RNAi/Gnu RNAI Early round2/particle_counts_summary.csv"),

    ("Gnu_Later_R2",
     "Gnu\nLater R2",
     "#E05C00",
     "Gnu RNAi/Gnu RNAi Later Round2/diffusion_summary.csv",
     "Gnu RNAi/Gnu RNAi Later Round2/particle_counts_summary.csv"),

    ("Gnu_FurLater_R2",
     "Gnu\nFurLater R2",
     "#8B2500",
     "Gnu RNAi/Gnu RNAi Further Later Round2/diffusion_summary.csv",
     "Gnu RNAi/Gnu RNAi Further Later Round2/particle_counts_summary.csv"),

    ("NC10",
     "NC10",
     "#D4380D",
     "Nuclear cycles 10/diffusion_summary.csv",
     "Nuclear cycles 10/particle_counts_summary.csv"),
]

# Inside/outside ER-ROI pooled summaries (WT)
WT_ROI_DIFF  = "PB ER pooled_diffusion_analysis/pooled_diffusion_summary.csv"
WT_ROI_ALPHA = "PB ER pooled_alpha_inside_outside_analysis/pooled_alpha_results_summary.csv"
WT_ROI_DIFF_STATS  = "PB ER pooled_diffusion_analysis/pooled_diffusion_stats.csv"
WT_ROI_ALPHA_STATS = "PB ER pooled_alpha_inside_outside_analysis/pooled_alpha_results_statistics.csv"

# Non-WT inside/outside ROI pooled summaries
# Format: list of (diff_summary_path, diff_stats_path, display_label, colour_key)
EXTRA_ROI_ENTRIES = [
    ("Gnu RNAi/Round 1 Lateronly pooled_diffusion_analysis/pooled_diffusion_summary.csv",
     "Gnu RNAi/Round 1 Lateronly pooled_diffusion_analysis/pooled_diffusion_stats.csv",
     "Gnu Later R1",
     "Gnu_Later_R1"),
    ("Nuclear cycles 10/pooled_diffusion_analysis/pooled_diffusion_summary.csv",
     "Nuclear cycles 10/pooled_diffusion_analysis/pooled_diffusion_stats.csv",
     "NC10",
     "NC10"),
]

NC10_ROI_ALPHA      = "Nuclear cycles 10/pooled_alpha_inside_outside_analysis/pooled_alpha_results_summary.csv"
NC10_ROI_ALPHA_STATS = "Nuclear cycles 10/pooled_alpha_inside_outside_analysis/pooled_alpha_results_statistics.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _stem(filename: str) -> str:
    """tracked_Traj_Em6.nd2_crop  →  Em6.nd2_crop"""
    s = str(filename)
    if s.startswith("tracked_Traj_"):
        s = s[len("tracked_Traj_"):]
    return s


def load_group(base: Path, key, label, colour, diff_rel, counts_rel):
    diff_path   = base / diff_rel
    counts_path = base / counts_rel
    if not diff_path.exists():
        print(f"  [skip] {label}: {diff_path} not found")
        return pd.DataFrame()

    diff = pd.read_csv(diff_path)
    diff["stem"] = diff["FileName"].apply(_stem)

    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        counts["stem"] = counts["FileName"].str.replace(r"\.tif$", "", regex=True).str.strip()
        merged = diff.merge(counts[["stem", "MeanParticlesPerFrame"]], on="stem", how="left")
    else:
        merged = diff.copy()
        merged["MeanParticlesPerFrame"] = np.nan

    merged["group"]   = key
    merged["label"]   = label
    merged["colour"]  = colour
    merged["overlink"] = (
        merged["FileName"].isin(HARD_EXCLUDE) |
        (merged["MeanTrackLength"] > MAX_MEAN_TRACK_LEN)
    )
    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python compare_conditions.py <analysed_data_dir>")
    base = Path(sys.argv[1])
    if not base.is_dir():
        sys.exit(f"Not a directory: {base}")

    print("Loading groups...")
    frames = []
    for g in GROUPS:
        df = load_group(base, *g)
        if not df.empty:
            print(f"  {g[1]:20s}  n={len(df):3d}  "
                  f"(flagged over-linkers: {df['overlink'].sum()})")
            frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    # save per-embryo table
    out_dir = base.parent
    csv_out = out_dir / "condition_comparison.csv"
    all_data[[
        "group", "FileName", "MedianDiffusion", "MeanTrackLength",
        "MeanParticlesPerFrame", "NumTracks", "overlink"
    ]].to_csv(csv_out, index=False)
    print(f"\nPer-embryo table: {csv_out}")

    # ── Figure 1: four-panel condition comparison ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("GEM diffusion — condition comparison", fontsize=13, y=0.98)

    group_keys   = [g[0] for g in GROUPS]
    group_labels = [g[1] for g in GROUPS]
    colours      = {g[0]: g[2] for g in GROUPS}

    # ── Panel A: per-embryo D jitter, marker size = density ───────────────────
    ax = axes[0, 0]
    jitter_rng = np.random.default_rng(42)

    density_all = all_data["MeanParticlesPerFrame"].dropna()
    dmin, dmax  = density_all.min(), density_all.max()

    def density_to_size(d):
        if pd.isna(d):
            return 40
        return 20 + 160 * (d - dmin) / (dmax - dmin + 1e-9)

    x_pos = {k: i for i, k in enumerate(group_keys)}
    for _, row in all_data.iterrows():
        xi = x_pos[row["group"]] + jitter_rng.uniform(-0.25, 0.25)
        sz = density_to_size(row["MeanParticlesPerFrame"])
        marker = "X" if row["overlink"] else "o"
        ec     = "red" if row["overlink"] else colours[row["group"]]
        ax.scatter(xi, row["MedianDiffusion"], s=sz,
                   color=colours[row["group"]], marker=marker,
                   edgecolors=ec, linewidths=0.8, alpha=0.85, zorder=3)

    # group medians (filtered)
    for i, key in enumerate(group_keys):
        sub = all_data[(all_data["group"] == key) & ~all_data["overlink"]]
        if len(sub):
            med = sub["MedianDiffusion"].median()
            ax.hlines(med, i - 0.35, i + 0.35, colors=colours[key],
                      linewidths=2.5, zorder=4)

    # WT shaded reference range (filtered)
    wt = all_data[(all_data["group"] == "WT") & ~all_data["overlink"]]
    ax.axhspan(wt["MedianDiffusion"].min(), wt["MedianDiffusion"].max(),
               color="#4878CF", alpha=0.07, zorder=0)

    ax.set_xticks(range(len(group_keys)))
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel("Per-embryo median D (µm²/s)")
    ax.set_title("A  Per-embryo median D by condition\n"
                 "(size = particle density; ✕ = over-linker flagged)")
    ax.axhline(0, color="k", lw=0.5, ls="--")

    # legend for marker size
    for sz_d in [300, 500, 800]:
        ax.scatter([], [], s=density_to_size(sz_d), color="grey", alpha=0.5,
                   label=f"{sz_d} p/frame")
    ax.legend(fontsize=7, title="Density", title_fontsize=7,
              loc="upper right", markerscale=1)

    # ── Panel B: D vs density scatter ─────────────────────────────────────────
    ax = axes[0, 1]
    for key in group_keys:
        sub = all_data[all_data["group"] == key]
        ok  = sub[~sub["overlink"]]
        bad = sub[sub["overlink"]]
        ax.scatter(ok["MeanParticlesPerFrame"],  ok["MedianDiffusion"],
                   color=colours[key], s=50, alpha=0.85, zorder=3)
        ax.scatter(bad["MeanParticlesPerFrame"], bad["MedianDiffusion"],
                   color=colours[key], marker="X", s=60, edgecolors="red",
                   linewidths=0.8, alpha=0.85, zorder=4)

    # WT regression line (filtered, for reference)
    wt_ok = all_data[(all_data["group"] == "WT") & ~all_data["overlink"]].dropna(
        subset=["MeanParticlesPerFrame", "MedianDiffusion"])
    if len(wt_ok) >= 3:
        slope, intercept, r, p, _ = stats.linregress(
            wt_ok["MeanParticlesPerFrame"], wt_ok["MedianDiffusion"])
        xr = np.array([dmin * 0.9, dmax * 1.05])
        ax.plot(xr, slope * xr + intercept,
                color="#4878CF", lw=1.5, ls="--", alpha=0.6,
                label=f"WT trend (r={r:.2f}, p={p:.3f})")
        ax.legend(fontsize=8)

    legend_elems = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=colours[k], markersize=8, label=lab)
                    for k, lab in zip(group_keys, group_labels)]
    ax.legend(handles=legend_elems, fontsize=7, ncol=2, loc="upper right")
    ax.set_xlabel("Mean particles per frame")
    ax.set_ylabel("Per-embryo median D (µm²/s)")
    ax.set_title("B  D vs particle density (✕ = over-linker)")

    # ── Panel C: Inside vs Outside ROI — D ────────────────────────────────────
    ax = axes[1, 0]
    _plot_roi_comparison(ax, base, "D",
                         WT_ROI_DIFF, WT_ROI_DIFF_STATS,
                         EXTRA_ROI_ENTRIES, colours)
    ax.set_title("C  Inside vs Outside ER-ROI: D\n(WT, NC10, Gnu Later R1)")
    ax.set_ylabel("Median D (µm²/s)")

    # ── Panel D: Inside vs Outside ROI — alpha ────────────────────────────────
    ax = axes[1, 1]
    nc10_alpha_entries = [
        (NC10_ROI_ALPHA, NC10_ROI_ALPHA_STATS, "NC10", "NC10"),
    ]
    _plot_roi_comparison(ax, base, "alpha",
                         WT_ROI_ALPHA, WT_ROI_ALPHA_STATS,
                         nc10_alpha_entries, colours)
    ax.set_title("D  Inside vs Outside ER-ROI: α\n(WT and NC10)")
    ax.set_ylabel("Median α  (1 = pure Brownian)")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_out = out_dir / "condition_comparison.png"
    plt.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_out}")

    # ── Figure 2: Inside/Outside detail ───────────────────────────────────────
    _plot_roi_detail(base, out_dir, colours)


def _read_roi_summary(path: Path):
    """Return (inside_median, outside_median, inside_sem, outside_sem, inside_n, outside_n)."""
    df = pd.read_csv(path)
    df[df.columns[0]] = df[df.columns[0]].str.replace("_", " ").str.strip()
    loc = df.columns[0]
    i_row = df[df[loc].str.contains("Inside",  case=False)]
    o_row = df[df[loc].str.contains("Outside", case=False)]
    i_med = i_row["Median"].values[0] if len(i_row) else np.nan
    o_med = o_row["Median"].values[0] if len(o_row) else np.nan
    i_sem = i_row["SEM"].values[0]    if len(i_row) else 0
    o_sem = o_row["SEM"].values[0]    if len(o_row) else 0
    i_n   = i_row["N"].values[0]      if (len(i_row) and "N" in df.columns) else \
            i_row["N_trajectories"].values[0] if (len(i_row) and "N_trajectories" in df.columns) else 0
    o_n   = o_row["N"].values[0]      if (len(o_row) and "N" in df.columns) else \
            o_row["N_trajectories"].values[0] if (len(o_row) and "N_trajectories" in df.columns) else 0
    return i_med, o_med, i_sem, o_sem, int(i_n), int(o_n)


def _plot_roi_comparison(ax, base, metric,
                         wt_path, wt_stats_path,
                         gnu_entries, colours):
    """Grouped bar plot: Inside vs Outside, WT side-by-side with Gnu conditions."""
    wt_file = base / wt_path
    if not wt_file.exists():
        ax.text(0.5, 0.5, "No WT ROI data found", ha="center", va="center",
                transform=ax.transAxes)
        return

    wt_i, wt_o, wt_i_sem, wt_o_sem, wt_in, wt_on = _read_roi_summary(wt_file)

    # collect all conditions that have data
    conditions = [("WT", wt_i, wt_o, wt_i_sem, wt_o_sem, colours["WT"], None)]
    for diff_path, stats_path, glabel, ckey in gnu_entries:
        gfile = base / diff_path
        if gfile.exists():
            gi, go, gi_sem, go_sem, gn_i, gn_o = _read_roi_summary(gfile)
            conditions.append((glabel, gi, go, gi_sem, go_sem, colours[ckey], stats_path))

    n_cond   = len(conditions)
    group_w  = 1.2
    bar_w    = 0.4
    x_groups = np.arange(n_cond) * group_w

    all_tops = []
    for xi, (label, i_med, o_med, i_sem, o_sem, col, spath) in zip(x_groups, conditions):
        b1 = ax.bar(xi - bar_w/2, i_med, bar_w, color=col, alpha=0.85,
                    yerr=i_sem, capsize=3, error_kw={"lw": 1},
                    label="Inside" if xi == x_groups[0] else "")
        b2 = ax.bar(xi + bar_w/2, o_med, bar_w, color=col, alpha=0.40,
                    yerr=o_sem, capsize=3, error_kw={"lw": 1},
                    label="Outside" if xi == x_groups[0] else "")
        all_tops.append(max(i_med + i_sem, o_med + o_sem))

        # significance bracket
        sp = spath or wt_stats_path
        sp_full = base / sp
        if sp_full.exists():
            st = pd.read_csv(sp_full)
            mw = st[st["Test"].str.contains("Mann", case=False)]
            if len(mw):
                p = float(mw["P_value"].values[0])
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                ytop = max(i_med, o_med) * 1.15
                ax.annotate("", xy=(xi + bar_w/2, ytop), xytext=(xi - bar_w/2, ytop),
                            arrowprops=dict(arrowstyle="-", lw=0.8))
                ax.text(xi, ytop * 1.01, sig, ha="center", fontsize=8)

        ax.text(xi, -0.003, label, ha="center", va="top", fontsize=7.5,
                transform=ax.get_xaxis_transform())

    ax.set_xticks([])
    ymax = max(all_tops) * 1.25
    ax.set_ylim(0, ymax)

    # legend for fill = inside/outside
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="grey", alpha=0.85, label="Inside ER-ROI"),
                       Patch(color="grey", alpha=0.40, label="Outside ER-ROI")],
              fontsize=8, loc="upper right")


def _plot_roi_detail(base: Path, out_dir: Path, colours):
    """Dedicated inside/outside figure with D and alpha side-by-side."""
    wt_diff_path  = base / WT_ROI_DIFF
    wt_alpha_path = base / WT_ROI_ALPHA
    if not wt_diff_path.exists() and not wt_alpha_path.exists():
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Inside vs Outside ER-ROI — WT preblastoderm GEM",
                 fontsize=12)

    # D panel
    ax = axes[0]
    if wt_diff_path.exists():
        df = pd.read_csv(wt_diff_path)
        locs   = df["Location"].str.replace("_", " ").tolist()
        meds   = df["Median"].tolist()
        sems   = df["SEM"].tolist()
        q25    = df["Q25"].tolist()
        q75    = df["Q75"].tolist()

        for i, (loc, med, sem, q1, q3) in enumerate(zip(locs, meds, sems, q25, q75)):
            c = "#E05C5C" if "Inside" in loc else "#5C7CE0"
            ax.bar(i, med, width=0.5, color=c, alpha=0.8, label=loc)
            ax.errorbar(i, med, yerr=sem, fmt="none", color="k", capsize=4, lw=1.2)
            ax.plot([i - 0.18, i + 0.18], [q1, q1], color="k", lw=0.8)
            ax.plot([i - 0.18, i + 0.18], [q3, q3], color="k", lw=0.8)
            ax.plot([i - 0.18, i - 0.18, i + 0.18, i + 0.18],
                    [q1, q3, q3, q1], color="k", lw=0.8)

        if (base / WT_ROI_DIFF_STATS).exists():
            st = pd.read_csv(base / WT_ROI_DIFF_STATS)
            mw = st[st["Test"].str.contains("Mann", case=False)]
            if len(mw):
                p = float(mw["P_value"].values[0])
                ymax = max(meds) * 1.3
                ax.annotate("", xy=(1, ymax), xytext=(0, ymax),
                            arrowprops=dict(arrowstyle="-", lw=1))
                ax.text(0.5, ymax * 1.02, f"Mann-Whitney p={p:.1e}",
                        ha="center", fontsize=9)
                ax.set_ylim(0, ymax * 1.2)

        n_in  = df[df["Location"].str.contains("Inside")]["N"].values[0]
        n_out = df[df["Location"].str.contains("Outside")]["N"].values[0]
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Inside\n(n={n_in})", f"Outside\n(n={n_out})"])
        ax.set_ylabel("Median D per track (µm²/s)")
        ax.set_title("Diffusion coefficient D")

    # Alpha panel
    ax = axes[1]
    if wt_alpha_path.exists():
        df = pd.read_csv(wt_alpha_path)
        locs  = df["Location"].tolist()
        meds  = df["Median"].tolist()
        sems  = df["SEM"].tolist()
        ns    = df["N_trajectories"].tolist()
        n_sub = df["N_subdiffusion"].tolist()
        n_nor = df["N_normal"].tolist()
        n_sup = df["N_superdiffusion"].tolist()

        for i, (loc, med, sem) in enumerate(zip(locs, meds, sems)):
            c = "#E05C5C" if "Inside" in loc else "#5C7CE0"
            ax.bar(i, med, width=0.5, color=c, alpha=0.8)
            ax.errorbar(i, med, yerr=sem, fmt="none", color="k", capsize=4, lw=1.2)

        ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="α=1 (Brownian)")
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", alpha=0.4)

        if (base / WT_ROI_ALPHA_STATS).exists():
            st = pd.read_csv(base / WT_ROI_ALPHA_STATS)
            mw = st[st["Test"].str.contains("Mann", case=False)]
            if len(mw):
                p = float(mw["P_value"].values[0])
                cd_row = st[st["Test"].str.contains("Cliff", case=False)]
                cd = float(cd_row["Statistic"].values[0]) if len(cd_row) else np.nan
                ymax = max(meds) * 1.12
                ax.annotate("", xy=(1, ymax), xytext=(0, ymax),
                            arrowprops=dict(arrowstyle="-", lw=1))
                label = f"p={p:.1e}"
                if not np.isnan(cd):
                    label += f"\nCliff's δ={cd:.3f}"
                ax.text(0.5, ymax * 1.005, label, ha="center", fontsize=8)
                ax.set_ylim(0, ymax * 1.18)

        # stacked fraction bars (inset-style text annotation)
        for i, (n, ns_, nn, nsu) in enumerate(zip(ns, n_sub, n_nor, n_sup)):
            pct_sub = 100 * ns_ / n
            pct_nor = 100 * nn  / n
            pct_sup = 100 * nsu / n
            ax.text(i, 0.04,
                    f"sub {pct_sub:.0f}%\nnorm {pct_nor:.0f}%\nsup {pct_sup:.0f}%",
                    ha="center", fontsize=6.5, va="bottom", color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="grey", alpha=0.4))

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Inside\n(n={ns[0]})", f"Outside\n(n={ns[1]})"])
        ax.set_ylabel("Median anomalous exponent α")
        ax.set_title("Anomalous diffusion exponent α\n(0.5=subdiffusion, 1=Brownian)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_out = out_dir / "roi_inside_outside.png"
    plt.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROI figure saved: {fig_out}")

    # Print summary
    if wt_diff_path.exists():
        df = pd.read_csv(wt_diff_path)
        print("\n── WT Inside/Outside ER-ROI — D ──────────────────────────")
        print(df[["Location", "N", "Median", "Mean", "SEM", "Q25", "Q75"]].to_string(index=False))
    if wt_alpha_path.exists():
        df = pd.read_csv(wt_alpha_path)
        print("\n── WT Inside/Outside ER-ROI — α ──────────────────────────")
        print(df[["Location", "N_trajectories", "Median", "Mean",
                  "N_subdiffusion", "N_normal", "N_superdiffusion"]].to_string(index=False))


if __name__ == "__main__":
    main()

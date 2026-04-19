#!/usr/bin/env python3
"""
run_pipeline.py  —  Mosaic Traj_*.csv → diffusion analysis in one command.

Replaces the manual 1→2→3→6 multi-script workflow.  Applies a per-track
maximum-length filter to reduce the density-based over-linking artefact before
computing D.

Usage
-----
    python run_pipeline.py <folder>  [options]

    # Default (max_track_length=20):
    python run_pipeline.py "Analysed data/Gastrulation Diffusion"

    # Disable the length cap to reproduce unfiltered results:
    python run_pipeline.py "Analysed data/Gastrulation Diffusion" --max_track_length 0

    # Stricter filter for very dense data:
    python run_pipeline.py "Analysed data/Cellularization data" --max_track_length 15

Options
-------
  --min_track_length INT    Exclude tracks shorter than this (default: 10)
  --max_track_length INT    Exclude tracks longer than this; 0 = no limit (default: 20)
  --dt FLOAT                Frame interval in seconds (default: 0.1)
  --pixel FLOAT             Pixel size in µm (default: 0.094367)
  --max_fit_points INT      Maximum lag points for MSD linear fit (default: 11)
  --fit_fraction FLOAT      Max fraction of track used for MSD fit (default: 0.8)
  --recursive               Also search sub-folders for Traj_*.csv files
  --output_dir PATH         Where to save results (default: <folder>/pipeline_output)
"""

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── defaults ──────────────────────────────────────────────────────────────────
_DT           = 0.1
_PIXEL        = 0.094367
_MIN_LEN      = 10
_MAX_LEN      = 20
_MAX_FIT_PTS  = 11
_FIT_FRACTION = 0.8
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Mosaic Traj CSV → per-embryo diffusion summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", help="Folder containing Traj_*.csv files")
    p.add_argument("--min_track_length", type=int, default=_MIN_LEN)
    p.add_argument("--max_track_length", type=int, default=_MAX_LEN,
                   help="0 = disabled (no upper limit)")
    p.add_argument("--dt",            type=float, default=_DT)
    p.add_argument("--pixel",         type=float, default=_PIXEL)
    p.add_argument("--max_fit_points",type=int,   default=_MAX_FIT_PTS)
    p.add_argument("--fit_fraction",  type=float, default=_FIT_FRACTION)
    p.add_argument("--recursive",     action="store_true",
                   help="Recurse into sub-folders")
    p.add_argument("--output_dir",    default=None)
    return p.parse_args()


# ── I/O ───────────────────────────────────────────────────────────────────────

def find_csv_files(folder, recursive=False):
    patterns = ["Traj_*.csv", "tracked_Traj_*.csv"]
    files = []
    for pat in patterns:
        if recursive:
            files += glob.glob(os.path.join(folder, "**", pat), recursive=True)
        else:
            files += glob.glob(os.path.join(folder, pat))
    return sorted(set(files))


def load_traj_csv(path):
    """Load a Mosaic Traj CSV.  Returns DataFrame with Trajectory/Frame/x/y or None."""
    required = {"Trajectory", "Frame", "x", "y"}
    for sep in [",", "\t", " "]:
        try:
            df = pd.read_csv(path, sep=sep, index_col=0)
            if required.issubset(df.columns):
                return df[list(required)]
        except Exception:
            pass
    # Fallback: auto-detect separator
    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if "traj" in lc or "track" in lc:
                rename[c] = "Trajectory"
            elif "frame" in lc:
                rename[c] = "Frame"
            elif lc in ("x", "x_pos", "posx", "pos_x"):
                rename[c] = "x"
            elif lc in ("y", "y_pos", "posy", "pos_y"):
                rename[c] = "y"
        df = df.rename(columns=rename)
        if required.issubset(df.columns):
            return df[list(required)]
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
    return None


def load_particle_density(folder):
    """Read particle_counts_summary.csv if present.  Returns dict filename→mean_density."""
    csv = os.path.join(folder, "particle_counts_summary.csv")
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_csv(csv)
        # Column names vary; try common variants
        name_col  = next((c for c in df.columns if "file" in c.lower()), None)
        dens_col  = next((c for c in df.columns if "mean" in c.lower() and "particle" in c.lower()), None)
        if name_col and dens_col:
            return dict(zip(df[name_col].str.lower(), df[dens_col]))
    except Exception:
        pass
    return {}


# ── MSD + fitting ─────────────────────────────────────────────────────────────

def compute_msd(x, y, max_points, fit_fraction):
    """Overlapping-lag MSD for one track.  Returns array of length max_lag."""
    n = len(x)
    max_lag = max(2, min(max_points, int(n * fit_fraction)))
    msd = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[lag - 1] = (dx**2 + dy**2).mean()
    return msd


def fit_d(msd, dt):
    """
    Fit MSD = 4Dt + C.
    Returns (D, offset, r_squared).  All NaN if fit fails.
    """
    n = len(msd)
    if n < 2:
        return np.nan, np.nan, np.nan
    t = np.arange(1, n + 1) * dt
    valid = ~np.isnan(msd)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan
    try:
        popt, _ = curve_fit(lambda t, D, c: 4 * D * t + c,
                            t[valid], msd[valid], p0=[0.05, 0.0],
                            maxfev=1000)
        D, c = popt
        pred  = 4 * D * t[valid] + c
        ss_res = ((msd[valid] - pred) ** 2).sum()
        ss_tot = ((msd[valid] - msd[valid].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(D), float(c), float(r2)
    except Exception:
        return np.nan, np.nan, np.nan


# ── per-file analysis ─────────────────────────────────────────────────────────

def analyse_file(path, args):
    """
    Analyse one Traj CSV.

    Returns
    -------
    summary  : dict  (per-embryo statistics)
    track_df : DataFrame  (one row per analysed track)
    name     : str
    """
    df = load_traj_csv(path)
    if df is None:
        return None

    name = Path(path).stem

    # Convert pixels → µm
    x_all = df["x"].values * args.pixel
    y_all = df["y"].values * args.pixel
    traj_ids  = df["Trajectory"].values

    n_total = 0
    n_short = 0
    n_long  = 0
    records = []
    long_frac_lengths = []   # lengths of ALL tracks including excluded ones

    for tid, grp in df.groupby("Trajectory"):
        grp = grp.sort_values("Frame")
        length = len(grp)
        n_total += 1
        long_frac_lengths.append(length)

        if length < args.min_track_length:
            n_short += 1
            continue

        if args.max_track_length > 0 and length > args.max_track_length:
            n_long += 1
            continue

        x = grp["x"].values * args.pixel
        y = grp["y"].values * args.pixel
        msd = compute_msd(x, y, args.max_fit_points, args.fit_fraction)
        D, offset, r2 = fit_d(msd, args.dt)

        records.append({
            "track_id": tid,
            "length":   length,
            "D":        D,
            "offset":   offset,
            "r_squared": r2,
        })

    track_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["track_id", "length", "D", "offset", "r_squared"]
    )
    valid = track_df.dropna(subset=["D"])

    # Fraction of ALL tracks that are long (including those below min_len)
    n_over20 = sum(1 for l in long_frac_lengths if l > 20)
    frac_over20 = n_over20 / n_total if n_total > 0 else np.nan

    summary = {
        "FileName":          name,
        "N_total_tracks":    n_total,
        "N_short_excl":      n_short,
        "N_long_excl":       n_long,
        "frac_over20fr":     round(frac_over20, 4),
        "N_analysed":        len(valid),
        "MeanTrackLength":   float(np.mean(long_frac_lengths)) if long_frac_lengths else np.nan,
        "MedianD":           float(np.median(valid["D"])) if len(valid) > 0 else np.nan,
        "MeanD":             float(np.mean(valid["D"]))   if len(valid) > 0 else np.nan,
        "StdD":              float(np.std(valid["D"]))    if len(valid) > 0 else np.nan,
        "SEMD":              float(valid["D"].sem())       if len(valid) > 0 else np.nan,
        "MinD":              float(valid["D"].min())       if len(valid) > 0 else np.nan,
        "MaxD":              float(valid["D"].max())       if len(valid) > 0 else np.nan,
    }

    # Quality flag
    mtl = summary["MeanTrackLength"]
    frac = frac_over20
    if summary["N_analysed"] == 0:
        flag = "FAIL_no_tracks"
    elif mtl > 20:
        flag = "WARN_high_mean_len"
    elif frac > 0.25:
        flag = "WARN_high_long_frac"
    elif frac > 0.18:
        flag = "NOTE_elevated_long_frac"
    else:
        flag = "PASS"
    summary["QualityFlag"] = flag

    return summary, track_df, name


# ── plotting ──────────────────────────────────────────────────────────────────

def _short_name(name, max_len=25):
    """Shorten embryo name for axis labels."""
    import re
    m = re.search(r'Traj_(.+?)(?:\.nd2|$)', name)
    if m:
        s = m.group(1)
        return s if len(s) <= max_len else s[:max_len - 2] + ".."
    return name if len(name) <= max_len else name[:max_len - 2] + ".."


def make_plots(results, output_dir, args):
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    labels = [_short_name(r[2]) for r in results]

    # ── 1. Per-track D jitter + median ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.0), 5))
    for i, ((summary, tdf, name), c) in enumerate(zip(results, colors)):
        Dvals = tdf.dropna(subset=["D"])["D"].values
        if len(Dvals) == 0:
            continue
        ax.scatter(np.random.normal(i, 0.07, len(Dvals)), Dvals,
                   s=2, alpha=0.25, color=c, rasterized=True)
        ax.scatter(i, summary["MedianD"], s=70, marker="D",
                   color="black", zorder=5,
                   label="median" if i == 0 else "")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("D (µm²/s)")
    cap = f"max_len={args.max_track_length}" if args.max_track_length > 0 else "no length cap"
    ax.set_title(f"Per-track D   [{cap}]")
    ax.legend(markerscale=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "D_jitter.png"), dpi=200)
    plt.close(fig)

    # ── 2. Median D bar chart ─────────────────────────────────────────────────
    medians = [r[0]["MedianD"] for r in results]
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.0), 4))
    ax.bar(range(len(results)), medians, color=colors)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median D (µm²/s)")
    ax.set_title("Median D per embryo")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "median_D_bar.png"), dpi=200)
    plt.close(fig)

    # ── 3. Track length distributions (all tracks, pre-filter) ───────────────
    # We have the filtered track_df; recompute from the raw lengths stored in
    # summary dict isn't possible — plot filtered lengths only and note it.
    fig, ax = plt.subplots(figsize=(8, 4))
    for (summary, tdf, name), c in zip(results, colors):
        lengths = tdf["length"].values
        if len(lengths) == 0:
            continue
        max_l = max(lengths)
        bins  = np.arange(0, min(max_l + 5, 105), 5)
        counts, edges = np.histogram(lengths, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, "-", color=c, label=_short_name(name), linewidth=1.2)
    ax.axvline(args.max_track_length, color="red", linestyle="--", linewidth=1,
               label=f"max_len={args.max_track_length}")
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Track length distributions (analysed tracks)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "track_lengths.png"), dpi=200)
    plt.close(fig)

    # ── 4. CDF of D ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for (summary, tdf, name), c in zip(results, colors):
        Dvals = np.sort(tdf.dropna(subset=["D"])["D"].values)
        if len(Dvals) == 0:
            continue
        y = np.arange(1, len(Dvals) + 1) / len(Dvals)
        ax.plot(Dvals, y, "-", color=c, label=_short_name(name), linewidth=1.2)
    ax.set_xlabel("D (µm²/s)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of D per embryo")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "CDF_D.png"), dpi=200)
    plt.close(fig)

    # ── 5. Quality overview table ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.2), 3.5))
    ax.axis("off")
    col_headers = ["Embryo", "N_total", "N_excl_long", "frac>20fr",
                   "N_analysed", "MeanTrackLen", "MedianD", "Flag"]
    table_data = []
    for summary, _, _ in results:
        table_data.append([
            _short_name(summary["FileName"]),
            summary["N_total_tracks"],
            summary["N_long_excl"],
            f"{summary['frac_over20fr']*100:.1f}%",
            summary["N_analysed"],
            f"{summary['MeanTrackLength']:.1f}",
            f"{summary['MedianD']:.4f}" if not np.isnan(summary["MedianD"]) else "—",
            summary["QualityFlag"],
        ])
    tbl = ax.table(cellText=table_data, colLabels=col_headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    # Colour-code the Flag column
    flag_col_idx = col_headers.index("Flag")
    for row_idx, (summary, _, _) in enumerate(results, start=1):
        flag = summary["QualityFlag"]
        cell = tbl[row_idx, flag_col_idx]
        if "FAIL" in flag:
            cell.set_facecolor("#ffcccc")
        elif "WARN" in flag:
            cell.set_facecolor("#ffe5b4")
        elif "NOTE" in flag:
            cell.set_facecolor("#ffffcc")
        else:
            cell.set_facecolor("#ccffcc")
    ax.set_title(f"Quality summary  [max_track_len={args.max_track_length}]",
                 fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quality_table.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    folder = args.folder

    if not os.path.isdir(folder):
        sys.exit(f"Folder not found: {folder}")

    csv_files = find_csv_files(folder, recursive=args.recursive)
    if not csv_files:
        sys.exit(f"No Traj_*.csv files found in {folder}"
                 + (" (try --recursive)" if not args.recursive else ""))

    # Output directory
    out = args.output_dir or os.path.join(folder, "pipeline_output")
    os.makedirs(out, exist_ok=True)

    # Optional: particle density lookup
    density_map = load_particle_density(folder)

    print(f"\n{'='*60}")
    print(f"GEM diffusion pipeline")
    print(f"  Input:            {folder}")
    print(f"  Files found:      {len(csv_files)}")
    print(f"  Track length:     [{args.min_track_length}, "
          + (f"{args.max_track_length}]" if args.max_track_length > 0 else "∞]"))
    print(f"  MSD fit:          first {args.max_fit_points} lags or ≤{args.fit_fraction*100:.0f}% of track")
    print(f"  dt={args.dt} s/fr   pixel={args.pixel} µm/px")
    print(f"  Output:           {out}")
    print(f"{'='*60}\n")

    results   = []
    summaries = []

    for path in csv_files:
        print(f"→ {os.path.basename(path)}")
        res = analyse_file(path, args)
        if res is None:
            print("  SKIP — could not load file\n")
            continue
        summary, track_df, name = res

        # Attach density if available
        density_key = Path(path).stem.lower().replace("traj_", "").replace(".nd2_crop", "")
        summary["MeanParticlesPerFrame"] = density_map.get(density_key, np.nan)

        results.append((summary, track_df, name))
        summaries.append(summary)

        flag_sym = {"PASS": "✓", "FAIL_no_tracks": "✗"}.get(
            summary["QualityFlag"], "⚠")
        print(f"  N_total={summary['N_total_tracks']}  "
              f"excl_long={summary['N_long_excl']} ({summary['frac_over20fr']*100:.1f}% >20fr)  "
              f"analysed={summary['N_analysed']}")
        print(f"  MeanTrackLen={summary['MeanTrackLength']:.1f}  "
              f"MedianD={summary['MedianD']:.4f}  "
              f"MeanD={summary['MeanD']:.4f} µm²/s  "
              f"[{flag_sym} {summary['QualityFlag']}]\n")

    if not results:
        sys.exit("No files successfully analysed.")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summaries)
    csv_out = os.path.join(out, "diffusion_summary.csv")
    summary_df.to_csv(csv_out, index=False)

    # ── Save per-track D CSVs ─────────────────────────────────────────────────
    all_tracks = []
    for summary, tdf, name in results:
        tdf = tdf.copy()
        tdf["embryo"] = name
        all_tracks.append(tdf)
    if all_tracks:
        pd.concat(all_tracks, ignore_index=True).to_csv(
            os.path.join(out, "all_tracks.csv"), index=False)

    # ── Group-level stats ─────────────────────────────────────────────────────
    all_D = [d for _, tdf, _ in results
             for d in tdf.dropna(subset=["D"])["D"].values]
    group_median = np.median(all_D) if all_D else np.nan
    group_mean   = np.mean(all_D)   if all_D else np.nan

    print(f"{'='*60}")
    print(f"Group summary ({len(results)} embryos, N={len(all_D)} tracks)")
    print(f"  Group median D = {group_median:.4f} µm²/s")
    print(f"  Group mean D   = {group_mean:.4f} µm²/s")
    print(f"  Summary CSV    → {csv_out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("  Generating plots ...")
    make_plots(results, out, args)
    print(f"  Plots           → {out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_trackmate_mosaic.py
Cross-validate Mosaic vs TrackMate diffusion coefficients on the same embryos.

For each embryo where both Traj_<name>.csv (Mosaic) and
Traj_<name>_trackmate.csv (TrackMate, from parse_trackmate_csv.py) exist,
computes per-track D from MSD linear fit and reports per-embryo median D.

Usage:
    python compare_trackmate_mosaic.py <folder>

Output:
    <folder>/trackmate_mosaic_comparison.csv   — per-embryo median D table
    <folder>/trackmate_mosaic_comparison.png   — scatter + violin plots
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PIXEL_UM   = 0.094367  # µm/pixel — same as Script 1
DT         = 0.1       # s per frame
MIN_LEN    = 10        # minimum track length (frames)
MAX_LAG    = 11        # max MSD lag points used for fit


# ── trajectory loading ─────────────────────────────────────────────────────

def _normalise_cols(df):
    """Return df with columns renamed to Trajectory, Frame, x, y."""
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "traj" in cl or "track" in cl:
            col_map[c] = "Trajectory"
        elif "frame" in cl or "time" in cl:
            col_map[c] = "Frame"
        elif cl in ("x", "pos_x", "position_x"):
            col_map[c] = "x"
        elif cl in ("y", "pos_y", "position_y"):
            col_map[c] = "y"
    return df.rename(columns=col_map)[["Trajectory", "Frame", "x", "y"]]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalise_cols(df).dropna()
    df["x_um"] = df["x"].astype(float) * PIXEL_UM
    df["y_um"] = df["y"].astype(float) * PIXEL_UM
    return df


# ── D computation ──────────────────────────────────────────────────────────

def _D_from_track(x: np.ndarray, y: np.ndarray) -> float:
    """Fit MSD(τ) = 4Dτ + C; return D (µm²/s) or NaN."""
    n = len(x)
    max_lag = min(int(n * 0.25), MAX_LAG)
    if max_lag < 3:
        return np.nan

    msds = np.array([
        np.mean((x[lag:] - x[:-lag])**2 + (y[lag:] - y[:-lag])**2)
        for lag in range(1, max_lag + 1)
    ])
    lags = np.arange(1, max_lag + 1) * DT
    slope = np.polyfit(lags, msds, 1)[0]
    D = slope / 4.0
    return D if D > 0 else np.nan


def compute_D_values(csv_path: Path) -> np.ndarray:
    df = load_csv(csv_path)
    D_list = []
    for _, grp in df.groupby("Trajectory"):
        grp = grp.sort_values("Frame")
        if len(grp) < MIN_LEN:
            continue
        D = _D_from_track(grp["x_um"].values, grp["y_um"].values)
        if not np.isnan(D):
            D_list.append(D)
    return np.array(D_list)


# ── file pairing ───────────────────────────────────────────────────────────

def find_pairs(folder: Path):
    """Yield (mosaic_csv, trackmate_csv, label) for matched embryos."""
    mosaic_files = [
        f for f in sorted(folder.glob("Traj_*.csv"))
        if "_trackmate" not in f.name
    ]
    for mf in mosaic_files:
        tf = folder / f"{mf.stem}_trackmate.csv"
        if tf.exists():
            label = mf.stem.removeprefix("Traj_")
            yield mf, tf, label


# ── main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python compare_trackmate_mosaic.py <folder>")

    folder = Path(sys.argv[1])
    pairs  = list(find_pairs(folder))
    if not pairs:
        sys.exit("No matched Mosaic/TrackMate CSV pairs found.\n"
                 "Run parse_trackmate_csv.py first to generate Traj_*_trackmate.csv files.")

    print(f"Found {len(pairs)} matched embryo(s).\n")
    header = f"{'Embryo':<32} {'Mosaic med D':>13} {'TM med D':>10} "    \
             f"{'N_mosaic':>9} {'N_tm':>6}"
    print(header)
    print("-" * len(header))

    records   = []
    all_D_m   = []
    all_D_tm  = []

    for mf, tf, label in pairs:
        D_m  = compute_D_values(mf)
        D_tm = compute_D_values(tf)
        med_m  = float(np.median(D_m))  if len(D_m)  else np.nan
        med_tm = float(np.median(D_tm)) if len(D_tm) else np.nan

        print(f"{label:<32} {med_m:>13.4f} {med_tm:>10.4f} "
              f"{len(D_m):>9} {len(D_tm):>6}")

        records.append({
            "embryo":      label,
            "D_mosaic":    med_m,
            "D_trackmate": med_tm,
            "n_mosaic":    len(D_m),
            "n_trackmate": len(D_tm),
        })
        all_D_m.extend(D_m)
        all_D_tm.extend(D_tm)

    all_D_m  = np.array(all_D_m)
    all_D_tm = np.array(all_D_tm)

    # ── save CSV ──
    summary_path = folder / "trackmate_mosaic_comparison.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)
    print(f"\nSummary saved : {summary_path}")

    # ── plots ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Scatter: per-embryo median D
    ax = axes[0]
    D_m_vals  = [r["D_mosaic"]    for r in records]
    D_tm_vals = [r["D_trackmate"] for r in records]
    ax.scatter(D_m_vals, D_tm_vals, s=70, zorder=3)
    for r in records:
        ax.annotate(r["embryo"], (r["D_mosaic"], r["D_trackmate"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    upper = max(max(D_m_vals), max(D_tm_vals)) * 1.15
    ax.plot([0, upper], [0, upper], "k--", lw=0.8, label="y = x")
    ax.set_xlabel("Mosaic median D (µm²/s)")
    ax.set_ylabel("TrackMate median D (µm²/s)")
    ax.set_title("Per-embryo median D: Mosaic vs TrackMate")
    ax.legend(fontsize=8)

    # Violin: pooled distributions
    ax = axes[1]
    parts = ax.violinplot(
        [all_D_m, all_D_tm], positions=[1, 2], showmedians=True, showextrema=False
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Mosaic", "TrackMate"])
    ax.set_ylabel("D (µm²/s)")
    ax.set_title("Pooled D distribution")
    for pos, vals, label in ((1, all_D_m, "Mosaic"), (2, all_D_tm, "TrackMate")):
        med = np.median(vals)
        ax.text(pos, med * 1.06, f"med={med:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig_path = folder / "trackmate_mosaic_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved    : {fig_path}")


if __name__ == "__main__":
    main()

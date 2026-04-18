#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7Noise_calculation_folder.py

Batch version of 7Noise_calculation.py.
Runs the partition-based noise analysis across all pkl trajectory files in a
folder, producing per-file CSVs and a combined cross-embryo summary.

Input:
- Folder containing tracked_*.pkl or analyzed_*.pkl files

Output:
- <output_dir>/<embryo>/partition_statistics.csv  (per-file partition stats)
- <output_dir>/noise_summary.csv                  (one row per embryo)
- <output_dir>/noise_comparison.png               (mean D vs partition size)
- <output_dir>/noise_floor_estimate.csv           (noise floor per embryo)

Usage:
python 7Noise_calculation_folder.py
"""

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# ── Parameters ────────────────────────────────────────────────────────────────
PARTITION_SIZES = [5, 10, 15, 20]
MIN_TRACK_LENGTH = 5
DT = 0.1          # time step in seconds – change if needed
MSD_FIT_POINTS = 4
# ──────────────────────────────────────────────────────────────────────────────


def embryo_label(file_path):
    """Extract short embryo name from file path, e.g. 'Em6' from tracked_Traj_Em6.nd2_crop.pkl"""
    name = os.path.splitext(os.path.basename(file_path))[0]
    m = re.search(r'Traj_(.+?)\.nd2', name)
    return m.group(1) if m else name


def load_trajectories(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  Could not load {file_path}: {e}")
        return None
    if 'trajectories' in data:
        return data['trajectories']
    print(f"  No 'trajectories' key in {file_path}")
    return None


def _linear_msd(t, D, offset):
    return 4 * D * t + offset


def _calc_msd(x, y, dt):
    n = len(x)
    msd = np.zeros(n - 1)
    for lag in range(1, n):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[lag - 1] = np.mean(dx**2 + dy**2)
    return msd, np.arange(1, n) * dt


def _partition_D(trajectories, partition_size, dt):
    """Return list of D values from all partitions of the given size."""
    D_values = []
    for traj in trajectories:
        x = np.asarray(traj['x'], dtype=float)
        y = np.asarray(traj['y'], dtype=float)
        n = len(x)
        if n < partition_size:
            continue
        for p in range(n // partition_size):
            xi = x[p * partition_size:(p + 1) * partition_size]
            yi = y[p * partition_size:(p + 1) * partition_size]
            msd, tlag = _calc_msd(xi, yi, dt)
            pts = min(MSD_FIT_POINTS, len(msd))
            if pts < 2:
                continue
            try:
                popt, _ = curve_fit(_linear_msd, tlag[:pts], msd[:pts],
                                    p0=[0.01, 0.0], maxfev=1000)
                D = popt[0] / 4  # match convention in 7Noise_calculation.py
                D_values.append(D)
            except Exception:
                pass
    return D_values


def analyse_file(file_path, partition_sizes, dt):
    """
    Run partition analysis for one file.
    Returns dict: {partition_size -> {mean, median, std, cv, n}}
    and overall_cv across partition means.
    """
    trajectories = load_trajectories(file_path)
    if trajectories is None:
        return None, None

    rows = []
    means = []
    for ps in partition_sizes:
        Ds = _partition_D(trajectories, ps, dt)
        if len(Ds) < 3:
            rows.append({'PartitionSize': ps, 'N': 0,
                         'MeanD': np.nan, 'MedianD': np.nan,
                         'StdD': np.nan, 'CV': np.nan})
            means.append(np.nan)
            continue
        arr = np.array(Ds)
        mean_d = np.mean(arr)
        rows.append({
            'PartitionSize': ps,
            'N': len(arr),
            'MeanD': mean_d,
            'MedianD': np.median(arr),
            'StdD': np.std(arr),
            'CV': np.std(arr) / mean_d if mean_d > 0 else np.nan,
        })
        means.append(mean_d)

    valid_means = [m for m in means if not np.isnan(m)]
    overall_cv = (np.std(valid_means) / np.mean(valid_means)
                  if len(valid_means) > 1 and np.mean(valid_means) > 0 else np.nan)

    return pd.DataFrame(rows), overall_cv


def make_comparison_plot(summary_rows, partition_sizes, output_path):
    """Plot mean D vs partition size for every embryo on one axes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_rows)))

    for (row, color) in zip(summary_rows, colors):
        label = row['embryo']
        means = [row.get(f'MeanD_part{ps}', np.nan) for ps in partition_sizes]
        ax.plot(partition_sizes, means, 'o-', color=color, label=label,
                linewidth=2, markersize=6)

    ax.set_xlabel('Partition size (frames)', fontsize=12)
    ax.set_ylabel('Mean D from partitions (µm²/s)', fontsize=12)
    ax.set_title('Noise characterisation: mean partition D vs window size', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Comparison plot saved: {output_path}")


def main():
    print("\nBatch Noise Calculation")
    print("=" * 50)

    folder = input("Enter folder containing trajectory pkl files: ").strip()
    if not os.path.isdir(folder):
        print(f"Not a valid directory: {folder}")
        return

    dt_input = input(f"Time step in seconds (default {DT}): ").strip()
    dt = float(dt_input) if dt_input else DT

    # Find pkl files
    patterns = ['tracked_*.pkl', 'analyzed_*.pkl']
    pkl_files = []
    for pat in patterns:
        pkl_files.extend(glob.glob(os.path.join(folder, pat)))
    pkl_files = sorted(set(pkl_files))

    if not pkl_files:
        print("No tracked_*.pkl or analyzed_*.pkl files found.")
        return
    print(f"Found {len(pkl_files)} files\n")

    output_dir = os.path.join(folder, 'noise_analysis_folder')
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    noise_floor_rows = []

    for fp in pkl_files:
        label = embryo_label(fp)
        print(f"Processing {label} ...")
        part_df, overall_cv = analyse_file(fp, PARTITION_SIZES, dt)

        if part_df is None:
            print(f"  Skipped.\n")
            continue

        # Save per-file CSV
        embryo_dir = os.path.join(output_dir, label)
        os.makedirs(embryo_dir, exist_ok=True)
        part_df.to_csv(os.path.join(embryo_dir, 'partition_statistics.csv'), index=False)

        print(f"  Overall CV: {overall_cv:.4f}" if not np.isnan(overall_cv) else "  Overall CV: n/a")
        for _, r in part_df.iterrows():
            print(f"  Partition {int(r['PartitionSize']):2d} frames: "
                  f"n={int(r['N'])}  mean D={r['MeanD']:.5f}  CV={r['CV']:.3f}")

        # Build summary row (one per embryo)
        row = {'embryo': label, 'overall_cv': overall_cv}
        for _, r in part_df.iterrows():
            ps = int(r['PartitionSize'])
            row[f'MeanD_part{ps}'] = r['MeanD']
            row[f'CV_part{ps}'] = r['CV']
            row[f'N_part{ps}'] = r['N']
        summary_rows.append(row)

        # Noise floor estimate = mean D at smallest partition size
        smallest_ps = PARTITION_SIZES[0]
        noise_D = part_df.loc[part_df['PartitionSize'] == smallest_ps, 'MeanD'].values
        noise_floor_rows.append({
            'embryo': label,
            'noise_floor_D_um2s': float(noise_D[0]) if len(noise_D) else np.nan,
            'partition_size_used': smallest_ps,
            'note': f'Mean D from {smallest_ps}-frame partitions (upper-bound noise estimate)'
        })
        print()

    if not summary_rows:
        print("No results to save.")
        return

    # Save combined CSVs
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'noise_summary.csv'), index=False)

    noise_df = pd.DataFrame(noise_floor_rows)
    noise_df.to_csv(os.path.join(output_dir, 'noise_floor_estimate.csv'), index=False)

    # Comparison plot
    make_comparison_plot(summary_rows, PARTITION_SIZES,
                         os.path.join(output_dir, 'noise_comparison.png'))

    # Print noise floor table
    print("\n" + "=" * 50)
    print("NOISE FLOOR ESTIMATES (mean D at smallest partition)")
    print("=" * 50)
    for _, r in noise_df.iterrows():
        print(f"  {r['embryo']:12s}  noise floor ≈ {r['noise_floor_D_um2s']:.5f} µm²/s")

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

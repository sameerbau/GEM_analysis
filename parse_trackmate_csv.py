#!/usr/bin/env python3
"""
parse_trackmate_csv.py
Convert a TrackMate spots CSV export into the existing GEM pipeline format.

TrackMate export steps (GUI):
  Analysis panel → bottom toolbar → "Spots" table icon → Export to CSV
  The spots table contains every detected spot with its track assignment.

This script produces two files next to the input CSV:
  Traj_<stem>_trackmate.csv  — Mosaic-compatible trajectory CSV
                                (pixel coords, same format as Traj_*.csv)
                                Drop straight into scripts 1-7 unchanged.
  spots_per_frame_<stem>.csv — All detected spots per frame (density QC)
                                Equivalent to macro 3 output.

Usage:
    python parse_trackmate_csv.py <spots_csv>
    python parse_trackmate_csv.py <folder>   # all *_spots.csv in folder

Expected TrackMate spots CSV columns (case-insensitive):
    TRACK_ID    — track assignment; blank/NaN = untracked spot
    FRAME       — frame index (0-based)
    POSITION_X  — x position in µm
    POSITION_Y  — y position in µm
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIXEL_UM = 0.094367   # µm/pixel — must match Script 1 CONVERSION value


def _find_col(df, *candidates):
    """Return the first column name (case-insensitive) that matches a candidate."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def parse_spots_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    col_track = _find_col(df, "TRACK_ID", "track_id", "TrackID")
    col_frame = _find_col(df, "FRAME", "frame", "Frame")
    col_x     = _find_col(df, "POSITION_X", "position_x", "X", "x")
    col_y     = _find_col(df, "POSITION_Y", "position_y", "Y", "y")

    df = df.rename(columns={
        col_track: "TRACK_ID",
        col_frame: "FRAME",
        col_x:     "POSITION_X",
        col_y:     "POSITION_Y",
    })

    df["FRAME"]      = pd.to_numeric(df["FRAME"],      errors="coerce")
    df["POSITION_X"] = pd.to_numeric(df["POSITION_X"], errors="coerce")
    df["POSITION_Y"] = pd.to_numeric(df["POSITION_Y"], errors="coerce")
    df["TRACK_ID"]   = pd.to_numeric(df["TRACK_ID"],   errors="coerce")

    # ── Trajectory CSV (tracked spots only) ──────────────────────────────
    tracked = df.dropna(subset=["TRACK_ID"]).copy()
    tracked["Trajectory"] = tracked["TRACK_ID"].astype(int)
    tracked["Frame"]      = tracked["FRAME"].astype(int)
    # TrackMate positions are in µm; convert to pixels to match Mosaic CSV
    # (Script 1 will multiply back by CONVERSION = 0.094367 µm/px)
    tracked["x"] = tracked["POSITION_X"] / PIXEL_UM
    tracked["y"] = tracked["POSITION_Y"] / PIXEL_UM

    traj_df = (
        tracked[["Trajectory", "Frame", "x", "y"]]
        .sort_values(["Trajectory", "Frame"])
        .reset_index(drop=True)
    )

    # ── Spots-per-frame CSV (all spots, tracked or not) ──────────────────
    all_spots = df.dropna(subset=["FRAME"]).copy()
    all_spots["Frame"] = all_spots["FRAME"].astype(int)
    spf = (
        all_spots.groupby("Frame")
        .size()
        .reset_index(name="ParticleCount")
        .assign(FileName=csv_path.name)
        [["FileName", "Frame", "ParticleCount"]]
    )

    return traj_df, spf


def process(csv_path: Path):
    print(f"Parsing: {csv_path.name}")
    traj_df, spf = parse_spots_csv(csv_path)

    out_dir = csv_path.parent
    stem    = csv_path.stem  # e.g. "Em001_crop_spots"

    # Strip common suffixes so output name matches the embryo
    clean_stem = stem.removesuffix("_spots").removesuffix("_Spots")

    traj_out    = out_dir / f"Traj_{clean_stem}_trackmate.csv"
    density_out = out_dir / f"spots_per_frame_{clean_stem}.csv"

    traj_df.to_csv(traj_out, index=False)
    spf.to_csv(density_out, index=False)

    n_tracks = traj_df["Trajectory"].nunique()
    mean_spf   = spf["ParticleCount"].mean()
    median_spf = spf["ParticleCount"].median()

    print(f"  Tracks CSV : {traj_out.name}  ({n_tracks} tracks)")
    print(f"  Density CSV: {density_out.name}  "
          f"(mean={mean_spf:.1f}, median={median_spf:.1f} spots/frame)")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python parse_trackmate_csv.py <spots_csv_or_folder>")

    p = Path(sys.argv[1])
    if p.is_dir():
        files = sorted(p.glob("*_spots.csv")) or sorted(p.glob("*_Spots.csv"))
        if not files:
            sys.exit(f"No *_spots.csv files found in {p}")
        for f in files:
            process(f)
    elif p.is_file():
        process(p)
    else:
        sys.exit(f"Not found: {p}")

    print("\nDone. Feed Traj_*_trackmate.csv into scripts 1-7 unchanged.")


if __name__ == "__main__":
    main()

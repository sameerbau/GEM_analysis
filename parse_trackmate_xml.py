#!/usr/bin/env python3
"""
parse_trackmate_xml.py
Convert a TrackMate XML export into files that plug directly into the
existing GEM analysis pipeline (scripts 1-7).

Outputs (written next to the input XML):
  Traj_<stem>_trackmate.csv  — Mosaic-compatible trajectory CSV (pixel coords)
                                Drop-in for Traj_*.csv in scripts 1-7.
                                Use CONVERSION = 0.094367 in Script 1 as usual.
  spots_per_frame_<stem>.csv — Per-frame spot counts for density QC
                                (complements ImageJ macro 3)

Usage:
    python parse_trackmate_xml.py <xml_file>
    python parse_trackmate_xml.py <folder>   # processes all *.xml in folder

TrackMate export: File > Export > Export tracks to XML file (full XML).
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PIXEL_UM_DEFAULT = 0.094367   # µm/pixel — used if not found in XML


def read_pixel_size(root):
    """Read spatial calibration from TrackMate XML; fall back to default."""
    try:
        img = root.find("Settings/ImageData")
        return float(img.get("pixelwidth", PIXEL_UM_DEFAULT))
    except (AttributeError, TypeError):
        return PIXEL_UM_DEFAULT


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pixel_um = read_pixel_size(root)

    model = root.find("Model")

    # --- Spot lookup: ID → {frame, x_um, y_um, quality} ---
    spots = {}
    for frame_el in model.find("AllSpots").findall("SpotsInFrame"):
        frame_idx = int(frame_el.get("frame"))
        for spot in frame_el.findall("Spot"):
            sid = int(spot.get("ID"))
            spots[sid] = {
                "frame":   frame_idx,
                "x_um":    float(spot.get("POSITION_X")),
                "y_um":    float(spot.get("POSITION_Y")),
                "quality": float(spot.get("QUALITY", 0.0)),
            }

    # --- Filtered track IDs ---
    filtered_ids = {
        int(el.get("TRACK_ID"))
        for el in model.find("FilteredTracks").findall("TrackID")
    }

    # --- Build tracks from edges ---
    rows = []
    for track_el in model.find("AllTracks").findall("Track"):
        tid = int(track_el.get("TRACK_ID"))
        if tid not in filtered_ids:
            continue
        spot_ids = set()
        for edge in track_el.findall("Edge"):
            spot_ids.add(int(edge.get("SPOT_SOURCE_ID")))
            spot_ids.add(int(edge.get("SPOT_TARGET_ID")))
        for sid in spot_ids:
            s = spots[sid]
            rows.append({
                "Trajectory": tid,
                "Frame":      s["frame"],
                # Convert µm → pixels so Script 1 can apply its usual
                # CONVERSION = 0.094367 pixel→µm factor unchanged
                "x": s["x_um"] / pixel_um,
                "y": s["y_um"] / pixel_um,
            })

    tracks_df = (
        pd.DataFrame(rows)
        .sort_values(["Trajectory", "Frame"])
        .reset_index(drop=True)
    )

    # --- Spots per frame (all detected spots, tracked or not) ---
    spots_df = pd.DataFrame(spots.values())
    spf = (
        spots_df.groupby("frame")
        .size()
        .reset_index(name="ParticleCount")
        .rename(columns={"frame": "Frame"})
        .assign(FileName=xml_path.name)
        [["FileName", "Frame", "ParticleCount"]]
    )

    return tracks_df, spf, pixel_um


def process(xml_path: Path):
    print(f"Parsing: {xml_path.name}")
    tracks_df, spf, pixel_um = parse_xml(xml_path)

    out_dir = xml_path.parent
    stem = xml_path.stem

    # 1. Mosaic-compatible trajectory CSV
    traj_out = out_dir / f"Traj_{stem}_trackmate.csv"
    tracks_df.to_csv(traj_out, index=False)
    n_tracks = tracks_df["Trajectory"].nunique()
    print(f"  Tracks CSV : {traj_out.name}  ({n_tracks} tracks, "
          f"{len(tracks_df)} rows)")

    # 2. Spots-per-frame density CSV
    density_out = out_dir / f"spots_per_frame_{stem}.csv"
    spf.to_csv(density_out, index=False)
    mean_spf   = spf["ParticleCount"].mean()
    median_spf = spf["ParticleCount"].median()
    print(f"  Density CSV: {density_out.name}  "
          f"(mean={mean_spf:.1f}, median={median_spf:.1f} spots/frame)")

    src = "XML" if pixel_um != PIXEL_UM_DEFAULT else "default (not in XML)"
    print(f"  Pixel size : {pixel_um:.6f} µm/pixel [{src}]")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python parse_trackmate_xml.py <xml_file_or_folder>")

    p = Path(sys.argv[1])
    if p.is_dir():
        xml_files = sorted(p.glob("*.xml"))
        if not xml_files:
            sys.exit(f"No .xml files found in {p}")
        for f in xml_files:
            process(f)
    elif p.is_file():
        process(p)
    else:
        sys.exit(f"Path not found: {p}")

    print("\nDone. Feed Traj_*_trackmate.csv into scripts 1-7 unchanged.")


if __name__ == "__main__":
    main()

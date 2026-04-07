"""
Cross-Validate Mitotic Cell Detections Across Two Z-Slices
===========================================================

Purpose
-------
When detecting mitotic cells from a single z-slice, the classifier can
mistakenly flag non-mitotic cells.  This script exploits the fact that the
same movie is tracked at two adjacent z-slices (z7 and z8, ~1 µm apart,
~1 s apart in acquisition) and keeps only cells that are called mitotic in
BOTH datasets.

Workflow
--------
1. Load the two "detected mitotic" CSV files (one per z-slice).
2. For each frame, spatially match detections: if a cell in z-slice A has a
   counterpart in z-slice B within `spatial_tolerance` pixels, it is
   considered a genuine hit.
3. Save the consensus (double-confirmed) detections to a new CSV.
4. Optionally mark the consensus cells on a reference image and save an
   annotated PNG.

Expected CSV format (same as TrackMate / GEM-analysis pipeline output)
-----------------------------------------------------------------------
Columns (at minimum): Trajectory, Frame, x, y
Example:
    ,Trajectory,Frame,x,y,z,...
    1,5,0,121.6,179.4,0,...

Usage
-----
    python cross_validate_mitotic_z_slices.py \\
        --z8  set1002-1_detected_mitotic_tracks.csv \\
        --z7  Aset1002-1_detected_mitotic_tracks.csv \\
        --output  consensus_mitotic_tracks.csv \\
        --tolerance 20 \\
        --image   MAX_set1002.tif          # optional
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tracks(path: str | Path) -> pd.DataFrame:
    """Load a tracks CSV robustly, auto-detecting the index column."""
    df = pd.read_csv(path, index_col=0)

    # Normalise column names: strip whitespace, lower-case for lookup
    df.columns = [c.strip() for c in df.columns]

    required = {"Trajectory", "Frame", "x", "y"}
    # Try case-insensitive match if the exact names are missing
    col_map = {}
    for req in required:
        if req in df.columns:
            continue
        match = next((c for c in df.columns if c.lower() == req.lower()), None)
        if match:
            col_map[match] = req
        else:
            raise ValueError(
                f"Required column '{req}' not found in {path}.\n"
                f"Available columns: {list(df.columns)}"
            )
    if col_map:
        df = df.rename(columns=col_map)

    df["Frame"] = df["Frame"].astype(int)
    df["Trajectory"] = df["Trajectory"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def representative_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse each track to one representative (x, y, Frame) per track.
    Uses the median position across all frames, and the earliest frame as
    the representative frame (for display purposes).
    """
    agg = (
        df.groupby("Trajectory")
        .agg(
            x=("x", "median"),
            y=("y", "median"),
            frame_start=("Frame", "min"),
            frame_end=("Frame", "max"),
            n_frames=("Frame", "count"),
        )
        .reset_index()
    )
    return agg


def match_tracks_by_position(
    rep_a: pd.DataFrame,
    rep_b: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """
    For every track in rep_a, find the nearest track in rep_b (Euclidean
    distance in x,y).  A match is accepted if the distance ≤ tolerance.

    Returns a DataFrame of matched pairs with columns:
        traj_a, traj_b, distance
    """
    coords_a = rep_a[["x", "y"]].values
    coords_b = rep_b[["x", "y"]].values

    matches = []
    for i, (xa, ya) in enumerate(coords_a):
        dists = np.sqrt((coords_b[:, 0] - xa) ** 2 + (coords_b[:, 1] - ya) ** 2)
        j = int(np.argmin(dists))
        if dists[j] <= tolerance:
            matches.append(
                {
                    "traj_a": int(rep_a.iloc[i]["Trajectory"]),
                    "traj_b": int(rep_b.iloc[j]["Trajectory"]),
                    "dist_px": round(float(dists[j]), 2),
                    "x_consensus": round((xa + coords_b[j, 0]) / 2, 3),
                    "y_consensus": round((ya + coords_b[j, 1]) / 2, 3),
                }
            )
    return pd.DataFrame(matches)


def build_consensus(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    matched_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a consensus DataFrame that contains:
      - The rows from df_a for matched tracks
      - Extra columns: traj_b (counterpart ID), dist_px, x_consensus, y_consensus
    """
    if matched_pairs.empty:
        return pd.DataFrame()

    merged = df_a[df_a["Trajectory"].isin(matched_pairs["traj_a"])].copy()
    pair_info = matched_pairs[
        ["traj_a", "traj_b", "dist_px", "x_consensus", "y_consensus"]
    ].rename(columns={"traj_a": "Trajectory"})
    consensus = merged.merge(pair_info, on="Trajectory", how="left")
    return consensus


# ---------------------------------------------------------------------------
# Marking / visualisation
# ---------------------------------------------------------------------------

def mark_on_image(
    image_path: str | Path,
    consensus: pd.DataFrame,
    output_path: str | Path,
    marker_radius: int = 15,
    marker_color: str = "cyan",
    label_tracks: bool = True,
):
    """
    Draw circles around consensus mitotic cells on a reference image and save
    an annotated PNG.  Works with single-frame TIF files or the first frame of
    a multi-frame stack.

    Parameters
    ----------
    image_path : path to the reference TIF (e.g. MAX projection)
    consensus  : DataFrame with columns x, y (pixel coords), Trajectory
    output_path: where to save the annotated image
    marker_radius: circle radius in pixels
    marker_color : matplotlib colour string for the circles
    label_tracks : whether to print the track ID next to each marker
    """
    try:
        from tifffile import imread as tif_imread
    except ImportError:
        from PIL import Image as PILImage
        import numpy as _np
        img_arr = _np.array(PILImage.open(image_path))
    else:
        img_arr = tif_imread(image_path)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Use the first frame if the image is a stack (T, Y, X) or (Z, Y, X)
    while img_arr.ndim > 2:
        img_arr = img_arr[0]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_arr, cmap="gray", interpolation="none")

    # One representative position per track (median x, y)
    rep = consensus.groupby("Trajectory").agg(x=("x", "median"), y=("y", "median"))

    for traj_id, row in rep.iterrows():
        circle = mpatches.Circle(
            (row["x"], row["y"]),
            radius=marker_radius,
            linewidth=1.5,
            edgecolor=marker_color,
            facecolor="none",
        )
        ax.add_patch(circle)
        if label_tracks:
            ax.text(
                row["x"] + marker_radius + 2,
                row["y"],
                str(traj_id),
                color=marker_color,
                fontsize=7,
                va="center",
            )

    n = len(rep)
    ax.set_title(
        f"Consensus mitotic cells (double-confirmed across z-slices): {n}",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Annotated image saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validate mitotic cell detections from two z-slices and "
            "keep only cells confirmed in both."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--z8", required=True, metavar="CSV",
        help="Detected mitotic tracks CSV from z-slice 8 (e.g. set1002-1_detected_mitotic_tracks.csv)",
    )
    parser.add_argument(
        "--z7", required=True, metavar="CSV",
        help="Detected mitotic tracks CSV from z-slice 7 (e.g. Aset1002-1_detected_mitotic_tracks.csv)",
    )
    parser.add_argument(
        "--output", default="consensus_mitotic_tracks.csv", metavar="CSV",
        help="Output CSV for double-confirmed mitotic tracks",
    )
    parser.add_argument(
        "--tolerance", type=float, default=20.0, metavar="PX",
        help=(
            "Maximum spatial distance (pixels) between a cell in z8 and its "
            "counterpart in z7 to be considered the same cell"
        ),
    )
    parser.add_argument(
        "--image", default=None, metavar="TIF",
        help="Optional: reference image (TIF) on which to mark consensus cells",
    )
    parser.add_argument(
        "--marker-radius", type=int, default=15, metavar="PX",
        help="Circle radius (pixels) used when marking on the image",
    )
    parser.add_argument(
        "--marker-color", default="cyan",
        help="Matplotlib colour string for the circle markers",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Suppress track-ID labels next to each marker",
    )
    args = parser.parse_args()

    # --- load ---
    print(f"\nLoading z8 detections : {args.z8}")
    df_z8 = load_tracks(args.z8)
    print(f"  {len(df_z8):>6} rows | {df_z8['Trajectory'].nunique()} tracks")

    print(f"Loading z7 detections : {args.z7}")
    df_z7 = load_tracks(args.z7)
    print(f"  {len(df_z7):>6} rows | {df_z7['Trajectory'].nunique()} tracks")

    # --- representative positions ---
    rep_z8 = representative_positions(df_z8)
    rep_z7 = representative_positions(df_z7)

    # --- spatial matching ---
    print(f"\nMatching cells (tolerance = {args.tolerance} px) …")
    matched = match_tracks_by_position(rep_z8, rep_z7, tolerance=args.tolerance)

    if matched.empty:
        print(
            "\n  No cells matched within the given tolerance.\n"
            "  Try increasing --tolerance (current: {args.tolerance} px)."
        )
        sys.exit(0)

    print(f"  {len(matched)} / {len(rep_z8)} z8 tracks matched a z7 counterpart")
    print(f"  Match distance stats (px):  "
          f"min={matched['dist_px'].min():.1f}  "
          f"mean={matched['dist_px'].mean():.1f}  "
          f"max={matched['dist_px'].max():.1f}")

    # --- consensus DataFrame ---
    consensus = build_consensus(df_z8, df_z7, matched)

    # --- save consensus CSV ---
    out_path = Path(args.output)
    consensus.to_csv(out_path)
    print(f"\nConsensus CSV saved → {out_path}  ({len(consensus)} rows, "
          f"{consensus['Trajectory'].nunique()} tracks)")

    # Print a human-readable summary table
    summary = matched[["traj_a", "traj_b", "dist_px", "x_consensus", "y_consensus"]].copy()
    summary.columns = ["Track (z8)", "Track (z7)", "Dist (px)", "x", "y"]
    print("\nMatched pairs:")
    print(summary.to_string(index=False))

    # --- optional image marking ---
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"\n  WARNING: image not found at {img_path} — skipping marking.")
        else:
            out_img = out_path.with_suffix("").with_name(
                out_path.stem + "_marked.png"
            )
            print(f"\nMarking consensus cells on image: {img_path}")
            mark_on_image(
                image_path=img_path,
                consensus=consensus,
                output_path=out_img,
                marker_radius=args.marker_radius,
                marker_color=args.marker_color,
                label_tracks=not args.no_labels,
            )


if __name__ == "__main__":
    main()

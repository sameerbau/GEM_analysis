"""
compare_spt_ddm.py — overlay SPT per-trajectory D distribution with DDM/kICS/JDD.

Loads SPT outputs (histogram_diffusion.csv, diffusion_summary.csv) and overlays
them with DDM (tiled and/or global), kICS, and JDD-EM diffusion coefficients.

Usage
-----
    python compare_spt_ddm.py <spt_results_dir> [options]

Arguments
---------
    spt_results_dir  : directory containing histogram_diffusion.csv and
                       diffusion_summary.csv (SPT pipeline output)

Options
-------
    --ddm-tiled CSV  : path to ddm_tiled_D.csv from analyse_ddm_tiled()
    --ddm-global D   : single D_GEM value from DDM two-component fit
    --kics-global D  : single D_GEM value from kICS two-component fit
    --jdd-global D   : single D_GEM value from JDD-EM
    --embryo NAME    : embryo name substring filter for SPT FileName
    --out DIR        : output directory (default: spt_results_dir)
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_columns(path: Path) -> dict:
    """Read a CSV into dict of column-name -> list of strings."""
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return {}
    header = rows[0]
    cols = {h: [] for h in header}
    for row in rows[1:]:
        for h, v in zip(header, row):
            cols[h].append(v)
    return cols


def _as_float_array(values) -> np.ndarray:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def load_spt_histogram(spt_dir: Path, embryo: str | None = None) -> dict:
    """Load SPT per-trajectory D histogram and (optionally) summary."""
    hist_path = spt_dir / "histogram_diffusion.csv"
    summ_path = spt_dir / "diffusion_summary.csv"

    if not hist_path.exists():
        raise FileNotFoundError(f"Missing {hist_path}")

    cols = _read_csv_columns(hist_path)
    # Tolerant column name lookup
    def _find(name_substr: str):
        for k in cols.keys():
            if name_substr.lower() in k.lower():
                return k
        return None

    diff_col = _find("DiffusionCoefficient") or _find("Diffusion")
    freq_col = _find("NormalizedFrequency") or _find("Frequency") or _find("Density")
    file_col = _find("FileName") or _find("File")

    if diff_col is None or freq_col is None:
        raise ValueError(
            f"Could not locate DiffusionCoefficient/NormalizedFrequency columns "
            f"in {hist_path} (got {list(cols.keys())})"
        )

    D_all   = _as_float_array(cols[diff_col])
    freq    = _as_float_array(cols[freq_col])
    file_v  = cols.get(file_col, [""] * len(D_all)) if file_col else [""] * len(D_all)

    if embryo:
        keep = np.array([(embryo.lower() in (f or "").lower()) for f in file_v])
        if keep.sum() == 0:
            warnings.warn(f"--embryo {embryo!r} matched zero rows; ignoring filter.")
        else:
            D_all = D_all[keep]
            freq  = freq[keep]
            file_v = [f for f, k in zip(file_v, keep) if k]

    # Restrict to positive D
    valid = np.isfinite(D_all) & np.isfinite(freq) & (D_all > 0)
    D_pos = D_all[valid]
    f_pos = freq[valid]

    summary = {}
    if summ_path.exists():
        scols = _read_csv_columns(summ_path)
        for k, vs in scols.items():
            try:
                summary[k] = _as_float_array(vs)
            except Exception:
                summary[k] = vs

    # Extract median if available
    spt_median = np.nan
    for k in ("MedianD", "Median", "median", "MedianDiffusion", "median_D"):
        if k in summary:
            arr = summary[k]
            if isinstance(arr, np.ndarray) and arr.size:
                spt_median = float(np.nanmedian(arr))
                break
    if not np.isfinite(spt_median) and len(D_pos):
        spt_median = float(np.nanmedian(D_pos))

    return dict(
        D_bins      = D_pos,
        norm_freq   = f_pos,
        D_raw       = D_all,
        median_D    = spt_median,
        file_names  = file_v,
        summary     = summary,
    )


def load_ddm_tiled_csv(path: Path) -> np.ndarray:
    cols = _read_csv_columns(path)
    for key in ("D_GEM", "D_gem", "DGEM"):
        if key in cols:
            arr = _as_float_array(cols[key])
            if "converged" in cols:
                ok = _as_float_array(cols["converged"]).astype(int).astype(bool)
                arr = np.where(ok, arr, np.nan)
            return arr
    raise ValueError(f"Could not find D_GEM column in {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Gaussian KDE on log10(D)
# ─────────────────────────────────────────────────────────────────────────────

def _kde_log10(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE on log10(D). Returns density on linear D x_grid."""
    v = values[np.isfinite(values) & (values > 0)]
    if len(v) < 2:
        return np.zeros_like(x_grid)
    log_v = np.log10(v)
    # Silverman's rule of thumb
    std = np.std(log_v, ddof=1)
    iqr = np.subtract(*np.percentile(log_v, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.2
    h = 0.9 * sigma * len(v) ** (-0.2)
    log_x = np.log10(x_grid)
    # Pairwise gaussian sum
    diffs = (log_x[:, None] - log_v[None, :]) / h
    kvals = np.exp(-0.5 * diffs ** 2) / (np.sqrt(2 * np.pi) * h)
    return kvals.mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_spt_vs_ddm(
    spt: dict,
    ddm_tiled: np.ndarray | None,
    ddm_global: float | None,
    kics_global: float | None,
    jdd_global: float | None,
    embryo_name: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # SPT histogram: D_bins are the bin centers, norm_freq = density
    D_bins = spt["D_bins"]
    freq   = spt["norm_freq"]
    if len(D_bins):
        # Plot as a step-like filled histogram using the existing binning.
        order = np.argsort(D_bins)
        D_s = D_bins[order]
        f_s = freq[order]
        # Compute widths (log10 equal spacing assumed)
        if len(D_s) > 1:
            log_edges = np.zeros(len(D_s) + 1)
            log_D = np.log10(D_s)
            log_edges[1:-1] = 0.5 * (log_D[:-1] + log_D[1:])
            log_edges[0]    = log_D[0]  - (log_D[1] - log_D[0]) / 2
            log_edges[-1]   = log_D[-1] + (log_D[-1] - log_D[-2]) / 2
            edges = 10 ** log_edges
            widths = np.diff(edges)
        else:
            widths = np.full_like(D_s, D_s[0] * 0.1)
        ax.bar(D_s, f_s, width=widths, align="center",
               color="lightblue", edgecolor="steelblue", alpha=0.7,
               label=f"SPT per-trajectory D (n_bins={len(D_s)})")

    # DDM tiled KDE
    if ddm_tiled is not None:
        x_grid = np.logspace(np.log10(0.005), np.log10(1.0), 400)
        dens   = _kde_log10(ddm_tiled, x_grid)
        if dens.max() > 0 and len(D_bins) and freq.max() > 0:
            dens = dens * (freq.max() / dens.max())
        ax.plot(x_grid, dens, "-", color="darkorange", lw=2.0,
                label=f"DDM spatial tiles (n={int(np.sum(np.isfinite(ddm_tiled)))})")

    # Vertical reference lines
    spt_med = spt.get("median_D", np.nan)
    if np.isfinite(spt_med):
        ax.axvline(spt_med, color="steelblue", ls="--", lw=1.5,
                   label=f"SPT median = {spt_med:.4f}")
    if ddm_global is not None and np.isfinite(ddm_global):
        ax.axvline(ddm_global, color="red", ls="--", lw=1.5,
                   label=f"DDM global D_GEM = {ddm_global:.4f}")
    if kics_global is not None and np.isfinite(kics_global):
        ax.axvline(kics_global, color="green", ls="--", lw=1.5,
                   label=f"kICS global D_GEM = {kics_global:.4f}")
    if jdd_global is not None and np.isfinite(jdd_global):
        ax.axvline(jdd_global, color="purple", ls="--", lw=1.5,
                   label=f"JDD-EM D_GEM = {jdd_global:.4f}")

    ax.set_xscale("log")
    ax.set_xlim(0.005, 1.0)
    ax.set_xlabel("D (µm²/s)")
    ax.set_ylabel("Normalised frequency / density")
    ax.set_title(f"D distribution: SPT vs DDM — {embryo_name}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {out_path}")


def plot_method_bar(
    spt_median: float,
    ddm_global: float | None,
    kics_global: float | None,
    jdd_global: float | None,
    out_path: Path,
    spt_err: float | None = None,
) -> None:
    labels, values, errs = [], [], []
    if np.isfinite(spt_median):
        labels.append("SPT median"); values.append(spt_median)
        errs.append(spt_err if spt_err is not None else 0.0)
    if ddm_global is not None and np.isfinite(ddm_global):
        labels.append("DDM global"); values.append(ddm_global); errs.append(0.0)
    if kics_global is not None and np.isfinite(kics_global):
        labels.append("kICS global"); values.append(kics_global); errs.append(0.0)
    if jdd_global is not None and np.isfinite(jdd_global):
        labels.append("JDD-EM"); values.append(jdd_global); errs.append(0.0)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    if values:
        colors = ["steelblue", "red", "green", "purple"][: len(values)]
        ax.bar(labels, values, yerr=errs, color=colors, alpha=0.8,
               edgecolor="black", capsize=4)
        if np.isfinite(spt_median):
            ax.axhline(spt_median, color="steelblue", ls=":", lw=1.2,
                       label=f"SPT median ref = {spt_median:.4f}")
            ax.legend(fontsize=8)
    ax.set_ylabel("D  (µm²/s)")
    ax.set_title("Diffusion coefficient by method")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SPT D distribution to DDM/kICS/JDD.")
    p.add_argument("spt_results_dir", type=Path,
                   help="Dir with histogram_diffusion.csv + diffusion_summary.csv")
    p.add_argument("--ddm-tiled", type=Path, default=None,
                   help="Path to ddm_tiled_D.csv")
    p.add_argument("--ddm-global", type=float, default=None,
                   help="Global DDM two-component D_GEM (µm²/s)")
    p.add_argument("--kics-global", type=float, default=None,
                   help="Global kICS two-component D_GEM (µm²/s)")
    p.add_argument("--jdd-global", type=float, default=None,
                   help="Global JDD-EM D_GEM (µm²/s)")
    p.add_argument("--embryo", type=str, default=None,
                   help="Substring filter on FileName column in SPT histogram")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default = spt_results_dir)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    spt_dir = args.spt_results_dir
    out_dir = Path(args.out or spt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embryo_name = args.embryo or spt_dir.name

    print(f"[compare] SPT dir: {spt_dir}")
    try:
        spt = load_spt_histogram(spt_dir, embryo=args.embryo)
    except Exception as exc:
        print(f"   Failed to load SPT data: {exc}")
        return 2
    print(f"   SPT bins loaded: {len(spt['D_bins'])}  median D = {spt['median_D']:.5f}")

    ddm_tiled_arr = None
    if args.ddm_tiled is not None:
        try:
            ddm_tiled_arr = load_ddm_tiled_csv(args.ddm_tiled)
            n_ok = int(np.sum(np.isfinite(ddm_tiled_arr)))
            print(f"   DDM tiled D values: {n_ok} / {len(ddm_tiled_arr)} valid")
        except Exception as exc:
            print(f"   Failed to load DDM-tiled CSV: {exc}")
            ddm_tiled_arr = None

    plot_spt_vs_ddm(
        spt, ddm_tiled_arr,
        args.ddm_global, args.kics_global, args.jdd_global,
        embryo_name=embryo_name,
        out_path=out_dir / "spt_vs_ddm_overlay.png",
    )
    plot_method_bar(
        spt["median_D"],
        args.ddm_global, args.kics_global, args.jdd_global,
        out_path=out_dir / "method_comparison_bar.png",
    )

    # Explanatory block
    print("\nSPT vs DDM distributions:")
    print(" SPT: per-trajectory D — heterogeneity of individual particles "
          "(includes measurement noise per track)")
    print(" DDM tiled: per-tile ensemble D — spatial heterogeneity; each tile "
          "is an ensemble average")
    print("   → DDM tile distribution is narrower than SPT (ensemble averaging "
          "suppresses single-particle noise)")
    print("   → But both should have similar median D_GEM")
    print(" DDM/kICS global: single ensemble-average D for the full FOV")

    # Summary text file
    try:
        summ_path = out_dir / "spt_vs_ddm_summary.txt"
        with open(summ_path, "w") as fh:
            fh.write(f"Embryo: {embryo_name}\n")
            fh.write(f"SPT median D : {spt['median_D']:.6f} µm²/s "
                     f"(n_bins={len(spt['D_bins'])})\n")
            if ddm_tiled_arr is not None:
                vals = ddm_tiled_arr[np.isfinite(ddm_tiled_arr)]
                if len(vals):
                    fh.write(f"DDM tiles    : median={np.median(vals):.6f}  "
                             f"mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  "
                             f"n={len(vals)}\n")
            if args.ddm_global is not None:
                fh.write(f"DDM global   : {args.ddm_global:.6f}\n")
            if args.kics_global is not None:
                fh.write(f"kICS global  : {args.kics_global:.6f}\n")
            if args.jdd_global is not None:
                fh.write(f"JDD-EM global: {args.jdd_global:.6f}\n")
        print(f"   summary → {summ_path}")
    except Exception as exc:
        print(f"   summary write failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

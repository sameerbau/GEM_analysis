"""
ddm_analysis.py — Differential Dynamic Microscopy for GEM diffusion.

Theory
------
For a widefield movie I(r, t), the image structure function is

    S(q, τ) = ⟨|I_q(t+τ) − I_q(t)|²⟩_t
            = 2·PS(q) − 2·Re[C(q, τ)]
            = A(q)·[1 − exp(−D·q²·τ)] + B(q)        (single Brownian species)

where PS(q) = ⟨|F_q(t)|²⟩ is the time-averaged power spectrum and
C(q,τ) = ⟨F_q*(t)·F_q(t+τ)⟩ is the temporal autocorrelation in k-space.

Key properties
--------------
Immobile particles cancel:  ΔI_immobile(τ) = I(t+τ) − I(t) = 0, so they
  contribute identically to PS and C, and vanish from S(q,τ).

Species separation by q:
  τ_c(q) = 1 / (D·q²)     correlation time of species with diffusion D.

  Fast diffusers (D ≈ 0.45 µm²/s): at q ≥ 8 rad/µm → τ_c ≤ 0.035 s < one frame.
    Their signal has already decayed to the B(q) noise floor at τ = 0.1 s.

  GEM 40-mers (D ≈ 0.044 µm²/s): at q = 10 rad/µm → τ_c ≈ 0.23 s = 2.3 frames.
    Signal decays comfortably within the 0.1–3 s fitting window.

  Slow / immobile:  cancelled as described above.

Therefore fitting S(q,τ) at q ≥ 8 rad/µm recovers D_GEM directly, free of
fast-diffuser contamination and immobile-fraction bias.

Two-component fit
-----------------
At small q both fast and GEM components are visible.  At large q only GEM
survives at τ ≥ 0.1 s.  Plotting D(q) — the single-component D from fitting
each q bin independently — reveals this transition: D(q) drops from D_fast
(small q) to D_GEM (large q) as the fast component exits the fit window.

Usage
-----
    python ddm_analysis.py /tmp/Em1_crop.tif 0.04374
"""

import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import curve_fit

# ── constants ────────────────────────────────────────────────────────────────
PIXEL_UM  = 0.094
DT        = 0.1
MAX_LAG   = 30
N_Q_BINS  = 35        # number of radial q bins across the full spectrum
Q_MIN_FIT = 8.0       # rad/µm — fast component already decayed here
Q_MAX_FIT = 22.0      # rad/µm — beyond PSF spatial scale, SNR drops


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing  (verbatim from imaging_fcs.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_tiff_stack(path: Path) -> np.ndarray:
    raw = tifffile.imread(str(path)).astype(np.float32)
    if raw.ndim == 2:
        raw = raw[np.newaxis]
    elif raw.ndim == 4:
        raw = raw[:, 0]
    if raw.ndim != 3:
        raise ValueError(f"Expected (T,Y,X), got {raw.shape}")
    return raw


def correct_bleaching(movie: np.ndarray) -> np.ndarray:
    frame_means = movie.mean(axis=(1, 2)).clip(min=1e-6)
    return movie * (frame_means[0] / frame_means)[:, None, None]


def subtract_background(movie: np.ndarray, percentile: float = 5) -> np.ndarray:
    bg = np.percentile(movie, percentile, axis=0)
    return np.maximum(movie - bg[np.newaxis], 0.0)


def preprocess(movie: np.ndarray, correct_drift_flag: bool = False) -> tuple:
    movie = correct_bleaching(movie)
    movie = subtract_background(movie)
    if correct_drift_flag:
        from imaging_fcs import correct_drift
        movie, shifts = correct_drift(movie)
    else:
        shifts = np.zeros((movie.shape[0], 2), dtype=np.float32)
    return movie, shifts


def _fft_crop(movie: np.ndarray) -> np.ndarray:
    T, Y, X = movie.shape
    side = 1 << int(np.floor(np.log2(min(Y, X))))
    y0, x0 = (Y - side) // 2, (X - side) // 2
    return movie[:, y0: y0 + side, x0: x0 + side]


# ═══════════════════════════════════════════════════════════════════════════════
# DDM core
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ddm_structure_function(
    movie: np.ndarray,
    max_lag: int = MAX_LAG,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    n_q_bins: int = N_Q_BINS,
    q_min_um: float | None = None,
    q_max_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute isotropic image structure function S(q, τ).

    S(q,τ) = 2·PS(q) − 2·Re[C(q,τ)]

    This is computed entirely from FFTs; no particle detection or linking.

    Parameters
    ----------
    movie     : (T, Y, X) float32 preprocessed movie (bleach + background corrected)
    max_lag   : maximum lag in frames
    pixel_um  : pixel size in µm
    dt        : frame interval in s
    n_q_bins  : number of radial q bins
    q_min_um  : minimum q in rad/µm (default: 2π / FOV_size)
    q_max_um  : maximum q in rad/µm (default: Nyquist = π / pixel_um)

    Returns
    -------
    S_mat     : (max_lag, n_q_bins) structure function
    q_centers : (n_q_bins,) bin centres in rad/µm
    taus      : (max_lag,) lag times in seconds
    """
    T, Y, X = movie.shape

    # Use float64 throughout: S = 2PS − 2C is a difference of large numbers
    arr   = movie.astype(np.float64)
    F_all = np.fft.rfft2(arr)               # (T, Y, X//2+1), complex128
    PS    = (np.abs(F_all) ** 2).mean(axis=0)   # time-averaged power spectrum

    # Radial q map (rad/µm)
    qy    = np.fft.fftfreq(Y, d=pixel_um) * 2 * np.pi   # (Y,)
    qx    = np.fft.rfftfreq(X, d=pixel_um) * 2 * np.pi  # (X//2+1,)
    QY, QX = np.meshgrid(qy, qx, indexing='ij')
    Q_map = np.sqrt(QY ** 2 + QX ** 2)         # (Y, X//2+1)

    # Q bin edges
    q_lo = q_min_um or (2 * np.pi / (min(Y, X) * pixel_um) * 1.5)
    q_hi = q_max_um or (np.pi / pixel_um)
    q_edges   = np.linspace(q_lo, q_hi, n_q_bins + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

    # Radial masks and PS per bin
    masks = []
    PS_q  = np.zeros(n_q_bins, dtype=np.float64)
    for i in range(n_q_bins):
        m = (Q_map >= q_edges[i]) & (Q_map < q_edges[i + 1])
        masks.append(m)
        PS_q[i] = PS[m].mean() if m.sum() > 0 else np.nan

    # Structure function S(q,τ) = 2·PS(q) − 2·Re[C(q,τ)]
    S_mat = np.zeros((max_lag, n_q_bins), dtype=np.float64)
    taus  = np.arange(1, max_lag + 1, dtype=np.float64) * dt

    for tau in range(1, max_lag + 1):
        n     = T - tau
        if n <= 0:
            break
        cross = (np.conj(F_all[:n]) * F_all[tau: tau + n]).mean(axis=0)
        C_re  = np.real(cross)                  # (Y, X//2+1)
        for iq, mask in enumerate(masks):
            if mask.sum() > 0 and np.isfinite(PS_q[iq]):
                S_mat[tau - 1, iq] = 2.0 * PS_q[iq] - 2.0 * C_re[mask].mean()

    return S_mat, q_centers, taus


def fit_ddm_per_q(
    S_mat: np.ndarray,
    q_centers: np.ndarray,
    taus: np.ndarray,
    q_min_fit: float = Q_MIN_FIT,
    q_max_fit: float = Q_MAX_FIT,
    lag_min: int = 1,
    lag_max: int = 20,
) -> dict:
    """
    Fit S(q,τ) = A(q)·[1−exp(−D·q²·τ)] + B(q) at each q bin independently.

    A(q)  — signal amplitude (mobile species at that spatial scale)
    B(q)  — noise floor (shot noise + fast-decayed component residual)
    D     — fitted per q; should be flat ≈ D_GEM for q in the fit range
              where the fast component has already decayed to B(q).

    q_min_fit / q_max_fit are in rad/µm (same units as q_centers).

    Returns dict with D_per_q, A_per_q, B_per_q, D_median, D_mean, D_sem, etc.
    """
    n_q      = len(q_centers)
    D_per_q  = np.full(n_q, np.nan)
    A_per_q  = np.full(n_q, np.nan)
    B_per_q  = np.full(n_q, np.nan)

    lag_min  = max(1, lag_min)
    lag_max  = min(lag_max, S_mat.shape[0])
    s_slice  = slice(lag_min - 1, lag_max)
    tau_fit  = taus[s_slice]

    for iq, q in enumerate(q_centers):
        if q < q_min_fit or q > q_max_fit:
            continue
        s = S_mat[s_slice, iq]
        if not np.all(np.isfinite(s)) or s.max() <= 0:
            continue

        # Model: S(τ) = A·(1 − exp(−Γ·τ)) + B,  Γ = D·q²
        def _model(tau, A, gamma, B):
            return A * (1.0 - np.exp(-gamma * tau)) + B

        s_max = float(s.max())
        s_min = float(max(s.min(), 0.0))
        p0     = [max(s_max - s_min, 1e-20), 0.044 * q ** 2, s_min]
        bounds = ([0.0, 1e-8, 0.0], [s_max * 10, q ** 2 * 20, s_max * 2])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(_model, tau_fit, s, p0=p0, bounds=bounds,
                                    maxfev=15000, method='trf')
            A, gamma, B = popt
            if A > 0 and gamma > 0:
                D_per_q[iq] = gamma / q ** 2
                A_per_q[iq] = A
                B_per_q[iq] = B
        except Exception:
            pass

    q_mask  = (q_centers >= q_min_fit) & (q_centers <= q_max_fit)
    D_valid = D_per_q[q_mask & np.isfinite(D_per_q)]

    return dict(
        D_per_q   = D_per_q,
        A_per_q   = A_per_q,
        B_per_q   = B_per_q,
        q_centers = q_centers,
        q_mask    = q_mask,
        taus      = taus,
        S_mat     = S_mat,
        D_median  = float(np.nanmedian(D_valid)) if len(D_valid) else np.nan,
        D_mean    = float(np.nanmean(D_valid))   if len(D_valid) else np.nan,
        D_sem     = float(np.std(D_valid) / np.sqrt(len(D_valid)))
                    if len(D_valid) > 1 else np.nan,
        n_valid_q = int(len(D_valid)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ddm_results(
    fit: dict,
    tracking_D: float | None = None,
    imsd_D: float | None = None,
    movie_name: str = "",
    out_path: Path | None = None,
) -> None:
    """Four-panel DDM diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    q     = fit["q_centers"]
    D_q   = fit["D_per_q"]
    taus  = fit["taus"]
    S     = fit["S_mat"]
    mask  = fit["q_mask"]
    A_q   = fit["A_per_q"]
    B_q   = fit["B_per_q"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"DDM analysis — {movie_name}", fontsize=11)

    # ── Panel 1: S(q,τ) curves for selected q bins ───────────────────────────
    ax = axes[0, 0]
    q_sel = np.percentile(q[mask & np.isfinite(D_q)],
                          [10, 30, 50, 70, 90]).round(1)
    colors = viridis(np.linspace(0.1, 0.9, len(q_sel)))
    for qi, col in zip(q_sel, colors):
        iq = np.argmin(np.abs(q - qi))
        ax.plot(taus, S[:, iq], "-", color=col, lw=1.2,
                label=f"q={q[iq]:.1f}")
        if np.isfinite(D_q[iq]) and np.isfinite(A_q[iq]):
            gamma = D_q[iq] * q[iq] ** 2
            s_fit = A_q[iq] * (1 - np.exp(-gamma * taus)) + B_q[iq]
            ax.plot(taus, s_fit, "--", color=col, lw=0.8, alpha=0.7)
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("S(q,τ)  (a.u.)")
    ax.set_title("Structure function curves + fits")
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 2: D(q) plot — key result ──────────────────────────────────────
    ax = axes[0, 1]
    valid = mask & np.isfinite(D_q)
    ax.plot(q[~mask & np.isfinite(D_q)], D_q[~mask & np.isfinite(D_q)],
            "o", color="lightgray", ms=5, label="outside fit range")
    ax.plot(q[valid], D_q[valid], "o", color="steelblue", ms=6,
            label=f"fit range  median={fit['D_median']:.4f} µm²/s")
    ax.axhline(fit["D_median"], color="steelblue", lw=1.5, ls="--",
               label=f"DDM median D = {fit['D_median']:.4f}")
    if tracking_D is not None:
        ax.axhline(tracking_D, color="red", lw=1.5, ls=":",
                   label=f"tracking D = {tracking_D:.4f}")
    if imsd_D is not None and np.isfinite(imsd_D):
        ax.axhline(imsd_D, color="green", lw=1.5, ls="-.",
                   label=f"iMSD D = {imsd_D:.4f}")
    # Mark fit range
    ax.axvspan(Q_MIN_FIT, Q_MAX_FIT, alpha=0.08, color="steelblue")
    ax.set_xlabel("q  (rad/µm)"); ax.set_ylabel("D(q)  (µm²/s)")
    ax.set_title("D(q) — flat region = single Brownian species")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.2, np.nanmax(D_q) * 1.3))

    # ── Panel 3: A(q) signal amplitude ───────────────────────────────────────
    ax = axes[1, 0]
    ax.semilogy(q[np.isfinite(A_q)], A_q[np.isfinite(A_q)], "s-",
                color="darkorange", ms=4)
    ax.set_xlabel("q  (rad/µm)"); ax.set_ylabel("A(q)  (a.u.)")
    ax.set_title("Signal amplitude A(q)  (mobile species power spectrum)")

    # ── Panel 4: normalised S(q,τ) vs q²τ (master curve) ────────────────────
    ax = axes[1, 1]
    q_plot = q[valid]
    for iq_abs in np.where(valid)[0]:
        qi = q[iq_abs]
        s  = S[:, iq_abs]
        A  = A_q[iq_abs]
        B  = B_q[iq_abs]
        if A > 0:
            g_norm = (s - B) / A          # → 1 − exp(−Dq²τ)
            q2tau  = qi ** 2 * taus
            ax.plot(q2tau, g_norm, ".", ms=2, alpha=0.4, color="steelblue")
    # Reference: exp curve at D_median
    xref = np.linspace(0, q_plot.max() ** 2 * taus[-1], 200)
    ax.plot(xref, 1 - np.exp(-fit["D_median"] * xref), "r-", lw=1.5,
            label=f"D={fit['D_median']:.4f} µm²/s")
    if tracking_D is not None:
        ax.plot(xref, 1 - np.exp(-tracking_D * xref), "k:", lw=1.5,
                label=f"tracking D={tracking_D:.4f}")
    ax.set_xlabel("q²τ  (rad²·s/µm²)")
    ax.set_ylabel("[S(q,τ) − B(q)] / A(q)")
    ax.set_title("Master curve — data should collapse for single species")
    ax.set_xlim(0, min(50, q_plot.max() ** 2 * taus[-1]))
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt; _plt.close(fig)
        print(f"   DDM plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt; _plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_ddm(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    max_lag: int = MAX_LAG,
    lag_min: int = 1,
    lag_max: int = 20,
    n_q_bins: int = N_Q_BINS,
    q_min_fit: float = Q_MIN_FIT,
    q_max_fit: float = Q_MAX_FIT,
    correct_drift_flag: bool = False,
) -> dict:
    """
    Full DDM pipeline on one TIFF movie.

    Steps
    -----
    1. Load + preprocess (bleach, background, optional drift)
    2. Centre-crop to power-of-2 square
    3. Compute S(q,τ) for all q bins and lags
    4. Fit D(q) per q bin with single-exponential model
    5. Report D_median over q_min_fit–q_max_fit
    6. Save plot

    Returns dict with DDM fit results.
    """
    import time
    tiff_path = Path(tiff_path)
    name      = tiff_path.stem
    out       = Path(out_dir or tiff_path.parent / "ddm_results")
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + preprocess ─────────────────────────────────────────────────
    print(f"\n[DDM] {name}")
    print("[1/4] Loading + preprocessing …")
    movie_raw          = load_tiff_stack(tiff_path)
    movie_pp, _        = preprocess(movie_raw, correct_drift_flag=correct_drift_flag)
    del movie_raw
    movie_c            = _fft_crop(movie_pp)
    T, Y, X            = movie_c.shape
    print(f"      {T}×{Y}×{X}  (cropped to power-of-2)")

    # ── 2. Structure function ────────────────────────────────────────────────
    print("[2/4] Computing S(q,τ) …")
    t0 = time.time()
    S_mat, q_centers, taus = compute_ddm_structure_function(
        movie_c, max_lag=max_lag, pixel_um=pixel_um, dt=dt, n_q_bins=n_q_bins,
    )
    print(f"      done in {time.time()-t0:.1f}s")

    # ── 3. Fit D(q) ──────────────────────────────────────────────────────────
    print(f"[3/4] Fitting D(q) over q={q_min_fit:.0f}–{q_max_fit:.0f} rad/µm …")
    fit = fit_ddm_per_q(S_mat, q_centers, taus,
                        q_min_fit=q_min_fit, q_max_fit=q_max_fit,
                        lag_min=lag_min, lag_max=lag_max)

    print(f"\n   DDM D (median, q={q_min_fit:.0f}–{q_max_fit:.0f} rad/µm): "
          f"{fit['D_median']:.5f} µm²/s  "
          f"(mean={fit['D_mean']:.5f}, sem={fit['D_sem']:.5f}, "
          f"n_q={fit['n_valid_q']})")
    if tracking_D is not None and np.isfinite(fit["D_median"]):
        print(f"   Tracking D:  {tracking_D:.5f}  ratio = {fit['D_median']/tracking_D:.3f}×")

    print("\n   D(q) table:")
    print(f"   {'q (rad/µm)':>12}  {'τ_c_GEM(s)':>10}  {'τ_c_fast(s)':>12}  "
          f"{'D_fit (µm²/s)':>14}  in_range?")
    for iq, q in enumerate(q_centers):
        tc_gem  = 1 / (0.044 * q ** 2) if q > 0 else np.inf
        tc_fast = 1 / (0.45  * q ** 2) if q > 0 else np.inf
        D       = fit["D_per_q"][iq]
        flag    = "  ←" if fit["q_mask"][iq] else ""
        Dstr    = f"{D:.4f}" if np.isfinite(D) else "  ---"
        print(f"   {q:>12.2f}  {tc_gem:>10.3f}  {tc_fast:>12.4f}  {Dstr:>14}{flag}")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    print("[4/4] Saving plot …")

    # Also grab raw-STICS iMSD D for comparison
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import imaging_fcs as _fcs
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _acf = _fcs.compute_stics(movie_c, max_lag=max_lag, intensity_mask_frac=0.10)
            _res = _fcs.compute_imsd(_acf, pixel_um=pixel_um, dt=dt,
                                     lag_min=1, lag_max=20)
        imsd_D = _res["D"]
        print(f"   Raw iMSD D (for overlay): {imsd_D:.5f} µm²/s")
    except Exception:
        imsd_D = None

    plot_ddm_results(fit, tracking_D=tracking_D, imsd_D=imsd_D,
                     movie_name=name,
                     out_path=out / f"{name}_ddm.png")
    fit["movie_name"] = name
    fit["imsd_D"]     = imsd_D
    fit["movie_pp"]   = movie_pp
    return fit


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, time
    tiff      = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/Em1_crop.tif")
    tracking_D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04374
    out_dir   = Path("/tmp/ddm_out")
    out_dir.mkdir(exist_ok=True)

    t0  = time.time()
    fit = analyse_ddm(tiff, tracking_D=tracking_D, out_dir=out_dir,
                      q_min_fit=Q_MIN_FIT, q_max_fit=Q_MAX_FIT)
    print(f"\n   Total: {time.time()-t0:.0f}s   Output → {out_dir}/")

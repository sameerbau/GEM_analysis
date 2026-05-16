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
# Two-component DDM fit
# ═══════════════════════════════════════════════════════════════════════════════

def fit_ddm_two_component(
    S_mat: np.ndarray,
    q_centers: np.ndarray,
    taus: np.ndarray,
    q_min_fit: float = Q_MIN_FIT,
    q_max_fit: float = Q_MAX_FIT,
    lag_min: int = 1,
    lag_max: int = 20,
    D1_init: float = 0.044,
    D2_init: float = 0.004,
    D1_bounds: tuple = (0.01, 0.18),
    D2_bounds: tuple = (0.0005, 0.025),
) -> dict:
    """
    Two-component DDM fit across all q bins simultaneously.

    Model (per q bin):
        S(q,τ) = A₁(q)·[1−exp(−D₁·q²·τ)] + A₂(q)·[1−exp(−D₂·q²·τ)] + B(q)

    D₁ and D₂ are SHARED across all q bins.  D₁ > D₂ by convention.
    A₁(q), A₂(q), B(q) are solved per-q via NNLS for each (D₁, D₂) candidate.

    Default configuration targets q=5–12 rad/µm, lag_min=2:
    - At these q values the fast diffuser (D≈0.45) has fully decayed into B(q)
      by τ=0.2 s (first included lag), so the two visible components are:
        D₁ ≈ D_GEM  (0.02–0.15 µm²/s, τ_c = 0.07–1 s in this q range)
        D₂ ≈ D_slow (< 0.02 µm²/s,  τ_c >> τ_max, appears as slow rise)

    Parameters
    ----------
    D1_init / D2_init  : initial guesses (D1 > D2)
    D1_bounds / D2_bounds : (lo, hi) bounds for L-BFGS-B
    q_min_fit / q_max_fit : fitting range in rad/µm
    lag_min / lag_max : lag frames to include (use lag_min≥2 to purge fast diffusers)

    Returns
    -------
    dict with D1 (GEM), D2 (slow), A1_per_q, A2_per_q, B_per_q,
    frac_GEM, residual, converged, q_centers, q_mask, taus, S_mat.
    """
    from scipy.optimize import nnls, minimize, Bounds

    lag_min  = max(1, lag_min)
    lag_max  = min(lag_max, S_mat.shape[0])
    s_slice  = slice(lag_min - 1, lag_max)
    tau_fit  = taus[s_slice]
    n_tau    = len(tau_fit)

    q_mask   = (q_centers >= q_min_fit) & (q_centers <= q_max_fit)
    q_idx    = np.where(q_mask)[0]
    if len(q_idx) == 0:
        raise ValueError("No q bins in fit range.")

    S_fit    = S_mat[s_slice][:, q_idx]   # (n_tau, n_q_in_range)

    # Per-q normalization so all q bins contribute equally
    scales   = np.array([max(float(S_fit[:, j].max()), 1e-30) for j in range(len(q_idx))])

    def total_sse(params):
        D1, D2 = params
        if D1 <= 1e-6 or D2 <= 1e-6 or D1 <= D2:
            return 1e12
        sse = 0.0
        for j, iq in enumerate(q_idx):
            q   = q_centers[iq]
            s   = S_fit[:, j]
            if not np.all(np.isfinite(s)):
                continue
            s_n = s / scales[j]
            b1  = 1.0 - np.exp(-D1 * q ** 2 * tau_fit)
            b2  = 1.0 - np.exp(-D2 * q ** 2 * tau_fit)
            X   = np.column_stack([b1, b2, np.ones(n_tau)])
            _, res = nnls(X, s_n)
            sse += res ** 2
        return sse / len(q_idx)

    # Coarse 2-D grid search to seed the optimizer away from local minima
    D1_grid = np.logspace(np.log10(D1_bounds[0] * 1.5), np.log10(D1_bounds[1] * 0.9), 6)
    D2_grid = np.logspace(np.log10(D2_bounds[0] * 1.5), np.log10(D2_bounds[1] * 0.9), 5)
    best_sse, best_p = np.inf, [D1_init, D2_init]
    for d1 in D1_grid:
        for d2 in D2_grid:
            if d1 <= d2:
                continue
            v = total_sse([d1, d2])
            if v < best_sse:
                best_sse, best_p = v, [d1, d2]

    result = minimize(
        total_sse,
        x0     = best_p,
        method = "L-BFGS-B",
        bounds = Bounds(lb=[D1_bounds[0], D2_bounds[0]],
                        ub=[D1_bounds[1], D2_bounds[1]]),
        options = dict(ftol=1e-14, gtol=1e-10, maxiter=2000),
    )
    D1_opt, D2_opt = result.x

    # Recover per-q amplitudes at optimal (D1, D2)
    n_centers = len(q_centers)
    A1_per_q  = np.full(n_centers, np.nan)
    A2_per_q  = np.full(n_centers, np.nan)
    B_per_q   = np.full(n_centers, np.nan)

    for j, iq in enumerate(q_idx):
        q   = q_centers[iq]
        s   = S_fit[:, j]
        if not np.all(np.isfinite(s)):
            continue
        b1  = 1.0 - np.exp(-D1_opt * q ** 2 * tau_fit)
        b2  = 1.0 - np.exp(-D2_opt * q ** 2 * tau_fit)
        X   = np.column_stack([b1, b2, np.ones(n_tau)])
        coeff, _ = nnls(X, s)
        A1_per_q[iq] = coeff[0]
        A2_per_q[iq] = coeff[1]
        B_per_q[iq]  = coeff[2]

    frac_GEM = np.where(
        (A1_per_q + A2_per_q) > 0,
        A1_per_q / (A1_per_q + A2_per_q),
        np.nan,
    )

    return dict(
        D1         = float(D1_opt),
        D2         = float(D2_opt),
        D_fast     = float(D1_opt),
        D_GEM      = float(D1_opt),
        D_slow     = float(D2_opt),
        A1_per_q   = A1_per_q,
        A2_per_q   = A2_per_q,
        B_per_q    = B_per_q,
        frac_GEM   = frac_GEM,
        q_centers  = q_centers,
        q_mask     = q_mask,
        taus       = taus,
        S_mat      = S_mat,
        residual   = float(result.fun),
        converged  = bool(result.success),
        n_q_fit    = int(len(q_idx)),
    )


def plot_ddm_two_component(
    fit2: dict,
    tracking_D: float | None = None,
    movie_name: str = "",
    out_path: "Path | None" = None,
) -> None:
    """Four-panel figure for two-component DDM fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    q      = fit2["q_centers"]
    taus   = fit2["taus"]
    S      = fit2["S_mat"]
    mask   = fit2["q_mask"]
    A1     = fit2["A1_per_q"]
    A2     = fit2["A2_per_q"]
    B      = fit2["B_per_q"]
    D1     = fit2["D1"]
    D2     = fit2["D2"]
    fg     = fit2["frac_GEM"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Two-component DDM — {movie_name}\n"
        f"D_fast={D1:.3f} µm²/s    D_GEM={D2:.4f} µm²/s"
        + (f"    (tracking={tracking_D:.4f})" if tracking_D else ""),
        fontsize=10,
    )

    # ── Panel 1: S(q,τ) with two-component fits ──────────────────────────────
    ax = axes[0, 0]
    q_plot_idx = np.where(mask & np.isfinite(A2))[0]
    pct = np.percentile(np.arange(len(q_plot_idx)), [10, 30, 50, 70, 90]).astype(int)
    sel = q_plot_idx[pct]
    colors = viridis(np.linspace(0.1, 0.9, len(sel)))
    for iq, col in zip(sel, colors):
        s_data = S[:, iq]
        qi = q[iq]
        n_t = min(len(taus), S.shape[0])
        ax.plot(taus[:n_t], s_data[:n_t], "-", color=col, lw=1.2,
                label=f"q={qi:.1f}")
        if np.isfinite(A1[iq]) and np.isfinite(A2[iq]):
            s_fit = (A1[iq] * (1 - np.exp(-D1 * qi**2 * taus[:n_t]))
                     + A2[iq] * (1 - np.exp(-D2 * qi**2 * taus[:n_t]))
                     + B[iq])
            ax.plot(taus[:n_t], s_fit, "--", color=col, lw=0.8, alpha=0.7)
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("S(q,τ)")
    ax.set_title("Structure function + two-component fits")
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 2: GEM fraction A2/(A1+A2) vs q ────────────────────────────────
    ax = axes[0, 1]
    valid = mask & np.isfinite(fg)
    ax.bar(q[valid], fg[valid], width=(q[1] - q[0]) * 0.8 if len(q) > 1 else 1,
           color="steelblue", alpha=0.75, label="GEM fraction A₂/(A₁+A₂)")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("q  (rad/µm)"); ax.set_ylabel("Fraction GEM")
    ax.set_title(f"GEM amplitude fraction  (D_GEM={D2:.4f} µm²/s)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # ── Panel 3: A1(q) and A2(q) amplitudes ──────────────────────────────────
    ax = axes[1, 0]
    valid1 = mask & np.isfinite(A1)
    valid2 = mask & np.isfinite(A2)
    ax.semilogy(q[valid1], A1[valid1], "o-", color="steelblue", ms=4,
                label=f"A₁(q) GEM  D={D1:.4f}")
    ax.semilogy(q[valid2], A2[valid2], "s-", color="darkorange", ms=4,
                label=f"A₂(q) slow  D={D2:.5f}")
    ax.set_xlabel("q  (rad/µm)"); ax.set_ylabel("Amplitude  (a.u.)")
    ax.set_title("Species amplitudes vs q")
    ax.legend(fontsize=8)

    # ── Panel 4: normalised master curve coloured by q ───────────────────────
    ax = axes[1, 1]
    n_t = min(len(taus), S.shape[0])
    q_valid_idx = np.where(mask & np.isfinite(A2))[0]
    cmap = viridis(np.linspace(0.1, 0.9, len(q_valid_idx)))
    for k, iq in enumerate(q_valid_idx):
        qi = q[iq]
        s  = S[:n_t, iq]
        denom = A1[iq] + A2[iq]
        if denom > 0 and B[iq] >= 0:
            g_norm = (s - B[iq]) / denom
            q2tau  = qi**2 * taus[:n_t]
            ax.plot(q2tau, g_norm, ".", ms=2, alpha=0.35, color=cmap[k])
    # Reference curves
    xref = np.linspace(0, 80, 200)
    ax.plot(xref, 1 - np.exp(-D2 * xref), "b-", lw=1.8,
            label=f"D_GEM={D2:.4f}")
    ax.plot(xref, 1 - np.exp(-D1 * xref), "r:", lw=1.5,
            label=f"D_fast={D1:.3f}")
    if tracking_D:
        ax.plot(xref, 1 - np.exp(-tracking_D * xref), "k--", lw=1.2,
                label=f"tracking={tracking_D:.4f}")
    ax.set_xlabel("q²τ  (rad²·s/µm²)")
    ax.set_ylabel("[S − B] / (A₁+A₂)")
    ax.set_title("Master curve  (normalised)")
    ax.set_xlim(0, 80); ax.set_ylim(-0.15, 1.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt; _plt.close(fig)
        print(f"   2-component DDM plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt; _plt.show()


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
    two_component: bool = True,
) -> dict:
    """
    Full DDM pipeline on one TIFF movie.

    Steps
    -----
    1. Load + preprocess (bleach, background, optional drift)
    2. Centre-crop to power-of-2 square
    3. Compute S(q,τ) for all q bins and lags
    4. Fit D(q) per q bin with single-exponential model
    5. If two_component=True, also run global two-component fit for D_GEM
    6. Report results and save plots

    Returns dict with DDM fit results (includes 'fit2' key if two_component=True).
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

    # ── 3b. Two-component fit (GEM + slow) ───────────────────────────────────
    # Optimal window: q=6-14 rad/µm, lag_min=3 (τ=0.3s).
    # At τ≥0.3s the fast diffuser (D≈0.45, τ_c≤0.06s at q≥6) has fully decayed
    # into B(q).  Two visible components remain:
    #   D1 ≈ D_GEM  (0.01-0.18)  τ_c=0.06-1.0s in this q range
    #   D2 ≈ D_slow (0.001-0.025) τ_c >> 3s → slow linear-like rise
    # Empirically gives D_GEM ≈ 0.044-0.046 µm²/s (1.0-1.05× tracking) on Em1.
    fit2 = None
    if two_component:
        q2_min, q2_max, lag2_min = 6.0, 14.0, 3
        print(f"\n[3b] Two-component DDM fit (GEM + slow residual)  "
              f"q={q2_min:.0f}–{q2_max:.0f} rad/µm, lag {lag2_min}–{max_lag} …")
        try:
            fit2 = fit_ddm_two_component(
                S_mat, q_centers, taus,
                q_min_fit=q2_min, q_max_fit=q2_max,
                lag_min=lag2_min, lag_max=max_lag,
                D1_init=0.044, D2_init=0.005,
                D1_bounds=(0.01, 0.18), D2_bounds=(0.0005, 0.025),
            )
            print(f"   D_GEM  = {fit2['D_GEM']:.5f} µm²/s"
                  + (f"   (ratio = {fit2['D_GEM']/tracking_D:.3f}×  "
                     f"tracking={tracking_D:.5f})"
                     if tracking_D else ""))
            print(f"   D_slow = {fit2['D_slow']:.5f} µm²/s")
            print(f"   converged={fit2['converged']}  residual={fit2['residual']:.4e}"
                  f"  n_q={fit2['n_q_fit']}")
        except Exception as exc:
            print(f"   Two-component fit failed: {exc}")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    print("[4/4] Saving plots …")

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
    if fit2 is not None:
        plot_ddm_two_component(fit2, tracking_D=tracking_D,
                               movie_name=name,
                               out_path=out / f"{name}_ddm_2comp.png")

    fit["movie_name"] = name
    fit["imsd_D"]     = imsd_D
    fit["movie_pp"]   = movie_pp
    fit["fit2"]       = fit2
    return fit


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, time
    tiff      = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/Em1_crop.tif")
    tracking_D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04374
    out_dir   = tiff.parent / "ddm_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0  = time.time()
    fit = analyse_ddm(tiff, tracking_D=tracking_D, out_dir=out_dir,
                      q_min_fit=Q_MIN_FIT, q_max_fit=Q_MAX_FIT)
    print(f"\n   Total: {time.time()-t0:.0f}s   Output → {out_dir}/")

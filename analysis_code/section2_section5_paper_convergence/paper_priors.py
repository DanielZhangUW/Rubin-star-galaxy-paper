"""Magnitude-prior helpers for COSMOS paper figures."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from paper_sample_selection import truth_masks


PRIOR_LOG10_NG_OVER_NS_BOUNDS = (-1.5, 2.0)


def _as_float_array(values):
    scalar = np.isscalar(values)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
        scalar = True
    return arr, scalar


def _restore_type(values, arr, name=None):
    if np.isscalar(values):
        return float(np.asarray(arr, dtype=float).reshape(-1)[0])
    if isinstance(values, pd.Series):
        return pd.Series(np.asarray(arr, dtype=float), index=values.index, name=name or values.name)
    return np.asarray(arr, dtype=float)


def _validate_prior_model(model: str) -> None:
    if model != "by_eye":
        raise ValueError(f"unsupported S/G prior model {model!r}; only 'by_eye' is implemented")


def _check_prior_bounds(log10_ng_over_ns, bounds_action: str) -> None:
    if bounds_action not in {"warn", "raise", "ignore"}:
        raise ValueError("bounds_action must be 'warn', 'raise', or 'ignore'")
    if bounds_action == "ignore":
        return
    lo, hi = PRIOR_LOG10_NG_OVER_NS_BOUNDS
    arr = np.asarray(log10_ng_over_ns, dtype=float)
    bad = np.isfinite(arr) & ((arr < lo) | (arr > hi))
    if not np.any(bad):
        return
    msg = (
        "by-eye S/G prior produced log10(NG/NS) values outside "
        f"[{lo}, {hi}]: min={np.nanmin(arr[bad]):.3f}, max={np.nanmax(arr[bad]):.3f}"
    )
    if bounds_action == "raise":
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def log10_ng_over_ns_prior_from_rmag(r_cmodel_mag, model: str = "by_eye", bounds_action: str = "warn"):
    """Return the magnitude prior as log10(N_galaxy / N_star).

    The current paper default is the by-eye r-band CModel-magnitude fit:

    - 1.40 + 0.28 * (rmag - 24), for rmag < 24
    - 1.40 + 0.20 * (rmag - 24), for rmag >= 24

    Non-finite magnitudes return NaN and are excluded from bounds checks.
    """

    _validate_prior_model(model)
    mag, scalar = _as_float_array(r_cmodel_mag)
    out = np.full(mag.shape, np.nan, dtype=float)
    finite = np.isfinite(mag)
    bright = finite & (mag < 24.0)
    faint = finite & ~bright
    out[bright] = 1.40 + 0.28 * (mag[bright] - 24.0)
    out[faint] = 1.40 + 0.20 * (mag[faint] - 24.0)
    _check_prior_bounds(out, bounds_action)
    if scalar:
        return float(out[0])
    return _restore_type(r_cmodel_mag, out, name="log10_NG_over_NS_prior")


def log10_ns_over_ng_prior_from_rmag(r_cmodel_mag, model: str = "by_eye", bounds_action: str = "warn"):
    """Return log10(N_star / N_galaxy), the inverse of log10(NG/NS)."""

    ng_over_ns = log10_ng_over_ns_prior_from_rmag(
        r_cmodel_mag, model=model, bounds_action=bounds_action
    )
    return -ng_over_ns


def log_prior_odds_star_over_gal(r_cmodel_mag, model: str = "by_eye", bounds_action: str = "warn"):
    """Return natural-log prior odds ln[P(S|m) / P(G|m)]."""

    log10_ns_over_ng = log10_ns_over_ng_prior_from_rmag(
        r_cmodel_mag, model=model, bounds_action=bounds_action
    )
    return np.log(10.0) * log10_ns_over_ng


def apply_sg_prior_to_log_likelihood_ratio(
    log_lr_star_over_gal,
    r_cmodel_mag,
    model: str = "by_eye",
    bounds_action: str = "warn",
):
    """Add the r-magnitude S/G prior once to a morphology log-likelihood ratio.

    `log_lr_star_over_gal` must be a natural-log morphology likelihood ratio:
    logL_star - logL_galaxy. The returned value is posterior log odds:

    logL_star - logL_galaxy + ln[P(S|m) / P(G|m)].
    """

    lr, scalar = _as_float_array(log_lr_star_over_gal)
    prior = np.asarray(
        log_prior_odds_star_over_gal(r_cmodel_mag, model=model, bounds_action=bounds_action),
        dtype=float,
    )
    out = lr + prior
    if scalar and np.asarray(out).size == 1:
        return float(np.asarray(out).reshape(-1)[0])
    return _restore_type(log_lr_star_over_gal, out, name="posterior_log_odds_star_over_gal")


def star_galaxy_prior_table(matched: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    stars, galaxies = truth_masks(matched)
    mag = pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = mag.ge(lo) & mag.lt(hi)
        n_star = int((in_bin & stars).sum())
        n_gal = int((in_bin & galaxies).sum())
        total = n_star + n_gal
        rows.append(
            {
                "mag_low": lo,
                "mag_high": hi,
                "mag_center": 0.5 * (lo + hi),
                "N_matched_star": n_star,
                "N_matched_galaxy": n_gal,
                "star_to_galaxy_ratio": n_star / n_gal if n_gal else np.nan,
                "p_star_prior": n_star / total if total else np.nan,
                "p_gal_prior": n_gal / total if total else np.nan,
                "notes": "counts from COSMOS2020-matched paper sample",
            }
        )
    return pd.DataFrame(rows)

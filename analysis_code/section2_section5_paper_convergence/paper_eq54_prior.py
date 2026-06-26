"""Explicit Eq.54-style prior calibration helpers for Section 2.

This module treats stored v9 pS values as pre-prior/model scores for this
controlled task, converts them to model log-likelihood-ratio-like logits, and
applies the explicit magnitude-dependent S/G prior from v9_ratio_data.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z", "y")
EPS_DEFAULT = 1e-6


@dataclass(frozen=True)
class PriorTable:
    """Validated explicit S/G prior table."""

    table: pd.DataFrame
    source_path: Path
    field: str
    band: str
    prior_column: str
    prior_direction: str
    magnitude_definition: str


def sigmoid(values):
    """Numerically stable logistic sigmoid."""

    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)
    pos = finite & (x >= 0)
    neg = finite & (x < 0)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    if np.isscalar(values):
        return float(out.reshape(-1)[0])
    if isinstance(values, pd.Series):
        return pd.Series(out, index=values.index, name=values.name)
    return out


def safe_logit(values, eps: float = EPS_DEFAULT):
    """Return logit(values) after clipping finite probabilities to [eps, 1-eps]."""

    p = np.asarray(values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    clipped = np.clip(p[finite], eps, 1.0 - eps)
    out[finite] = np.log(clipped / (1.0 - clipped))
    if np.isscalar(values):
        return float(out.reshape(-1)[0])
    if isinstance(values, pd.Series):
        return pd.Series(out, index=values.index, name=values.name)
    return out


def parse_mag_bin(mag_bin: str) -> tuple[float, float]:
    """Parse a left-closed/right-open magnitude bin string like '23.0-23.5'."""

    low_text, high_text = str(mag_bin).split("-", maxsplit=1)
    return float(low_text), float(high_text)


def load_v9_ps_table(path: Path, bands: tuple[str, ...] = BANDS) -> pd.DataFrame:
    """Load the stored v9 pS table columns needed for calibration."""

    columns = ["object_id", "ra", "dec", *[f"pS_{band}" for band in bands], "ps_version", "field"]
    return pd.read_parquet(path, columns=columns)


def load_v9_ratio_prior(
    path: Path,
    field: str = "COSMOS",
    band: str = "r",
    prior_column: str = "log10_NG_NS",
) -> PriorTable:
    """Load and validate the explicit magnitude-dependent prior table.

    The accepted table convention is log10_NG_NS = log10(N_G / N_S).
    """

    ratio = pd.read_csv(path)
    required = {"field", "band", "mag_bin", "star_fraction_wU", "galaxy_fraction", prior_column}
    missing = sorted(required - set(ratio.columns))
    if missing:
        raise KeyError(f"missing required prior columns in {path}: {missing}")

    selected = ratio[ratio["field"].eq(field) & ratio["band"].eq(band)].copy()
    if selected.empty:
        raise ValueError(f"no explicit prior rows for field={field!r}, band={band!r}")

    selected[["mag_low", "mag_high"]] = selected["mag_bin"].apply(
        lambda value: pd.Series(parse_mag_bin(value))
    )
    selected["star_fraction_wU"] = pd.to_numeric(selected["star_fraction_wU"], errors="coerce")
    selected["galaxy_fraction"] = pd.to_numeric(selected["galaxy_fraction"], errors="coerce")
    selected[prior_column] = pd.to_numeric(selected[prior_column], errors="coerce")

    finite = selected.dropna(subset=["star_fraction_wU", "galaxy_fraction", prior_column]).copy()
    if finite.empty:
        raise ValueError(f"no finite explicit prior values for field={field!r}, band={band!r}")
    bad_fraction = ~(
        finite["star_fraction_wU"].between(0, 1)
        & finite["galaxy_fraction"].between(0, 1)
        & finite["star_fraction_wU"].gt(0)
        & finite["galaxy_fraction"].gt(0)
    )
    if bool(bad_fraction.any()):
        raise ValueError("explicit prior table has non-finite or out-of-range fractions")
    galaxy_resid = np.abs(finite["galaxy_fraction"] - (1.0 - finite["star_fraction_wU"]))
    if float(galaxy_resid.max()) > 1e-5:
        raise ValueError("explicit prior table galaxy_fraction is not consistent with 1 - star_fraction_wU")
    recomputed = np.log10(finite["galaxy_fraction"] / finite["star_fraction_wU"])
    log_resid = np.abs(finite[prior_column] - recomputed)
    if float(log_resid.max()) > 5e-5:
        raise ValueError(f"{prior_column} is not consistent with log10(galaxy_fraction/star_fraction)")

    selected = selected.sort_values("mag_low").reset_index(drop=True)
    return PriorTable(
        table=selected,
        source_path=path,
        field=field,
        band=band,
        prior_column=prior_column,
        prior_direction="NG_over_NS",
        magnitude_definition="uncorrected r-band CModel magnitude; left-closed/right-open mag_bin",
    )


def infer_prior_ratio_direction(prior: PriorTable) -> str:
    """Return the validated ratio direction for this explicit prior table."""

    return prior.prior_direction


def assign_prior_by_rmag(r_cmodel_mag: pd.Series, prior: PriorTable) -> pd.DataFrame:
    """Assign explicit prior values to objects by r-band CModel magnitude."""

    mag = pd.to_numeric(r_cmodel_mag, errors="coerce")
    intervals = pd.IntervalIndex.from_arrays(
        prior.table["mag_low"],
        prior.table["mag_high"],
        closed="left",
    )
    idx = intervals.get_indexer(mag.to_numpy(dtype=float))
    out = pd.DataFrame(index=r_cmodel_mag.index)
    out["prior_mag_bin"] = pd.Series(pd.NA, index=r_cmodel_mag.index, dtype="object")
    out["prior_mag_low"] = np.nan
    out["prior_mag_high"] = np.nan
    out["log10_NG_over_NS_eq54prior"] = np.nan
    valid_idx = idx >= 0
    if np.any(valid_idx):
        matched = prior.table.iloc[idx[valid_idx]].reset_index(drop=True)
        target_index = out.index[valid_idx]
        out.loc[target_index, "prior_mag_bin"] = matched["mag_bin"].to_numpy()
        out.loc[target_index, "prior_mag_low"] = matched["mag_low"].to_numpy(dtype=float)
        out.loc[target_index, "prior_mag_high"] = matched["mag_high"].to_numpy(dtype=float)
        out.loc[target_index, "log10_NG_over_NS_eq54prior"] = matched[prior.prior_column].to_numpy(dtype=float)
    out["log_prior_odds_eq54"] = -np.log(10.0) * out["log10_NG_over_NS_eq54prior"]
    return out


def apply_eq54_prior_to_single_band(
    p_pre,
    log_prior_odds,
    eps: float = EPS_DEFAULT,
):
    """Apply the explicit prior once to a single-band pre-prior pS score."""

    log_lr = safe_logit(p_pre, eps=eps)
    posterior_logit = np.asarray(log_lr, dtype=float) + np.asarray(log_prior_odds, dtype=float)
    return sigmoid(posterior_logit), log_lr


def combine_loglr_multiband(
    loglr_frame: pd.DataFrame,
    bands: tuple[str, ...],
    require_all_bands: bool = True,
) -> pd.Series:
    """Combine model logLRs across bands before applying the prior once."""

    cols = [f"logLR_{band}_model" for band in bands]
    values = loglr_frame[cols].apply(pd.to_numeric, errors="coerce")
    if require_all_bands:
        valid = values.notna().all(axis=1)
    else:
        valid = values.notna().any(axis=1)
    summed = values.sum(axis=1, skipna=not require_all_bands)
    summed[~valid] = np.nan
    return summed


def compute_eq54_prior_calibrated_ps(
    ps: pd.DataFrame,
    r_cmodel_mag: pd.Series,
    prior: PriorTable,
    eps: float = EPS_DEFAULT,
    bands: tuple[str, ...] = BANDS,
) -> pd.DataFrame:
    """Return pS table augmented with Eq.54-prior calibrated columns."""

    out = ps.copy()
    out["cmodel_mag_r"] = pd.to_numeric(r_cmodel_mag, errors="coerce")
    prior_assigned = assign_prior_by_rmag(out["cmodel_mag_r"], prior)
    out = pd.concat([out, prior_assigned], axis=1)

    for band in bands:
        p_col = f"pS_{band}"
        eq_col = f"pS_{band}_eq54prior"
        loglr_col = f"logLR_{band}_model"
        eq, loglr = apply_eq54_prior_to_single_band(out[p_col], out["log_prior_odds_eq54"], eps=eps)
        out[loglr_col] = loglr
        out[eq_col] = eq

    for name, combo_bands in {
        "r": ("r",),
        "gri": ("g", "r", "i"),
        "ugrizy": ("u", "g", "r", "i", "z", "y"),
    }.items():
        combined_loglr = combine_loglr_multiband(out, combo_bands, require_all_bands=True)
        out[f"logLR_{name}_model"] = combined_loglr
        out[f"pS_{name}_eq54prior"] = sigmoid(combined_loglr + out["log_prior_odds_eq54"])

    return out


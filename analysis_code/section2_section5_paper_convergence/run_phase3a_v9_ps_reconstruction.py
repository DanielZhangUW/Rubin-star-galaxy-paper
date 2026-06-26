"""Validate v9 fit parameters against stored COSMOS r-band pS values.

This Phase 3A diagnostic is intentionally small: it reads existing parquet/CSV
inputs, samples at most 5000 COSMOS r-band objects, and writes only summary
tables/figures. It does not write object-level pS products.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfc

from paper_priors import log_prior_odds_star_over_gal


FIT_TABLE = Path("paper_convergence/tables/v9_fit_parameters.csv")
ANALYSIS_TABLE = Path("outputs/dp2_cosmos_analysis_table.parquet")
PS_TABLE = Path("outputs/dp2_cosmos_ps_v9.parquet")
RESULT_DIR = Path("paper_convergence/results/section2_bayesian_method")
FIGURE_DIR = Path("paper_convergence/figures/section2_bayesian_method")

SAMPLE_SIZE = 5000
RANDOM_SEED = 20260615
MAG_COL = "cmodel_mag_r"
DM_COL = "psf_minus_cmodel_r"
PS_COL = "pS_r"
ID_COL = "object_id"


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def gauss(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.abs(sigma) + 1e-12
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def skewnorm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    sigma = np.abs(sigma) + 1e-12
    t = (x - mu) / sigma
    return 2.0 * gauss(x, mu, sigma) * 0.5 * erfc(-alpha * t / np.sqrt(2.0))


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def parse_mag_bin(text: str) -> tuple[float, float]:
    low_text, high_text = str(text).split("-", maxsplit=1)
    return float(low_text), float(high_text)


def load_fit_params(repo_root: Path) -> pd.DataFrame:
    fit = pd.read_csv(repo_root / FIT_TABLE)
    required = [
        "field",
        "band",
        "mag_bin",
        "muU",
        "sigmaU",
        "wU",
        "wR1",
        "muR1",
        "sigmaR1",
        "alphaR1",
        "wR2",
        "muR2",
        "sigmaR2",
        "alphaR2",
        "wR3",
        "muR3",
        "sigmaR3",
        "alphaR3",
    ]
    missing = [col for col in required if col not in fit.columns]
    if missing:
        raise KeyError(f"missing v9 fit columns: {missing}")
    fit = fit[fit["field"].eq("COSMOS") & fit["band"].eq("r")].copy()
    if fit.empty:
        raise ValueError("no COSMOS r-band rows in v9 fit table")
    fit[["mag_low", "mag_high"]] = fit["mag_bin"].apply(
        lambda value: pd.Series(parse_mag_bin(value))
    )
    fit = fit.sort_values("mag_low").reset_index(drop=True)
    return fit


def load_validation_sample(repo_root: Path) -> pd.DataFrame:
    analysis = pd.read_parquet(repo_root / ANALYSIS_TABLE, columns=[ID_COL, MAG_COL, DM_COL])
    ps = pd.read_parquet(repo_root / PS_TABLE, columns=[ID_COL, PS_COL])
    sample = analysis.merge(ps, on=ID_COL, how="inner")
    valid = (
        np.isfinite(sample[MAG_COL])
        & np.isfinite(sample[DM_COL])
        & np.isfinite(sample[PS_COL])
        & sample[MAG_COL].ge(16.0)
        & sample[MAG_COL].lt(26.0)
    )
    sample = sample.loc[valid, [ID_COL, MAG_COL, DM_COL, PS_COL]].copy()
    sample = sample.sample(
        n=min(SAMPLE_SIZE, len(sample)),
        random_state=RANDOM_SEED,
        replace=False,
    ).sort_values(ID_COL).reset_index(drop=True)
    return sample


def attach_fit_params(sample: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    intervals = pd.IntervalIndex.from_arrays(fit["mag_low"], fit["mag_high"], closed="left")
    idx = intervals.get_indexer(sample[MAG_COL])
    if np.any(idx < 0):
        raise ValueError("sample contains magnitudes outside fit-table bins")
    out = sample.copy()
    fit_cols = [
        "mag_bin",
        "muU",
        "sigmaU",
        "wU",
        "wR1",
        "muR1",
        "sigmaR1",
        "alphaR1",
        "wR2",
        "muR2",
        "sigmaR2",
        "alphaR2",
        "wR3",
        "muR3",
        "sigmaR3",
        "alphaR3",
    ]
    for col in fit_cols:
        out[col] = fit.iloc[idx][col].to_numpy()
    return out


def compute_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    dm = df[DM_COL].to_numpy(dtype=float)
    p_star_shape = gauss(dm, df["muU"].to_numpy(float), df["sigmaU"].to_numpy(float))
    p_star_weighted = df["wU"].to_numpy(float) * p_star_shape

    p_gal_weighted = np.zeros(len(df), dtype=float)
    for k in (1, 2, 3):
        w = df[f"wR{k}"].to_numpy(float)
        component = skewnorm(
            dm,
            df[f"muR{k}"].to_numpy(float),
            df[f"sigmaR{k}"].to_numpy(float),
            df[f"alphaR{k}"].to_numpy(float),
        )
        p_gal_weighted += w * component

    denom = p_star_weighted + p_gal_weighted
    reconstructed = np.divide(
        p_star_weighted,
        denom,
        out=np.full_like(denom, np.nan),
        where=denom > 0,
    )

    out = df.copy()
    out["p_star_weighted"] = p_star_weighted
    out["p_gal_weighted"] = p_gal_weighted
    out["pS_reconstructed"] = reconstructed
    out["abs_diff_reconstructed_minus_stored"] = np.abs(out["pS_reconstructed"] - out[PS_COL])
    return out


def add_morphology_prior_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    dm = df[DM_COL].to_numpy(dtype=float)
    p_star_shape = gauss(dm, df["muU"].to_numpy(float), df["sigmaU"].to_numpy(float))
    p_gal_shape = np.zeros(len(df), dtype=float)
    wR_sum = np.zeros(len(df), dtype=float)
    for k in (1, 2, 3):
        wR_sum += df[f"wR{k}"].to_numpy(float)
    for k in (1, 2, 3):
        w = df[f"wR{k}"].to_numpy(float)
        component = skewnorm(
            dm,
            df[f"muR{k}"].to_numpy(float),
            df[f"sigmaR{k}"].to_numpy(float),
            df[f"alphaR{k}"].to_numpy(float),
        )
        normalized_w = np.divide(w, wR_sum, out=np.zeros_like(w), where=wR_sum > 0)
        p_gal_shape += normalized_w * component

    floor = 1e-300
    loglr_morph = np.log(np.clip(p_star_shape, floor, np.inf)) - np.log(np.clip(p_gal_shape, floor, np.inf))
    prior_log_odds = log_prior_odds_star_over_gal(df[MAG_COL].to_numpy(float), bounds_action="ignore")
    out = df.copy()
    out["wR_sum"] = wR_sum
    out["logLR_morph"] = loglr_morph
    out["prior_log_odds_star_over_gal"] = prior_log_odds
    out["pS_prior_diagnostic"] = sigmoid(loglr_morph + prior_log_odds)
    return out


def summarize(result: pd.DataFrame, fit: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    diff = result["abs_diff_reconstructed_minus_stored"]
    finite = result[np.isfinite(diff)].copy()
    corr = float(np.corrcoef(finite[PS_COL], finite["pS_reconstructed"])[0, 1]) if len(finite) > 1 else np.nan
    median_abs = float(diff.median())
    p95_abs = float(diff.quantile(0.95))
    max_abs = float(diff.max())
    acceptable = bool(median_abs < 1e-10 and p95_abs < 1e-6 and max_abs < 1e-3)
    rows = [
        {
            "diagnostic": "weighted_pS_reconstruction",
            "N_tested": int(len(result)),
            "median_abs_diff": median_abs,
            "p95_abs_diff": p95_abs,
            "max_abs_diff": max_abs,
            "correlation": corr,
            "acceptable": acceptable,
            "morphology_logLR_computed": acceptable,
            "prior_diagnostic_computed": acceptable,
            "magnitude_column": MAG_COL,
            "delta_column": DM_COL,
            "stored_pS_column": PS_COL,
            "fit_rows_COSMOS_r": int(len(fit)),
            "fit_has_fit_success": "fit_success" in fit.columns,
            "fit_has_fallback_used": "fallback_used" in fit.columns,
            "notes": "weighted reconstruction uses wU*Gaussian over weighted unresolved+resolved mixture",
        }
    ]
    for bin_name, group in result.groupby("mag_bin", sort=True):
        d = group["abs_diff_reconstructed_minus_stored"]
        rows.append(
            {
                "diagnostic": f"weighted_pS_reconstruction_bin_{bin_name}",
                "N_tested": int(len(group)),
                "median_abs_diff": float(d.median()),
                "p95_abs_diff": float(d.quantile(0.95)),
                "max_abs_diff": float(d.max()),
                "correlation": np.nan,
                "acceptable": acceptable,
                "morphology_logLR_computed": acceptable,
                "prior_diagnostic_computed": acceptable,
                "magnitude_column": MAG_COL,
                "delta_column": DM_COL,
                "stored_pS_column": PS_COL,
                "fit_rows_COSMOS_r": int(len(fit)),
                "fit_has_fit_success": "fit_success" in fit.columns,
                "fit_has_fallback_used": "fallback_used" in fit.columns,
                "notes": "per-bin absolute reconstruction-difference summary",
            }
        )
    return pd.DataFrame(rows), acceptable


def write_markdown(path: Path, summary: pd.DataFrame, result: pd.DataFrame, acceptable: bool) -> None:
    overall = summary.iloc[0]
    worst = result.nlargest(10, "abs_diff_reconstructed_minus_stored")[
        [ID_COL, MAG_COL, DM_COL, "mag_bin", PS_COL, "pS_reconstructed", "abs_diff_reconstructed_minus_stored"]
    ]
    lines = [
        "# Phase 3A v9 pS Reconstruction Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Fit parameters: `{FIT_TABLE}`",
        f"- COSMOS analysis table: `{ANALYSIS_TABLE}`",
        f"- Stored v9 pS: `{PS_TABLE}`",
        "",
        "## Columns And Conventions",
        "",
        f"- Object ID column: `{ID_COL}`",
        f"- Magnitude column: `{MAG_COL}`",
        f"- Morphology delta column: `{DM_COL}`",
        f"- Delta sign convention: `PSF magnitude - CModel magnitude`; `r_diff` is identical to `{DM_COL}` in the analysis table.",
        f"- Stored pS column: `{PS_COL}`",
        "- Magnitude bin convention: left-closed, right-open bins from `mag_bin`, e.g. `23.0-23.5` means `23.0 <= rmag < 23.5`.",
        "- Fit parameter table has no `fit_success` or `fallback_used` columns.",
        "",
        "## Weighted pS Reconstruction Formula",
        "",
        "`p_star_weighted = wU * Gaussian(dm | muU, sigmaU)`",
        "",
        "`p_gal_weighted = sum_k wRk * SkewNormal(dm | muRk, sigmaRk, alphaRk)`",
        "",
        "`pS_reconstructed = p_star_weighted / (p_star_weighted + p_gal_weighted)`",
        "",
        "The skew-normal convention follows `src/psf_cmodel_fit.py`: `Gaussian * erfc(-alpha * t / sqrt(2))`, where `t=(dm-mu)/sigma`.",
        "",
        "## Reconstruction Metrics",
        "",
        f"- Objects tested: `{int(overall['N_tested'])}`",
        f"- Median absolute difference: `{overall['median_abs_diff']}`",
        f"- 95th percentile absolute difference: `{overall['p95_abs_diff']}`",
        f"- Max absolute difference: `{overall['max_abs_diff']}`",
        f"- Correlation: `{overall['correlation']}`",
        f"- Reconstruction acceptable: `{acceptable}`",
        "",
        "## Morphology-Only logLR Formula",
        "",
        "Computed only when weighted pS reconstruction is acceptable.",
        "",
        "`p(dm | S) = Gaussian(dm | muU, sigmaU)`",
        "",
        "`p(dm | G) = sum_k [wRk / sum(wR)] * SkewNormal(dm | muRk, sigmaRk, alphaRk)`",
        "",
        "`logLR_morph = log p(dm | S) - log p(dm | G)`",
        "",
        "The total `wU` and total `sum(wR)` class weights are excluded from morphology-only logLR.",
        "",
        "## Prior Diagnostic",
        "",
        "Computed only on this small sample, without writing object-level pS products:",
        "",
        "`posterior_log_odds = logLR_morph + log_prior_odds_star_over_gal(rmag)`",
        "",
        "`pS_prior_diagnostic = sigmoid(posterior_log_odds)`",
        "",
        f"- Morphology logLR computed: `{acceptable}`",
        f"- Prior diagnostic computed: `{acceptable}`",
        "",
        "## Worst Reconstruction Differences",
        "",
        worst.to_csv(index=False).strip(),
        "",
        "## Scope",
        "",
        "- No v8/v9 pS parquet files were modified.",
        "- No large object-level output table was written.",
        "- Fig 2.4-2.7 were not regenerated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_design_note(path: Path, acceptable: bool, summary: pd.DataFrame) -> None:
    overall = summary.iloc[0]
    block = [
        "",
        "## Phase 3A COSMOS r-band Reconstruction Check",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Fit table: `{FIT_TABLE}`",
        f"- Stored pS table: `{PS_TABLE}`",
        f"- Analysis table: `{ANALYSIS_TABLE}`",
        f"- Magnitude column: `{MAG_COL}`",
        f"- Morphology delta column: `{DM_COL}` (`PSF magnitude - CModel magnitude`)",
        f"- Stored pS column: `{PS_COL}`",
        "- Magnitude bins: left-closed, right-open `mag_bin` intervals from the fit table.",
        "",
        "Weighted reconstruction:",
        "`pS = wU*Gaussian_U / (wU*Gaussian_U + sum_k wRk*SkewNormal_Rk)`.",
        "",
        "Morphology-only logLR:",
        "`logLR_morph = log Gaussian_U - log sum_k[(wRk/sum(wR))*SkewNormal_Rk]`.",
        "",
        f"- Objects tested: `{int(overall['N_tested'])}`",
        f"- Median absolute difference: `{overall['median_abs_diff']}`",
        f"- 95th percentile absolute difference: `{overall['p95_abs_diff']}`",
        f"- Max absolute difference: `{overall['max_abs_diff']}`",
        f"- Correlation: `{overall['correlation']}`",
        f"- Reconstruction validates fit-parameter interpretation: `{acceptable}`",
        f"- Safe to proceed to Phase 3B small-sample prior-corrected pS: `{acceptable}`",
    ]
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Morphology Log-Likelihood Ratio Design Note\n"
    marker = "\n## Phase 3A COSMOS r-band Reconstruction Check\n"
    if marker in existing:
        existing = existing.split(marker, maxsplit=1)[0].rstrip()
    path.write_text(existing.rstrip() + "\n" + "\n".join(block) + "\n", encoding="utf-8")


def plot_reconstruction(path: Path, result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(result[PS_COL], result["pS_reconstructed"], s=6, alpha=0.25, color="black", rasterized=True)
    ax.plot([0, 1], [0, 1], color="tab:red", lw=1.2)
    ax.set_xlabel("stored v9 pS_r")
    ax.set_ylabel("reconstructed weighted pS_r")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="0.88", lw=0.7)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prior_effect(path: Path, result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.scatter(result[MAG_COL], result["pS_reconstructed"], s=6, alpha=0.18, label="weighted v9 pS", rasterized=True)
    ax.scatter(result[MAG_COL], result["pS_prior_diagnostic"], s=6, alpha=0.18, label="morphology logLR + by-eye prior", rasterized=True)
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("P(star)")
    ax.set_xlim(16, 26)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = repo_root_from_file()
    result_dir = repo_root / RESULT_DIR
    figure_dir = repo_root / FIGURE_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fit = load_fit_params(repo_root)
    sample = load_validation_sample(repo_root)
    with_params = attach_fit_params(sample, fit)
    result = compute_reconstruction(with_params)
    summary, acceptable = summarize(result, fit)
    if acceptable:
        result = add_morphology_prior_diagnostic(result)

    summary_path = result_dir / "phase3a_v9_ps_reconstruction_summary.csv"
    md_path = result_dir / "phase3a_v9_ps_reconstruction_summary.md"
    recon_plot = figure_dir / "phase3a_pS_reconstructed_vs_stored.png"
    prior_plot = figure_dir / "phase3a_prior_effect_diagnostic.png"
    design_note = result_dir / "morphology_loglr_design_notes.md"

    summary.to_csv(summary_path, index=False)
    write_markdown(md_path, summary, result, acceptable)
    update_design_note(design_note, acceptable, summary)
    plot_reconstruction(recon_plot, result)
    if acceptable:
        plot_prior_effect(prior_plot, result)

    outputs = [summary_path, md_path, recon_plot]
    if acceptable:
        outputs.append(prior_plot)
    outputs.append(design_note)
    for path in outputs:
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    main()

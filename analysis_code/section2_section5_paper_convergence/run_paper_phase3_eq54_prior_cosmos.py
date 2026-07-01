"""Generate Section 2 Eq.54-prior-calibrated pS products for COSMOS.

This runner does not modify the stored collaborator v9 pS parquet. It treats
stored pS values as pre-prior/model scores for this controlled explicit-prior
task, applies the explicit S/G ratio from v9_ratio_data.csv, and writes new
derived outputs with an ``_eq54prior`` suffix.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from paper_eq54_prior import (
    BANDS,
    compute_eq54_prior_calibrated_ps,
    infer_prior_ratio_direction,
    load_v9_ps_table,
    load_v9_ratio_prior,
)
from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style


PS_V9_TABLE = Path("outputs/dp2_cosmos_ps_v9.parquet")
ANALYSIS_TABLE = Path("outputs/dp2_cosmos_analysis_table.parquet")
MATCHED_TABLE = Path("outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet")
RATIO_TABLE = Path("paper_convergence/tables/v9_ratio_data.csv")
FIT_TABLE = Path("paper_convergence/tables/v9_fit_parameters.csv")
EQ54_PS_TABLE = Path("outputs/dp2_cosmos_ps_v9_eq54prior.parquet")

FIGURE_DIR = Path("paper_convergence/figures/section2_bayesian_method")
RESULT_DIR = Path("paper_convergence/results/section2_bayesian_method")
DOC_DIR = Path("paper_convergence/docs")

PERFORMANCE_BINS = ((16.0, 24.0), (24.0, 25.0), (25.0, 26.0))
PERFORMANCE_BIN_LABELS = [f"{lo:g}-{hi:g}" for lo, hi in PERFORMANCE_BINS]
PERFORMANCE_METRICS = ("completeness", "contamination", "purity")
THRESHOLDS = np.linspace(0.0, 1.0, 101)
OPERATING_THRESHOLD = 0.5
RANDOM_SEED = 20260615


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def operating_metrics(
    y_true_star: pd.Series,
    score: pd.Series | None = None,
    threshold: float | None = None,
    pred_star: pd.Series | None = None,
) -> dict[str, float | int]:
    truth = pd.to_numeric(y_true_star, errors="coerce")
    if pred_star is None:
        if score is None or threshold is None:
            raise ValueError("score and threshold are required when pred_star is not supplied")
        values = pd.to_numeric(score, errors="coerce")
        valid = truth.isin([0, 1]) & np.isfinite(values)
        pred = values.ge(threshold)
    else:
        pred_series = pd.Series(pred_star, index=truth.index)
        valid = truth.isin([0, 1]) & pred_series.notna()
        pred = pred_series.astype(bool)

    truth = truth[valid].astype(int)
    pred = pred[valid].astype(bool)
    truth_star = truth.eq(1)
    truth_gal = truth.eq(0)

    star_tp = int((truth_star & pred).sum())
    star_fp = int((truth_gal & pred).sum())
    star_fn = int((truth_star & ~pred).sum())
    star_tn = int((truth_gal & ~pred).sum())

    gal_pred = ~pred
    gal_tp = int((truth_gal & gal_pred).sum())
    gal_fp = int((truth_star & gal_pred).sum())
    gal_fn = int((truth_gal & ~gal_pred).sum())
    gal_tn = int((truth_star & ~gal_pred).sum())

    return {
        "N_valid": int(valid.sum()),
        "N_star": int(truth_star.sum()),
        "N_galaxy": int(truth_gal.sum()),
        "star_TP": star_tp,
        "star_FP": star_fp,
        "star_TN": star_tn,
        "star_FN": star_fn,
        "star_completeness": _safe_divide(star_tp, star_tp + star_fn),
        "star_contamination": _safe_divide(star_fp, star_tp + star_fp),
        "star_purity": _safe_divide(star_tp, star_tp + star_fp),
        "galaxy_TP": gal_tp,
        "galaxy_FP": gal_fp,
        "galaxy_TN": gal_tn,
        "galaxy_FN": gal_fn,
        "galaxy_completeness": _safe_divide(gal_tp, gal_tp + gal_fn),
        "galaxy_contamination": _safe_divide(gal_fp, gal_tp + gal_fp),
        "galaxy_purity": _safe_divide(gal_tp, gal_tp + gal_fp),
    }


def threshold_curve_rows(df: pd.DataFrame, score_col: str, method: str, combination: str, mag_low: float, mag_high: float) -> list[dict]:
    rows: list[dict] = []
    rmag = pd.to_numeric(df["dp2_cmodel_mag_r"], errors="coerce")
    subset = df[rmag.ge(mag_low) & rmag.lt(mag_high)].copy()
    for threshold in THRESHOLDS:
        metrics = operating_metrics(subset["truth_binary"], score=subset[score_col], threshold=float(threshold))
        for positive_class in ("star", "galaxy"):
            prefix = "star" if positive_class == "star" else "galaxy"
            rows.append(
                {
                    "method": method,
                    "combination": combination,
                    "positive_class": positive_class,
                    "threshold": float(threshold),
                    "mag_low": mag_low,
                    "mag_high": mag_high,
                    "magnitude_bin": f"{mag_low:g} <= r < {mag_high:g}",
                    "N_valid": metrics["N_valid"],
                    "N_star": metrics["N_star"],
                    "N_galaxy": metrics["N_galaxy"],
                    "TP": metrics[f"{prefix}_TP"],
                    "FP": metrics[f"{prefix}_FP"],
                    "TN": metrics[f"{prefix}_TN"],
                    "FN": metrics[f"{prefix}_FN"],
                    "completeness": metrics[f"{prefix}_completeness"],
                    "contamination": metrics[f"{prefix}_contamination"],
                    "purity": metrics[f"{prefix}_purity"],
                    "score_column": score_col,
                    "score_definition": "Eq.54 explicit-prior calibrated pS; star-like if score >= threshold",
                }
            )
    return rows


def extendedness_rows(df: pd.DataFrame, mag_low: float, mag_high: float) -> list[dict]:
    rmag = pd.to_numeric(df["dp2_cmodel_mag_r"], errors="coerce")
    subset = df[rmag.ge(mag_low) & rmag.lt(mag_high)].copy()
    ext = pd.to_numeric(subset["dp2_extendedness_r"], errors="coerce")
    pred_star = ext.eq(0).astype(object)
    pred_star[~ext.isin([0, 1])] = np.nan
    metrics = operating_metrics(subset["truth_binary"], pred_star=pred_star)
    rows: list[dict] = []
    for positive_class in ("star", "galaxy"):
        prefix = "star" if positive_class == "star" else "galaxy"
        rows.append(
            {
                "method": "r_extendedness",
                "combination": "r_extendedness",
                "positive_class": positive_class,
                "threshold": np.nan,
                "mag_low": mag_low,
                "mag_high": mag_high,
                "magnitude_bin": f"{mag_low:g} <= r < {mag_high:g}",
                "N_valid": metrics["N_valid"],
                "N_star": metrics["N_star"],
                "N_galaxy": metrics["N_galaxy"],
                "TP": metrics[f"{prefix}_TP"],
                "FP": metrics[f"{prefix}_FP"],
                "TN": metrics[f"{prefix}_TN"],
                "FN": metrics[f"{prefix}_FN"],
                "completeness": metrics[f"{prefix}_completeness"],
                "contamination": metrics[f"{prefix}_contamination"],
                "purity": metrics[f"{prefix}_purity"],
                "score_column": "dp2_extendedness_r",
                "score_definition": "binary extendedness; 0=unresolved/star-like, 1=resolved/galaxy-like",
            }
        )
    return rows


def operating_point_table(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    return summary[summary["method"].eq(method) & np.isclose(summary["threshold"].fillna(-1), OPERATING_THRESHOLD)].copy()


def metric_axis_label(metric: str) -> str:
    return {"completeness": "completeness", "contamination": "contamination", "purity": "purity"}[metric]


def positive_class_title(positive_class: str) -> str:
    return "star / unresolved selected" if positive_class == "star" else "galaxy / resolved selected"


def create_eq54_ps_table(repo_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ps_path = repo_root / PS_V9_TABLE
    analysis_path = repo_root / ANALYSIS_TABLE
    ratio_path = repo_root / RATIO_TABLE
    out_path = repo_root / EQ54_PS_TABLE

    prior = load_v9_ratio_prior(ratio_path, field="COSMOS", band="r")

    if out_path.exists():
        pf = pq.ParquetFile(out_path)
        meta_rows = int(pf.metadata.num_rows)
        small = pd.read_parquet(
            out_path,
            columns=["object_id", "cmodel_mag_r", "log_prior_odds_eq54"],
        )
        rmag = pd.to_numeric(small["cmodel_mag_r"], errors="coerce")
        prior_assigned = np.isfinite(pd.to_numeric(small["log_prior_odds_eq54"], errors="coerce"))
        fig_cols = ["object_id", "pS_r_eq54prior", "pS_gri_eq54prior", "pS_ugrizy_eq54prior", "log_prior_odds_eq54"]
        fig_eq = pd.read_parquet(out_path, columns=fig_cols)
        info = {
            "input_ps_table": str(PS_V9_TABLE),
            "analysis_table": str(ANALYSIS_TABLE),
            "ratio_table": str(RATIO_TABLE),
            "fit_table": str(FIT_TABLE),
            "output_table": str(EQ54_PS_TABLE),
            "input_row_count": meta_rows,
            "output_row_count": meta_rows,
            "valid_rmag_count": int(np.isfinite(rmag).sum()),
            "prior_assigned_count": int(prior_assigned.sum()),
            "prior_column": prior.prior_column,
            "prior_direction": infer_prior_ratio_direction(prior),
            "prior_field": prior.field,
            "prior_band": prior.band,
            "prior_magnitude_definition": prior.magnitude_definition,
            "rmag_min": float(rmag.min(skipna=True)),
            "rmag_max": float(rmag.max(skipna=True)),
            "rows_outside_or_missing_prior": int((~prior_assigned).sum()),
            "output_reused": True,
        }
        return fig_eq, info

    ps = load_v9_ps_table(ps_path)
    analysis = _read_parquet_columns(analysis_path, ["object_id", "cmodel_mag_r"])
    merged = ps.merge(analysis, on="object_id", how="left", validate="one_to_one")
    calibrated = compute_eq54_prior_calibrated_ps(
        merged.drop(columns=["cmodel_mag_r"]),
        merged["cmodel_mag_r"],
        prior,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(out_path, index=False)

    rmag = pd.to_numeric(calibrated["cmodel_mag_r"], errors="coerce")
    prior_assigned = np.isfinite(pd.to_numeric(calibrated["log_prior_odds_eq54"], errors="coerce"))
    info = {
        "input_ps_table": str(PS_V9_TABLE),
        "analysis_table": str(ANALYSIS_TABLE),
        "ratio_table": str(RATIO_TABLE),
        "fit_table": str(FIT_TABLE),
        "output_table": str(EQ54_PS_TABLE),
        "input_row_count": int(len(ps)),
        "output_row_count": int(len(calibrated)),
        "valid_rmag_count": int(np.isfinite(rmag).sum()),
        "prior_assigned_count": int(prior_assigned.sum()),
        "prior_column": prior.prior_column,
        "prior_direction": infer_prior_ratio_direction(prior),
        "prior_field": prior.field,
        "prior_band": prior.band,
        "prior_magnitude_definition": prior.magnitude_definition,
        "rmag_min": float(rmag.min(skipna=True)),
        "rmag_max": float(rmag.max(skipna=True)),
        "rows_outside_or_missing_prior": int((~prior_assigned).sum()),
        "output_reused": False,
    }
    fig_cols = ["object_id", "pS_r_eq54prior", "pS_gri_eq54prior", "pS_ugrizy_eq54prior", "log_prior_odds_eq54"]
    return calibrated[fig_cols].copy(), info


def load_analysis_with_eq54(repo_root: Path, eq: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    analysis_cols = ["object_id", "cmodel_mag_r", "psf_minus_cmodel_r", "ug", "gr", "ri", "iz", "zy"]
    analysis = _read_parquet_columns(repo_root / ANALYSIS_TABLE, analysis_cols)
    keep = [
        "object_id",
        "pS_r_eq54prior",
        "pS_gri_eq54prior",
        "pS_ugrizy_eq54prior",
        "log_prior_odds_eq54",
    ]
    merged = analysis.merge(eq[keep], on="object_id", how="inner", validate="one_to_one")
    return merged, {
        "analysis_table": str(ANALYSIS_TABLE),
        "eq54_ps_table": str(EQ54_PS_TABLE),
        "join_key": "object_id",
        "r_magnitude_column": "cmodel_mag_r",
        "morphology_delta_column": "psf_minus_cmodel_r",
        "color_columns": "ug, gr, ri, iz, zy",
    }


def load_matched_with_eq54(repo_root: Path, eq: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    matched_cols = ["dp2_object_id", "object_id", "truth_binary", "truth_label", "dp2_cmodel_mag_r", "dp2_extendedness_r"]
    matched = _read_parquet_columns(repo_root / MATCHED_TABLE, matched_cols)
    keep = [
        "object_id",
        "pS_r_eq54prior",
        "pS_gri_eq54prior",
        "pS_ugrizy_eq54prior",
        "log_prior_odds_eq54",
    ]
    merged = matched.merge(eq[keep], left_on="dp2_object_id", right_on="object_id", how="inner", suffixes=("_external", "_dp2"), validate="many_to_one")
    return merged, {
        "matched_table": str(MATCHED_TABLE),
        "eq54_ps_table": str(EQ54_PS_TABLE),
        "join_key": "matched.dp2_object_id -> eq54_ps.object_id",
        "r_magnitude_column": "dp2_cmodel_mag_r",
        "truth_column": "truth_binary (1=star, 0=galaxy)",
        "truth_label_column": "truth_label",
        "extendedness_column": "dp2_extendedness_r (0=unresolved/star-like, 1=resolved/galaxy-like)",
    }


def _series_stats(path: Path, column: str) -> dict[str, object]:
    values = pd.to_numeric(pd.read_parquet(path, columns=[column])[column], errors="coerce")
    return {
        "finite": int(values.notna().sum()),
        "min": float(values.min(skipna=True)),
        "median": float(values.median(skipna=True)),
        "max": float(values.max(skipna=True)),
    }


def write_calibration_summary(repo_root: Path, info: dict[str, object]) -> list[Path]:
    rows: list[dict[str, object]] = []
    eq_path = repo_root / EQ54_PS_TABLE
    for band in BANDS:
        pre = _series_stats(eq_path, f"pS_{band}")
        post = _series_stats(eq_path, f"pS_{band}_eq54prior")
        rows.append(
            {
                **info,
                "score_name": f"pS_{band}",
                "pre_finite": pre["finite"],
                "post_finite": post["finite"],
                "pre_min": pre["min"],
                "pre_median": pre["median"],
                "pre_max": pre["max"],
                "post_min": post["min"],
                "post_median": post["median"],
                "post_max": post["max"],
            }
        )
    for name in ("r", "gri", "ugrizy"):
        post = _series_stats(eq_path, f"pS_{name}_eq54prior")
        rows.append(
            {
                **info,
                "score_name": f"pS_{name}_eq54prior",
                "pre_finite": np.nan,
                "post_finite": post["finite"],
                "pre_min": np.nan,
                "pre_median": np.nan,
                "pre_max": np.nan,
                "post_min": post["min"],
                "post_median": post["median"],
                "post_max": post["max"],
            }
        )

    csv_path = repo_root / RESULT_DIR / "eq54_prior_calibration_summary.csv"
    md_path = repo_root / RESULT_DIR / "eq54_prior_calibration_summary.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    lines = [
        "# Eq.54 Explicit-Prior Calibration Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- Pre-prior/model pS table: `{info['input_ps_table']}`",
        f"- r magnitude source: `{info['analysis_table']}`",
        f"- explicit prior table: `{info['ratio_table']}`",
        f"- fit parameter table, documentation only: `{info['fit_table']}`",
        "",
        "## Prior convention",
        f"- prior column: `{info['prior_column']}`",
        f"- prior direction: `{info['prior_direction']}`",
        "- `log10_NG_NS = log10(N_G / N_S)`",
        "- `log_prior_odds_eq54 = ln(N_S / N_G) = -ln(10) * log10_NG_NS`",
        "- The explicit prior is assigned by uncorrected r-band CModel magnitude.",
        "",
        "## Counts",
        f"- input rows: {info['input_row_count']:,}",
        f"- output rows: {info['output_row_count']:,}",
        f"- rows with valid rmag: {info['valid_rmag_count']:,}",
        f"- rows assigned finite prior: {info['prior_assigned_count']:,}",
        f"- rows outside/missing prior: {info['rows_outside_or_missing_prior']:,}",
        "",
        "## Formula",
        "- Single band: `logit(pS_b_eq54prior) = logit(pS_b_pre) + log_prior_odds_eq54`.",
        "- Multiband: model logits are summed first; `log_prior_odds_eq54` is applied once.",
        "- The fitted mixture weights in `v9_fit_parameters.csv` are not treated as the explicit prior in this task.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path]


def plot_fig2_4(repo_root: Path, full: pd.DataFrame, columns: dict[str, str]) -> list[Path]:
    set_paper_style()
    out_png = repo_root / FIGURE_DIR / "fig2_4_cosmos_r_pS_map_16_26_eq54prior.png"
    summary_path = repo_root / RESULT_DIR / "fig2_4_cosmos_r_pS_map_16_26_eq54prior_summary.csv"

    rmag = pd.to_numeric(full["cmodel_mag_r"], errors="coerce")
    delta = pd.to_numeric(full["psf_minus_cmodel_r"], errors="coerce")
    ps = pd.to_numeric(full["pS_r_eq54prior"], errors="coerce")
    valid = rmag.ge(16.0) & rmag.lt(26.0) & np.isfinite(delta) & np.isfinite(ps)
    used = full.loc[valid, ["object_id", "cmodel_mag_r", "psf_minus_cmodel_r", "pS_r_eq54prior"]].copy()
    plot_df = downsample_frame(used, 100_000, random_state=RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    scatter = ax.scatter(plot_df["cmodel_mag_r"], plot_df["psf_minus_cmodel_r"], c=plot_df["pS_r_eq54prior"], s=1.2, cmap="viridis", vmin=0, vmax=1, alpha=0.75, rasterized=True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("pS_r Eq.54 prior")
    ax.set_xlim(16, 26)
    finite_y = used["psf_minus_cmodel_r"].to_numpy(float)
    finite_y = finite_y[np.isfinite(finite_y)]
    ylo, yhi = np.nanpercentile(finite_y, [0.2, 99.8]) if finite_y.size else (-0.2, 1.2)
    ax.set_ylim(max(-0.4, ylo), min(2.2, yhi))
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("r PSF - CModel")
    ax.set_title("COSMOS r-band pS map, Eq.54 prior")
    saved = save_figure(fig, out_png, write_pdf=True)

    pd.DataFrame(
        [
            {
                "figure": "fig2_4_cosmos_r_pS_map_16_26_eq54prior",
                "N_analysis_rows": int(len(full)),
                "N_in_mag_range_and_finite": int(len(used)),
                "N_downsampled_for_plot": int(len(plot_df)),
                "score_column": "pS_r_eq54prior",
                "magnitude_column": columns["r_magnitude_column"],
                "morphology_delta_column": columns["morphology_delta_column"],
                "join_key": columns["join_key"],
            }
        ]
    ).to_csv(summary_path, index=False)
    return [*saved, summary_path]


def plot_performance_grid(summary: pd.DataFrame, out_png: Path, method: str, combinations: tuple[str, ...], labels: dict[str, str], colors: dict[str, str]) -> list[Path]:
    set_paper_style()
    op = operating_point_table(summary, method)
    x = np.arange(len(PERFORMANCE_BINS))
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.0), sharex=True, sharey="row")
    for col_idx, positive_class in enumerate(("star", "galaxy")):
        for row_idx, metric in enumerate(PERFORMANCE_METRICS):
            ax = axes[row_idx, col_idx]
            for combination in combinations:
                vals = []
                for mag_low, mag_high in PERFORMANCE_BINS:
                    row = op[op["positive_class"].eq(positive_class) & op["combination"].eq(combination) & op["mag_low"].eq(mag_low) & op["mag_high"].eq(mag_high)]
                    vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                ax.plot(x, vals, color=colors[combination], marker="o", ms=7, lw=2.2, label=labels[combination])
            ax.set_ylim(-0.04, 1.04)
            ax.set_ylabel(metric_axis_label(metric))
            ax.set_title(positive_class_title(positive_class) if row_idx == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels(PERFORMANCE_BIN_LABELS)
            ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
            if row_idx == 2:
                ax.set_xlabel("r CModel magnitude bin")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(combinations), bbox_to_anchor=(0.5, 0.005), frameon=True)
    fig.subplots_adjust(bottom=0.11, hspace=0.24, wspace=0.18)
    return save_figure(fig, out_png, write_pdf=True)


def plot_fig2_5(repo_root: Path, matched: pd.DataFrame) -> list[Path]:
    set_paper_style()
    rows: list[dict] = []
    for mag_low, mag_high in PERFORMANCE_BINS:
        rows.extend(threshold_curve_rows(matched, "pS_r_eq54prior", "eq54prior_pS", "r", mag_low, mag_high))
        rows.extend(extendedness_rows(matched, mag_low, mag_high))
    summary = pd.DataFrame(rows)
    summary_path = repo_root / RESULT_DIR / "fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    ps_op = operating_point_table(summary, "eq54prior_pS")
    ext_op = summary[summary["method"].eq("r_extendedness")].copy()
    x = np.arange(len(PERFORMANCE_BINS))
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.0), sharex=True, sharey="row")
    for col_idx, positive_class in enumerate(("star", "galaxy")):
        for row_idx, metric in enumerate(PERFORMANCE_METRICS):
            ax = axes[row_idx, col_idx]
            ps_vals = []
            ext_vals = []
            for mag_low, mag_high in PERFORMANCE_BINS:
                ps_row = ps_op[ps_op["positive_class"].eq(positive_class) & ps_op["mag_low"].eq(mag_low) & ps_op["mag_high"].eq(mag_high)]
                ext_row = ext_op[ext_op["positive_class"].eq(positive_class) & ext_op["mag_low"].eq(mag_low) & ext_op["mag_high"].eq(mag_high)]
                ps_vals.append(float(ps_row[metric].iloc[0]) if len(ps_row) else np.nan)
                ext_vals.append(float(ext_row[metric].iloc[0]) if len(ext_row) else np.nan)
            ax.plot(x, ps_vals, color="#222222", marker="o", ms=7, lw=2.2, label=f"pS_r Eq.54 prior at {OPERATING_THRESHOLD}")
            ax.plot(x, ext_vals, color="#ff7f0e", marker="s", ms=7, lw=2.2, label="r extendedness")
            ax.set_ylim(-0.04, 1.04)
            ax.set_ylabel(metric_axis_label(metric))
            ax.set_title(positive_class_title(positive_class) if row_idx == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels(PERFORMANCE_BIN_LABELS)
            ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
            if row_idx == 2:
                ax.set_xlabel("r CModel magnitude bin")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=True)
    fig.subplots_adjust(bottom=0.11, hspace=0.24, wspace=0.18)
    saved = save_figure(fig, repo_root / FIGURE_DIR / "fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior.png", write_pdf=True)
    return [*saved, summary_path]


def plot_fig2_6(repo_root: Path, matched: pd.DataFrame) -> list[Path]:
    rows: list[dict] = []
    for mag_low, mag_high in PERFORMANCE_BINS:
        for combination, score_col in (
            ("r", "pS_r_eq54prior"),
            ("gri", "pS_gri_eq54prior"),
            ("ugrizy", "pS_ugrizy_eq54prior"),
        ):
            rows.extend(threshold_curve_rows(matched, score_col, "eq54prior_multiband_pS", combination, mag_low, mag_high))
    summary = pd.DataFrame(rows)
    summary_path = repo_root / RESULT_DIR / "fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    saved = plot_performance_grid(
        summary,
        repo_root / FIGURE_DIR / "fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior.png",
        "eq54prior_multiband_pS",
        ("r", "gri", "ugrizy"),
        {"r": "r", "gri": "g+r+i", "ugrizy": "u+g+r+i+z+y"},
        {"r": "#222222", "gri": "#2ca02c", "ugrizy": "#9467bd"},
    )
    return [*saved, summary_path]


def plot_fig2_7(repo_root: Path, full: pd.DataFrame) -> list[Path]:
    set_paper_style()
    out_png = repo_root / FIGURE_DIR / "fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior.png"
    summary_path = repo_root / RESULT_DIR / "fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_summary.csv"
    color_planes = [("ug", "gr", "u-g", "g-r"), ("gr", "ri", "g-r", "r-i"), ("ri", "iz", "r-i", "i-z"), ("iz", "zy", "i-z", "z-y")]
    mag_bins = [(16.0, 25.0), (25.0, 26.0)]
    rmag = pd.to_numeric(full["cmodel_mag_r"], errors="coerce")
    score = pd.to_numeric(full["pS_ugrizy_eq54prior"], errors="coerce")
    method_star = score.ge(OPERATING_THRESHOLD)
    fig, axes = plt.subplots(4, 2, figsize=FIG_SIZES["4x2"], sharex=False, sharey=False)
    rows: list[dict] = []
    for row_idx, (xcol, ycol, xlabel, ylabel) in enumerate(color_planes):
        xlim, ylim = COLOR_COLOR_LIMITS[(xcol, ycol)]
        for col_idx, (mag_low, mag_high) in enumerate(mag_bins):
            ax = axes[row_idx, col_idx]
            finite = rmag.ge(mag_low) & rmag.lt(mag_high) & np.isfinite(score) & np.isfinite(pd.to_numeric(full[xcol], errors="coerce")) & np.isfinite(pd.to_numeric(full[ycol], errors="coerce"))
            panel = full.loc[finite, [xcol, ycol, "pS_ugrizy_eq54prior"]].copy()
            panel["method_star_like"] = method_star[finite].to_numpy()
            gal = panel[~panel["method_star_like"]]
            star = panel[panel["method_star_like"]]
            gal_plot = downsample_frame(gal, 45_000, random_state=RANDOM_SEED + row_idx * 10 + col_idx)
            star_plot = downsample_frame(star, 45_000, random_state=RANDOM_SEED + row_idx * 10 + col_idx + 1)
            ax.scatter(gal_plot[xcol], gal_plot[ycol], s=1.1, alpha=0.12, color=COLORS["galaxy"], rasterized=True)
            ax.scatter(star_plot[xcol], star_plot[ycol], s=1.1, alpha=0.22, color=COLORS["star"], rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel, labelpad=8)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_title(f"{mag_low:g} < r < {mag_high:g}\nN_unres={len(star):,}, N_res={len(gal):,}", pad=10)
            rows.append(
                {
                    "color_plane": f"{xlabel} vs {ylabel}",
                    "x_column": xcol,
                    "y_column": ycol,
                    "mag_low": mag_low,
                    "mag_high": mag_high,
                    "magnitude_column": "cmodel_mag_r",
                    "score_column": "pS_ugrizy_eq54prior",
                    "threshold": OPERATING_THRESHOLD,
                    "N_finite_panel": int(len(panel)),
                    "N_method_star_like": int(len(star)),
                    "N_method_galaxy_like": int(len(gal)),
                    "N_method_star_like_plotted": int(len(star_plot)),
                    "N_method_galaxy_like_plotted": int(len(gal_plot)),
                }
            )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="resolved"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="unresolved"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.012), frameon=True)
    fig.subplots_adjust(hspace=0.62, wspace=0.28, bottom=0.08)
    saved = save_figure(fig, out_png, write_pdf=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return [*saved, summary_path]


def write_docs(repo_root: Path, info: dict[str, object], generated: dict[str, list[Path]]) -> list[Path]:
    report = repo_root / DOC_DIR / "section2_explicit_prior_integration_report.md"
    checklist = repo_root / DOC_DIR / "section2_explicit_prior_integration_checklist.csv"
    status = repo_root / DOC_DIR / "section2_status_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    output_lines = []
    for group, paths in generated.items():
        output_lines.append(f"### {group}")
        output_lines.extend(f"- `{path.relative_to(repo_root)}`" for path in paths)
        output_lines.append("")

    report.write_text(
        "\n".join(
            [
                "# Section 2 Explicit-Prior Integration Report",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Definition used in this pass",
                "",
                "- Stored v9 `pS_*` values are treated as pre-prior/model-based scores for this controlled task.",
                "- The explicit prior is from `paper_convergence/tables/v9_ratio_data.csv`.",
                "- The prior column is `log10_NG_NS = log10(N_G/N_S)` for `field=COSMOS`, `band=r`.",
                "- Fitted mixture weights in `paper_convergence/tables/v9_fit_parameters.csv` are not treated as the explicit prior in this task.",
                "- Single-band calibrated pS uses `logit(pS_b_pre) + log_prior_odds_eq54`.",
                "- Multiband calibrated pS sums model logits first and applies `log_prior_odds_eq54` once.",
                "- Old finite-mean `pS_gri` / `pS_ugrizy` figures remain draft diagnostics, not final Bayesian multiband posterior results.",
                "",
                "## Summary",
                f"- input rows: {info['input_row_count']:,}",
                f"- rows assigned finite prior: {info['prior_assigned_count']:,}",
                f"- rows outside/missing prior: {info['rows_outside_or_missing_prior']:,}",
                f"- derived output table: `{info['output_table']}`",
                "",
                "## Generated outputs",
                "",
                *output_lines,
            ]
        )
        + "\n"
    )

    checklist_rows = [
        {
            "task_id": "eq54-1",
            "task_name": "Create Eq.54 prior-calibrated pS parquet",
            "status": "DONE",
            "input_files": f"{PS_V9_TABLE}; {ANALYSIS_TABLE}; {RATIO_TABLE}",
            "output_files": str(EQ54_PS_TABLE),
            "formula_used": "logit(pS_eq54prior)=logit(pS_pre)+ln(NS/NG)",
            "prior_column": "log10_NG_NS",
            "prior_direction": "NG_over_NS",
            "caveat": "Stored pS is treated as pre-prior/model-based for this controlled task.",
            "next_action": "Review with collaborator before replacing old draft figures.",
        },
        {
            "task_id": "eq54-2",
            "task_name": "Generate Eq.54 Section 2.4-2.7 figures",
            "status": "DONE",
            "input_files": str(EQ54_PS_TABLE),
            "output_files": "; ".join(str(p.relative_to(repo_root)) for group in ("fig2_4", "fig2_5", "fig2_6", "fig2_7") for p in generated[group]),
            "formula_used": "single-band prior once; multiband summed logits plus prior once",
            "prior_column": "log10_NG_NS",
            "prior_direction": "NG_over_NS",
            "caveat": "Old non-prior-calibrated draft figures were not overwritten.",
            "next_action": "Visually inspect and decide which figures replace drafts.",
        },
    ]
    pd.DataFrame(checklist_rows).to_csv(checklist, index=False)

    status.write_text(
        "\n".join(
            [
                "# Section 2 Status Report",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "Fig 2.4-2.7 now have new Eq.54-explicit-prior candidate versions with `_eq54prior` filenames.",
                "The original stored-v9-pS draft figures remain available as diagnostics and were not overwritten.",
                "The Eq.54 pass treats stored v9 pS as pre-prior/model scores by task definition.",
                "The explicit prior is from `paper_convergence/tables/v9_ratio_data.csv`, using COSMOS r-band `log10_NG_NS`.",
                "Section 1 was not touched.",
            ]
        )
        + "\n"
    )
    return [report, checklist, status]


def run_all(repo_root: Path | None = None) -> dict[str, list[Path]]:
    repo_root = repo_root or repo_root_from_file()
    (repo_root / FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    (repo_root / RESULT_DIR).mkdir(parents=True, exist_ok=True)

    eq, info = create_eq54_ps_table(repo_root)
    generated: dict[str, list[Path]] = {}
    generated["calibration_summary"] = write_calibration_summary(repo_root, info)

    full, full_cols = load_analysis_with_eq54(repo_root, eq)
    matched, _matched_cols = load_matched_with_eq54(repo_root, eq)
    generated["fig2_4"] = plot_fig2_4(repo_root, full, full_cols)
    generated["fig2_5"] = plot_fig2_5(repo_root, matched)
    generated["fig2_6"] = plot_fig2_6(repo_root, matched)
    generated["fig2_7"] = plot_fig2_7(repo_root, full)
    generated["docs"] = write_docs(repo_root, info, generated)
    return generated


def main() -> None:
    repo_root = repo_root_from_file()
    generated = run_all(repo_root)
    for group, paths in generated.items():
        print(f"[{group}]")
        for path in paths:
            print(f"  {path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

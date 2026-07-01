"""Generate Section 2 draft figures from stored COSMOS v9 pS outputs.

This runner intentionally uses the existing object-level v9 pS table as the
current method output. It does not reconstruct pS from fit parameters, does not
apply an additional magnitude prior, and does not write object-level products.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style


ANALYSIS_TABLE = Path("outputs/dp2_cosmos_analysis_table.parquet")
MATCHED_TABLE = Path("outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet")
PS_V9_TABLE = Path("outputs/dp2_cosmos_ps_v9.parquet")

FIGURE_DIR = Path("paper_convergence/figures/section2_bayesian_method")
RESULT_DIR = Path("paper_convergence/results/section2_bayesian_method")

BANDS = ("u", "g", "r", "i", "z", "y")
COMBINATIONS = {
    "r": ("r",),
    "gri": ("g", "r", "i"),
    "ugrizy": ("u", "g", "r", "i", "z", "y"),
}
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


def _first_existing(columns: set[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"none of these columns exist: {candidates}")


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def finite_mean_score(df: pd.DataFrame, bands: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    cols = [f"pS_{band}" for band in bands]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    n_used = values.notna().sum(axis=1)
    score = values.mean(axis=1, skipna=True)
    score[n_used.eq(0)] = np.nan
    return score, n_used


def add_combination_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, bands in COMBINATIONS.items():
        score, n_used = finite_mean_score(out, bands)
        out[f"pS_{name}"] = score
        out[f"n_bands_{name}"] = n_used
    return out


def load_v9_ps(repo_root: Path) -> pd.DataFrame:
    cols = ["object_id", *[f"pS_{band}" for band in BANDS], "ps_version", "field"]
    ps = _read_parquet_columns(repo_root / PS_V9_TABLE, cols)
    return ps


def load_analysis_with_v9_ps(repo_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    analysis_path = repo_root / ANALYSIS_TABLE
    ps_path = repo_root / PS_V9_TABLE
    analysis_cols = [
        "object_id",
        "cmodel_mag_r",
        "psf_minus_cmodel_r",
        "extendedness_r",
        "ug",
        "gr",
        "ri",
        "iz",
        "zy",
    ]
    analysis = _read_parquet_columns(analysis_path, analysis_cols)
    ps = load_v9_ps(repo_root)
    merged = analysis.merge(ps, on="object_id", how="inner", validate="one_to_one")
    merged = add_combination_scores(merged)
    columns = {
        "analysis_table": str(ANALYSIS_TABLE),
        "ps_table": str(PS_V9_TABLE),
        "join_key": "object_id",
        "r_magnitude_column": "cmodel_mag_r",
        "morphology_delta_column": "psf_minus_cmodel_r",
        "extendedness_column": "extendedness_r",
        "color_columns": "ug, gr, ri, iz, zy",
    }
    return merged, columns


def load_matched_with_v9_ps(repo_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    matched_path = repo_root / MATCHED_TABLE
    ps_path = repo_root / PS_V9_TABLE
    matched_cols = [
        "dp2_object_id",
        "object_id",
        "truth_binary",
        "truth_label",
        "dp2_cmodel_mag_r",
        "dp2_extendedness_r",
    ]
    matched = _read_parquet_columns(matched_path, matched_cols)
    ps = load_v9_ps(repo_root)
    merged = matched.merge(
        ps,
        left_on="dp2_object_id",
        right_on="object_id",
        how="inner",
        suffixes=("_external", "_dp2"),
        validate="many_to_one",
    )
    merged = add_combination_scores(merged)
    columns = {
        "matched_table": str(MATCHED_TABLE),
        "ps_table": str(PS_V9_TABLE),
        "join_key": "matched.dp2_object_id -> v9_ps.object_id",
        "r_magnitude_column": "dp2_cmodel_mag_r",
        "truth_column": "truth_binary (1=star, 0=galaxy)",
        "truth_label_column": "truth_label",
        "extendedness_column": "dp2_extendedness_r (0=unresolved/star-like, 1=resolved/galaxy-like)",
    }
    return merged, columns


def operating_metrics(y_true_star: pd.Series, score: pd.Series | None = None, threshold: float | None = None, pred_star: pd.Series | None = None) -> dict[str, float | int]:
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
    gal_truth = truth_gal
    gal_tp = int((gal_truth & gal_pred).sum())
    gal_fp = int((truth_star & gal_pred).sum())
    gal_fn = int((gal_truth & ~gal_pred).sum())
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
    subset = df[
        pd.to_numeric(df["dp2_cmodel_mag_r"], errors="coerce").ge(mag_low)
        & pd.to_numeric(df["dp2_cmodel_mag_r"], errors="coerce").lt(mag_high)
    ].copy()
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
                    "score_definition": "stored v9 pS threshold; star-like if score >= threshold",
                }
            )
    return rows


def extendedness_rows(df: pd.DataFrame, mag_low: float, mag_high: float) -> list[dict]:
    subset = df[
        pd.to_numeric(df["dp2_cmodel_mag_r"], errors="coerce").ge(mag_low)
        & pd.to_numeric(df["dp2_cmodel_mag_r"], errors="coerce").lt(mag_high)
    ].copy()
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


def operating_point_table(summary: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Return operating-point rows for plotting summary performance panels."""

    if method is None:
        return summary[summary["threshold"].isna()].copy()
    return summary[summary["method"].eq(method) & np.isclose(summary["threshold"].fillna(-1), OPERATING_THRESHOLD)].copy()


def metric_axis_label(metric: str) -> str:
    labels = {
        "completeness": "completeness",
        "contamination": "contamination",
        "purity": "purity",
    }
    return labels[metric]


def positive_class_title(positive_class: str) -> str:
    if positive_class == "star":
        return "star / unresolved selected"
    if positive_class == "galaxy":
        return "galaxy / resolved selected"
    return positive_class


def plot_fig2_4(repo_root: Path, full: pd.DataFrame, columns: dict[str, str]) -> list[Path]:
    set_paper_style()
    out_png = repo_root / FIGURE_DIR / "fig2_4_cosmos_r_pS_map_v9.png"
    summary_path = repo_root / RESULT_DIR / "fig2_4_cosmos_r_pS_map_v9_summary.csv"
    notes_path = repo_root / RESULT_DIR / "fig2_4_cosmos_r_pS_map_v9_notes.md"

    rmag = pd.to_numeric(full["cmodel_mag_r"], errors="coerce")
    delta = pd.to_numeric(full["psf_minus_cmodel_r"], errors="coerce")
    ps = pd.to_numeric(full["pS_r"], errors="coerce")
    valid = rmag.ge(16.0) & rmag.lt(26.0) & np.isfinite(delta) & np.isfinite(ps)
    used = full.loc[valid, ["object_id", "cmodel_mag_r", "psf_minus_cmodel_r", "pS_r"]].copy()
    plot_df = downsample_frame(used, 250_000, random_state=RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    scatter = ax.scatter(
        plot_df["cmodel_mag_r"],
        plot_df["psf_minus_cmodel_r"],
        c=plot_df["pS_r"],
        s=1.2,
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.75,
        rasterized=True,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("pS_r")
    ax.set_xlim(16, 26)
    yvals = used["psf_minus_cmodel_r"].to_numpy(float)
    finite_y = yvals[np.isfinite(yvals)]
    ylo, yhi = np.nanpercentile(finite_y, [0.2, 99.8]) if finite_y.size else (-0.2, 1.2)
    ax.set_ylim(max(-0.4, ylo), min(2.2, yhi))
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("r PSF - CModel")
    ax.set_title("COSMOS v9 r-band pS map")
    saved = save_figure(fig, out_png, write_pdf=True)

    summary = pd.DataFrame(
        [
            {
                "figure": "fig2_4_cosmos_r_pS_map_v9",
                "N_analysis_rows": int(len(full)),
                "N_in_mag_range_and_finite": int(len(used)),
                "N_downsampled_for_plot": int(len(plot_df)),
                "pS_finite_fraction_in_mag_range": float(np.isfinite(ps[rmag.ge(16.0) & rmag.lt(26.0)]).mean()),
                "mag_low": 16.0,
                "mag_high": 26.0,
                "magnitude_column": columns["r_magnitude_column"],
                "morphology_delta_column": columns["morphology_delta_column"],
                "score_column": "pS_r",
                "join_key": columns["join_key"],
            }
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    notes_path.write_text(
        "\n".join(
            [
                "# Fig 2.4 COSMOS v9 r-band pS map",
                "",
                "This draft figure uses the stored v9 pS output as the current method output.",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Inputs",
                f"- Full COSMOS analysis table: `{columns['analysis_table']}`",
                f"- Stored v9 pS table: `{columns['ps_table']}`",
                f"- Join key: `{columns['join_key']}`",
                "",
                "## Columns",
                f"- r magnitude: `{columns['r_magnitude_column']}`",
                f"- morphology delta: `{columns['morphology_delta_column']}`",
                "- score: `pS_r` from `outputs/dp2_cosmos_ps_v9.parquet`",
                "",
                "No pS values were reconstructed or prior-corrected in this pass.",
            ]
        )
        + "\n"
    )
    return [*saved, summary_path, notes_path]


def plot_fig2_5(repo_root: Path, matched: pd.DataFrame, columns: dict[str, str]) -> list[Path]:
    set_paper_style()
    out_png = repo_root / FIGURE_DIR / "fig2_5_cosmos_r_pS_vs_extendedness_performance_v9.png"
    summary_path = repo_root / RESULT_DIR / "fig2_5_cosmos_r_pS_vs_extendedness_performance_v9_summary.csv"
    notes_path = repo_root / RESULT_DIR / "fig2_5_cosmos_r_pS_vs_extendedness_performance_v9_notes.md"

    rows: list[dict] = []
    for mag_low, mag_high in PERFORMANCE_BINS:
        rows.extend(threshold_curve_rows(matched, "pS_r", "stored_v9_pS", "r", mag_low, mag_high))
        rows.extend(extendedness_rows(matched, mag_low, mag_high))
    summary = pd.DataFrame(rows)

    ps_op = operating_point_table(summary, "stored_v9_pS")
    ext_op = summary[summary["method"].eq("r_extendedness")].copy()
    x = np.arange(len(PERFORMANCE_BINS))

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.0), sharex=True, sharey="row")
    for col_idx, positive_class in enumerate(("star", "galaxy")):
        for row_idx, metric in enumerate(PERFORMANCE_METRICS):
            ax = axes[row_idx, col_idx]
            pS_vals = []
            ext_vals = []
            for mag_low, mag_high in PERFORMANCE_BINS:
                pS_row = ps_op[
                    ps_op["positive_class"].eq(positive_class)
                    & ps_op["mag_low"].eq(mag_low)
                    & ps_op["mag_high"].eq(mag_high)
                ]
                ext_row = ext_op[
                    ext_op["positive_class"].eq(positive_class)
                    & ext_op["mag_low"].eq(mag_low)
                    & ext_op["mag_high"].eq(mag_high)
                ]
                pS_vals.append(float(pS_row[metric].iloc[0]) if len(pS_row) else np.nan)
                ext_vals.append(float(ext_row[metric].iloc[0]) if len(ext_row) else np.nan)
            ax.plot(x, pS_vals, color="#222222", marker="o", ms=7, lw=2.2, label=f"pS_r at {OPERATING_THRESHOLD}")
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
    saved = save_figure(fig, out_png, write_pdf=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    notes_path.write_text(
        "\n".join(
            [
                "# Fig 2.5 COSMOS v9 r-band pS vs r-band extendedness performance",
                "",
                "This figure compares stored v9 `pS_r` performance at threshold 0.5 with the binary r-band extendedness baseline.",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Inputs and columns",
                f"- Matched validation table: `{columns['matched_table']}`",
                f"- Stored v9 pS table: `{columns['ps_table']}`",
                f"- Join key: `{columns['join_key']}`",
                f"- r magnitude: `{columns['r_magnitude_column']}`",
                f"- truth labels: `{columns['truth_column']}`",
                f"- extendedness: `{columns['extendedness_column']}`",
                "",
                "## Metric definitions",
                "- The figure uses a 2-column x 3-row layout: columns are star/unresolved-selected and galaxy/resolved-selected samples; rows are completeness, contamination, and purity.",
                "- For star-selected pS metrics, predicted star means `pS_r >= threshold`.",
                "- For galaxy-selected pS metrics, predicted galaxy means `pS_r < threshold`.",
                "- For extendedness, predicted star means `dp2_extendedness_r == 0`; predicted galaxy means `dp2_extendedness_r == 1`.",
                "- Completeness = TP / (TP + FN).",
                "- Contamination = FP / (TP + FP).",
                "- Purity = TP / (TP + FP).",
                "",
                f"The plotted pS operating threshold is {OPERATING_THRESHOLD}. The CSV also retains the full threshold grid.",
                "No pS values were reconstructed or prior-corrected in this pass.",
            ]
        )
        + "\n"
    )
    return [*saved, summary_path, notes_path]


def plot_fig2_6(repo_root: Path, matched: pd.DataFrame, columns: dict[str, str]) -> list[Path]:
    set_paper_style()
    out_png = repo_root / FIGURE_DIR / "fig2_6_cosmos_multiband_pS_performance_v9.png"
    summary_path = repo_root / RESULT_DIR / "fig2_6_cosmos_multiband_pS_performance_v9_summary.csv"
    notes_path = repo_root / RESULT_DIR / "fig2_6_cosmos_multiband_pS_performance_v9_notes.md"

    rows: list[dict] = []
    for mag_low, mag_high in PERFORMANCE_BINS:
        for combination in ("r", "gri", "ugrizy"):
            rows.extend(
                threshold_curve_rows(
                    matched,
                    f"pS_{combination}",
                    "stored_v9_pS_finite_mean",
                    combination,
                    mag_low,
                    mag_high,
                )
            )
    summary = pd.DataFrame(rows)
    for combination in ("r", "gri", "ugrizy"):
        n_col = f"n_bands_{combination}"
        if n_col in matched.columns:
            for mag_low, mag_high in PERFORMANCE_BINS:
                mask = (
                    pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce").ge(mag_low)
                    & pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce").lt(mag_high)
                    & np.isfinite(pd.to_numeric(matched[f"pS_{combination}"], errors="coerce"))
                )
                nvals = pd.to_numeric(matched.loc[mask, n_col], errors="coerce")
                row_mask = summary["combination"].eq(combination) & summary["mag_low"].eq(mag_low) & summary["mag_high"].eq(mag_high)
                summary.loc[row_mask, "number_of_bands_used_min"] = float(nvals.min()) if len(nvals) else np.nan
                summary.loc[row_mask, "number_of_bands_used_median"] = float(nvals.median()) if len(nvals) else np.nan
                summary.loc[row_mask, "number_of_bands_used_max"] = float(nvals.max()) if len(nvals) else np.nan

    colors = {"r": "#222222", "gri": "#2ca02c", "ugrizy": "#9467bd"}
    labels = {"r": "r", "gri": "g+r+i", "ugrizy": "u+g+r+i+z+y"}
    op = summary[summary["method"].eq("stored_v9_pS_finite_mean") & np.isclose(summary["threshold"], OPERATING_THRESHOLD)].copy()
    x = np.arange(len(PERFORMANCE_BINS))
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.0), sharex=True, sharey="row")
    for col_idx, positive_class in enumerate(("star", "galaxy")):
        for row_idx, metric in enumerate(PERFORMANCE_METRICS):
            ax = axes[row_idx, col_idx]
            for combination in ("r", "gri", "ugrizy"):
                vals = []
                for mag_low, mag_high in PERFORMANCE_BINS:
                    row = op[
                        op["positive_class"].eq(positive_class)
                        & op["combination"].eq(combination)
                        & op["mag_low"].eq(mag_low)
                        & op["mag_high"].eq(mag_high)
                    ]
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
    handles, labels_for_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_for_legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), frameon=True)
    fig.subplots_adjust(bottom=0.11, hspace=0.24, wspace=0.18)
    saved = save_figure(fig, out_png, write_pdf=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    notes_path.write_text(
        "\n".join(
            [
                "# Fig 2.6 COSMOS v9 multiband pS performance",
                "",
                "This figure compares stored v9 pS band combinations at threshold 0.5 using completeness, contamination, and purity.",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Inputs and columns",
                f"- Matched validation table: `{columns['matched_table']}`",
                f"- Stored v9 pS table: `{columns['ps_table']}`",
                f"- Join key: `{columns['join_key']}`",
                f"- r magnitude: `{columns['r_magnitude_column']}`",
                f"- truth labels: `{columns['truth_column']}`",
                "",
                "## Band-combination definitions",
                "- `pS_r = pS_r` from the stored v9 table.",
                "- `pS_gri = mean(pS_g, pS_r, pS_i)` over finite values.",
                "- `pS_ugrizy = mean(pS_u, pS_g, pS_r, pS_i, pS_z, pS_y)` over finite values.",
                "",
                "## Metric definitions",
                "- The figure uses the same 2-column x 3-row layout as Fig 2.5: columns are star/unresolved-selected and galaxy/resolved-selected samples; rows are completeness, contamination, and purity.",
                "- Completeness = TP / (TP + FN).",
                "- Contamination = FP / (TP + FP).",
                "- Purity = TP / (TP + FP).",
                f"- The plotted pS operating threshold is {OPERATING_THRESHOLD}. The CSV also retains the full threshold grid.",
                "",
                "No pS values were reconstructed or prior-corrected in this pass.",
            ]
        )
        + "\n"
    )
    return [*saved, summary_path, notes_path]


def plot_fig2_7(repo_root: Path, full: pd.DataFrame, columns: dict[str, str]) -> list[Path]:
    set_paper_style()
    out_png = repo_root / FIGURE_DIR / "fig2_7_cosmos_ugrizy_method_color_color_v9.png"
    summary_path = repo_root / RESULT_DIR / "fig2_7_cosmos_ugrizy_method_color_color_v9_summary.csv"
    notes_path = repo_root / RESULT_DIR / "fig2_7_cosmos_ugrizy_method_color_color_v9_notes.md"

    color_planes = [
        ("ug", "gr", "u-g", "g-r"),
        ("gr", "ri", "g-r", "r-i"),
        ("ri", "iz", "r-i", "i-z"),
        ("iz", "zy", "i-z", "z-y"),
    ]
    mag_bins = [(16.0, 25.0), (25.0, 26.0)]
    rmag = pd.to_numeric(full["cmodel_mag_r"], errors="coerce")
    score = pd.to_numeric(full["pS_ugrizy"], errors="coerce")
    method_star = score.ge(OPERATING_THRESHOLD)

    fig, axes = plt.subplots(4, 2, figsize=FIG_SIZES["4x2"], sharex=False, sharey=False)
    rows: list[dict] = []
    for row_idx, (xcol, ycol, xlabel, ylabel) in enumerate(color_planes):
        xlim, ylim = COLOR_COLOR_LIMITS[(xcol, ycol)]
        for col_idx, (mag_low, mag_high) in enumerate(mag_bins):
            ax = axes[row_idx, col_idx]
            finite = (
                rmag.ge(mag_low)
                & rmag.lt(mag_high)
                & np.isfinite(score)
                & np.isfinite(pd.to_numeric(full[xcol], errors="coerce"))
                & np.isfinite(pd.to_numeric(full[ycol], errors="coerce"))
            )
            panel = full.loc[finite, [xcol, ycol, "pS_ugrizy"]].copy()
            panel["method_star_like"] = method_star[finite].to_numpy()
            gal = panel[~panel["method_star_like"]]
            star = panel[panel["method_star_like"]]
            gal_plot = downsample_frame(gal, 45_000, random_state=RANDOM_SEED + row_idx * 10 + col_idx)
            star_plot = downsample_frame(star, 45_000, random_state=RANDOM_SEED + row_idx * 10 + col_idx + 1)
            ax.scatter(
                gal_plot[xcol],
                gal_plot[ycol],
                s=1.1,
                alpha=0.12,
                color=COLORS["galaxy"],
                rasterized=True,
            )
            ax.scatter(
                star_plot[xcol],
                star_plot[ycol],
                s=1.1,
                alpha=0.22,
                color=COLORS["star"],
                rasterized=True,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel, labelpad=8)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_title(
                f"{mag_low:g} < r < {mag_high:g}\n"
                f"N_unres={len(star):,}, N_res={len(gal):,}",
                pad=10,
            )
            rows.append(
                {
                    "color_plane": f"{xlabel} vs {ylabel}",
                    "x_column": xcol,
                    "y_column": ycol,
                    "mag_low": mag_low,
                    "mag_high": mag_high,
                    "magnitude_column": "cmodel_mag_r",
                    "score_column": "pS_ugrizy",
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

    summary = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    notes_path.write_text(
        "\n".join(
            [
                "# Fig 2.7 COSMOS v9 ugrizy method-label color-color plot",
                "",
                "This draft figure uses labels from the stored v9 pS ugrizy finite-mean score.",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Inputs and columns",
                f"- Full COSMOS analysis table: `{columns['analysis_table']}`",
                f"- Stored v9 pS table: `{columns['ps_table']}`",
                f"- Join key: `{columns['join_key']}`",
                f"- r magnitude: `{columns['r_magnitude_column']}`",
                f"- colors: `{columns['color_columns']}`",
                "",
                "## Method-label definition",
                "- `pS_ugrizy = mean(pS_u, pS_g, pS_r, pS_i, pS_z, pS_y)` over finite values.",
                f"- unresolved if `pS_ugrizy >= {OPERATING_THRESHOLD}`.",
                f"- resolved if `pS_ugrizy < {OPERATING_THRESHOLD}`.",
                "",
                "No pS values were reconstructed or prior-corrected in this pass.",
            ]
        )
        + "\n"
    )
    return [*saved, summary_path, notes_path]


def write_run_summary(repo_root: Path, generated: list[Path], full: pd.DataFrame, matched: pd.DataFrame, full_cols: dict[str, str], matched_cols: dict[str, str]) -> Path:
    path = repo_root / RESULT_DIR / "section2_v9_ps_figures_run_summary.md"
    full_rmag = pd.to_numeric(full["cmodel_mag_r"], errors="coerce")
    matched_rmag = pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce")
    n_full_in_mag_range = int((full_rmag.ge(16) & full_rmag.lt(26)).sum())
    n_matched_in_mag_range = int((matched_rmag.ge(16) & matched_rmag.lt(26)).sum())
    lines = [
        "# Section 2 v9 pS figure run summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "Generated draft Section 2.4-2.7 figures from the stored COSMOS v9 pS table.",
        "No v9 pS parquet files were modified, no object-level outputs were written, and no prior correction was applied to stored pS.",
        "",
        "## Data provenance",
        f"- Full analysis table: `{full_cols['analysis_table']}`",
        f"- Matched validation table: `{matched_cols['matched_table']}`",
        f"- Stored v9 pS table: `{full_cols['ps_table']}`",
        f"- Full-table join: `{full_cols['join_key']}`",
        f"- Matched-table join: `{matched_cols['join_key']}`",
        "",
        "## Counts",
        f"- Full analysis rows joined to v9 pS: {len(full):,}",
        f"- Full analysis rows with 16 <= r < 26: {n_full_in_mag_range:,}",
        f"- Matched validation rows joined to v9 pS: {len(matched):,}",
        f"- Matched validation rows with 16 <= r < 26: {n_matched_in_mag_range:,}",
        "",
        "## pS combination definitions",
        "- `pS_r = pS_r` from the stored v9 table.",
        "- `pS_gri = mean(pS_g, pS_r, pS_i)` over finite values.",
        "- `pS_ugrizy = mean(pS_u, pS_g, pS_r, pS_i, pS_z, pS_y)` over finite values.",
        "",
        "## Generated files",
    ]
    lines.extend(f"- `{item.relative_to(repo_root)}`" for item in generated)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def run_all(repo_root: Path | None = None) -> dict[str, list[Path]]:
    repo_root = repo_root or repo_root_from_file()
    FIGURE_DIR_ABS = repo_root / FIGURE_DIR
    RESULT_DIR_ABS = repo_root / RESULT_DIR
    FIGURE_DIR_ABS.mkdir(parents=True, exist_ok=True)
    RESULT_DIR_ABS.mkdir(parents=True, exist_ok=True)

    full, full_cols = load_analysis_with_v9_ps(repo_root)
    matched, matched_cols = load_matched_with_v9_ps(repo_root)

    generated: dict[str, list[Path]] = {}
    generated["fig2_4"] = plot_fig2_4(repo_root, full, full_cols)
    generated["fig2_5"] = plot_fig2_5(repo_root, matched, matched_cols)
    generated["fig2_6"] = plot_fig2_6(repo_root, matched, matched_cols)
    generated["fig2_7"] = plot_fig2_7(repo_root, full, full_cols)
    all_paths = [path for paths in generated.values() for path in paths]
    run_summary = write_run_summary(repo_root, all_paths, full, matched, full_cols, matched_cols)
    generated["run_summary"] = [run_summary]
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

"""Build first-pass Eq.54 + color-integrated COSMOS pS diagnostics.

This runner treats the existing color-only Random Forest probability as an
additional likelihood-ratio factor:

    logLR_total = logLR_morphology + logit(pS_color)
    logit(pS_posterior) = logLR_total + log_prior_odds_eq54

The explicit magnitude prior is applied once. Existing Eq.54 and pS_color
parquet files are read-only inputs; this script writes a new derived parquet
and new figures with an ``_eq54prior_color`` suffix.
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

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style


EQ54_INPUT = Path("outputs/dp2_cosmos_ps_v9_eq54prior.parquet")
COLOR_INPUT = Path("outputs/dp2_cosmos_ps_color_v1.parquet")
MATCHED_INPUT = Path("outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet")
ANALYSIS_INPUT = Path("outputs/dp2_cosmos_analysis_table.parquet")
OUTPUT_PARQUET = Path("outputs/dp2_cosmos_ps_v9_eq54prior_color.parquet")

FIGURE_DIR = Path("paper_convergence/figures/section2_bayesian_method")
FIGURE5_DIR = Path("paper_convergence/figures/section5_discussion")
RESULT_DIR = Path("paper_convergence/results/section2_bayesian_method")
DOC_DIR = Path("paper_convergence/docs")

PERFORMANCE_BINS = ((16.0, 24.0), (24.0, 25.0), (25.0, 26.0))
PERFORMANCE_BIN_LABELS = [f"{lo:g}-{hi:g}" for lo, hi in PERFORMANCE_BINS]
OPERATING_THRESHOLD = 0.5
EPS = 1e-6
RANDOM_SEED = 20260626


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def sigmoid(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    pos = valid & (arr >= 0)
    neg = valid & (arr < 0)
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def logit_probability(p: pd.Series | np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    clipped = np.clip(arr[valid], eps, 1.0 - eps)
    out[valid] = np.log(clipped / (1.0 - clipped))
    return out


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def compute_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.isin([0, 1]) & np.isfinite(s)
    y = y[valid].astype(int).to_numpy()
    s = s[valid].to_numpy(float)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    change_idx = np.r_[np.where(s[1:] != s[:-1])[0], len(s) - 1]
    tps = np.cumsum(y == 1)[change_idx]
    fps = np.cumsum(y == 0)[change_idx]
    tpr = np.r_[0.0, tps / (y == 1).sum()]
    fpr = np.r_[0.0, fps / (y == 0).sum()]
    return float(np.trapezoid(tpr, fpr))


def operating_metrics(y_true: pd.Series, score: pd.Series | None = None, pred_star: pd.Series | None = None) -> dict[str, float | int]:
    truth = pd.to_numeric(y_true, errors="coerce")
    if pred_star is None:
        values = pd.to_numeric(score, errors="coerce")
        valid = truth.isin([0, 1]) & np.isfinite(values)
        pred = values[valid].ge(OPERATING_THRESHOLD)
    else:
        pred_series = pd.Series(pred_star, index=truth.index)
        valid = truth.isin([0, 1]) & pred_series.notna()
        pred = pred_series[valid].astype(bool)
    truth = truth[valid].astype(int)
    truth_star = truth.eq(1)
    truth_gal = truth.eq(0)
    tp = int((truth_star & pred).sum())
    fp = int((truth_gal & pred).sum())
    fn = int((truth_star & ~pred).sum())
    tn = int((truth_gal & ~pred).sum())
    return {
        "N_valid": int(valid.sum()),
        "N_star": int(truth_star.sum()),
        "N_galaxy": int(truth_gal.sum()),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "completeness": safe_divide(tp, tp + fn),
        "contamination": safe_divide(fp, tp + fp),
        "purity": safe_divide(tp, tp + fp),
    }


def read_inputs(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eq_cols = [
        "object_id",
        "ra",
        "dec",
        "cmodel_mag_r",
        "log_prior_odds_eq54",
        "logLR_u_model",
        "logLR_g_model",
        "logLR_r_model",
        "logLR_i_model",
        "logLR_z_model",
        "logLR_y_model",
        "pS_r_eq54prior",
        "pS_gri_eq54prior",
        "pS_ugrizy_eq54prior",
    ]
    color_cols = [
        "object_id",
        "ra",
        "dec",
        "cmodel_mag_r",
        "ug",
        "gr",
        "ri",
        "iz",
        "zy",
        "truth_label",
        "truth_binary",
        "pS_color",
        "valid_color_features",
        "ps_color_version",
    ]
    matched_cols = ["dp2_object_id", "truth_binary", "truth_label", "dp2_cmodel_mag_r", "dp2_extendedness_r"]
    analysis_cols = ["object_id", "psf_minus_cmodel_r"]
    eq = pd.read_parquet(repo_root / EQ54_INPUT, columns=eq_cols)
    color = pd.read_parquet(repo_root / COLOR_INPUT, columns=color_cols)
    matched = pd.read_parquet(repo_root / MATCHED_INPUT, columns=matched_cols)
    analysis = pd.read_parquet(repo_root / ANALYSIS_INPUT, columns=analysis_cols)
    return eq, color, matched, analysis


def validate_join_inputs(eq: pd.DataFrame, color: pd.DataFrame) -> None:
    if "object_id" not in eq.columns or "object_id" not in color.columns:
        raise RuntimeError("No reliable object_id join key found in both Eq.54 and pS_color inputs.")
    if eq["object_id"].nunique() != len(eq):
        raise RuntimeError("Eq.54 input object_id is not unique.")
    if color["object_id"].nunique() != len(color):
        raise RuntimeError("pS_color input object_id is not unique.")
    required_eq = [
        "log_prior_odds_eq54",
        "logLR_u_model",
        "logLR_g_model",
        "logLR_r_model",
        "logLR_i_model",
        "logLR_z_model",
        "logLR_y_model",
    ]
    missing = [c for c in required_eq if c not in eq.columns]
    if missing:
        raise RuntimeError(f"Missing Eq.54 model/prior columns: {missing}")


def build_color_integrated_table(eq: pd.DataFrame, color: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    validate_join_inputs(eq, color)
    joined = eq.merge(
        color.drop(columns=["ra", "dec", "cmodel_mag_r"], errors="ignore"),
        on="object_id",
        how="inner",
        validate="one_to_one",
    )
    joined = joined.merge(analysis, on="object_id", how="left", validate="one_to_one")
    joined["pS_color_clip"] = np.clip(pd.to_numeric(joined["pS_color"], errors="coerce"), EPS, 1.0 - EPS)
    joined["logLR_color"] = logit_probability(joined["pS_color"], EPS)
    joined["logLR_r_total_color"] = joined["logLR_r_model"] + joined["logLR_color"]
    joined["logLR_gri_total_color"] = joined[["logLR_g_model", "logLR_r_model", "logLR_i_model"]].sum(axis=1, min_count=3) + joined["logLR_color"]
    joined["logLR_ugrizy_total_color"] = joined[
        ["logLR_u_model", "logLR_g_model", "logLR_r_model", "logLR_i_model", "logLR_z_model", "logLR_y_model"]
    ].sum(axis=1, min_count=6) + joined["logLR_color"]
    joined["pS_r_eq54prior_color"] = sigmoid(joined["logLR_r_total_color"] + joined["log_prior_odds_eq54"])
    joined["pS_gri_eq54prior_color"] = sigmoid(joined["logLR_gri_total_color"] + joined["log_prior_odds_eq54"])
    joined["pS_ugrizy_eq54prior_color"] = sigmoid(joined["logLR_ugrizy_total_color"] + joined["log_prior_odds_eq54"])
    joined["ps_version"] = "eq54prior_color_v1"
    keep = [
        "object_id",
        "ra",
        "dec",
        "cmodel_mag_r",
        "psf_minus_cmodel_r",
        "ug",
        "gr",
        "ri",
        "iz",
        "zy",
        "truth_label",
        "truth_binary",
        "valid_color_features",
        "pS_color",
        "pS_color_clip",
        "logLR_color",
        "log_prior_odds_eq54",
        "logLR_r_model",
        "logLR_g_model",
        "logLR_i_model",
        "logLR_u_model",
        "logLR_z_model",
        "logLR_y_model",
        "logLR_r_total_color",
        "logLR_gri_total_color",
        "logLR_ugrizy_total_color",
        "pS_r_eq54prior",
        "pS_gri_eq54prior",
        "pS_ugrizy_eq54prior",
        "pS_r_eq54prior_color",
        "pS_gri_eq54prior_color",
        "pS_ugrizy_eq54prior_color",
        "ps_version",
        "ps_color_version",
    ]
    return joined[keep].copy()


def write_output_parquet(repo_root: Path, joined: pd.DataFrame) -> Path:
    path = repo_root / OUTPUT_PARQUET
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing derived output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(path, index=False)
    return path


def score_stats(df: pd.DataFrame, col: str) -> dict[str, float | int]:
    vals = pd.to_numeric(df[col], errors="coerce")
    finite = vals[np.isfinite(vals)]
    return {
        "finite_count": int(finite.size),
        "min": float(finite.min()) if finite.size else np.nan,
        "median": float(finite.median()) if finite.size else np.nan,
        "max": float(finite.max()) if finite.size else np.nan,
    }


def build_performance_rows(eval_df: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = [
        ("eq54_morphology_r", "pS_r_eq54prior", "r morphology only"),
        ("pS_color_only", "pS_color", "color-only Random Forest"),
        ("eq54_color_r", "pS_r_eq54prior_color", "r morphology + color"),
        ("eq54_morphology_gri", "pS_gri_eq54prior", "gri morphology only"),
        ("eq54_color_gri", "pS_gri_eq54prior_color", "gri morphology + color"),
        ("eq54_morphology_ugrizy", "pS_ugrizy_eq54prior", "ugrizy morphology only"),
        ("eq54_color_ugrizy", "pS_ugrizy_eq54prior_color", "ugrizy morphology + color"),
    ]
    for lo, hi in PERFORMANCE_BINS:
        in_bin = eval_df["cmodel_mag_r"].ge(lo) & eval_df["cmodel_mag_r"].lt(hi)
        sub = eval_df.loc[in_bin].copy()
        for method, score_col, label in methods:
            metrics = operating_metrics(sub["truth_binary"], score=sub[score_col])
            rows.append(
                {
                    "magnitude_bin": f"{lo:g} <= r < {hi:g}",
                    "mag_low": lo,
                    "mag_high": hi,
                    "method": method,
                    "score_column": score_col,
                    "score_label": label,
                    "AUC": compute_auc(sub["truth_binary"], sub[score_col]),
                    **metrics,
                }
            )
        matched_sub = matched[matched["dp2_cmodel_mag_r"].ge(lo) & matched["dp2_cmodel_mag_r"].lt(hi)].copy()
        ext = pd.to_numeric(matched_sub["dp2_extendedness_r"], errors="coerce")
        pred_star = ext.eq(0).astype(object)
        pred_star[~ext.isin([0, 1])] = np.nan
        metrics = operating_metrics(matched_sub["truth_binary"], pred_star=pred_star)
        rows.append(
            {
                "magnitude_bin": f"{lo:g} <= r < {hi:g}",
                "mag_low": lo,
                "mag_high": hi,
                "method": "r_extendedness",
                "score_column": "dp2_extendedness_r",
                "score_label": "r-band extendedness",
                "AUC": np.nan,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def write_summary(repo_root: Path, eq: pd.DataFrame, color: pd.DataFrame, joined: pd.DataFrame, performance: pd.DataFrame) -> list[Path]:
    rows: list[dict[str, object]] = [
        {"metric": "eq54_rows", "value": len(eq), "notes": str(EQ54_INPUT)},
        {"metric": "pS_color_rows", "value": len(color), "notes": str(COLOR_INPUT)},
        {"metric": "joined_rows", "value": len(joined), "notes": "inner join on unique object_id"},
        {"metric": "valid_color_feature_rows", "value": int(joined["valid_color_features"].fillna(False).astype(bool).sum()), "notes": "pS_color valid feature flag"},
        {"metric": "truth_labeled_rows", "value": int(pd.to_numeric(joined["truth_binary"], errors="coerce").isin([0, 1]).sum()), "notes": "truth labels from pS_color output"},
    ]
    for col in [
        "pS_r_eq54prior",
        "pS_r_eq54prior_color",
        "pS_gri_eq54prior",
        "pS_gri_eq54prior_color",
        "pS_ugrizy_eq54prior",
        "pS_ugrizy_eq54prior_color",
        "pS_color",
    ]:
        stats = score_stats(joined, col)
        rows.extend(
            [
                {"metric": f"{col}_finite_count", "value": stats["finite_count"], "notes": ""},
                {"metric": f"{col}_min", "value": stats["min"], "notes": ""},
                {"metric": f"{col}_median", "value": stats["median"], "notes": ""},
                {"metric": f"{col}_max", "value": stats["max"], "notes": ""},
            ]
        )
    summary = pd.DataFrame(rows)
    RESULT_DIR_ABS = repo_root / RESULT_DIR
    RESULT_DIR_ABS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR_ABS / "eq54_color_integration_summary.csv"
    md_path = RESULT_DIR_ABS / "eq54_color_integration_summary.md"
    perf_path = RESULT_DIR_ABS / "eq54_color_integration_performance_by_rmag.csv"
    summary.to_csv(csv_path, index=False)
    performance.to_csv(perf_path, index=False)
    lines = [
        "# Eq.54 + pS_color Integration Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- Eq.54 morphology/model posterior input: `{EQ54_INPUT}`",
        f"- Color-only input: `{COLOR_INPUT}`",
        "- Join key: `object_id` (unique in both inputs).",
        "",
        "## Formula",
        "- `logLR_color = logit(pS_color)` with clipping epsilon `1e-6`.",
        "- `logLR_total = logLR_morphology + logLR_color`.",
        "- `pS_eq54prior_color = sigmoid(logLR_total + log_prior_odds_eq54)`.",
        "- The Eq.54 magnitude prior is applied once, not once per band.",
        "",
        "## Counts",
        f"- Eq.54 rows: {len(eq):,}",
        f"- pS_color rows: {len(color):,}",
        f"- joined rows: {len(joined):,}",
        f"- valid color-feature rows: {int(joined['valid_color_features'].fillna(False).astype(bool).sum()):,}",
        f"- truth-labeled rows: {int(pd.to_numeric(joined['truth_binary'], errors='coerce').isin([0, 1]).sum()):,}",
        "",
        "## Caveats",
        "- This is a first-pass post-hoc likelihood-factor integration.",
        "- It assumes the exploratory Random Forest `pS_color` is calibrated enough for logit conversion.",
        "- The color classifier was not retrained in this run.",
        "- Existing Eq.54 and pS_color parquet inputs were not modified.",
        "",
        f"Detailed scalar summary: `{csv_path.relative_to(repo_root)}`",
        f"Performance summary: `{perf_path.relative_to(repo_root)}`",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path, perf_path]


def plot_fig2_4(repo_root: Path, joined: pd.DataFrame) -> list[Path]:
    set_paper_style()
    out = repo_root / FIGURE_DIR / "fig2_4_cosmos_r_pS_map_16_26_eq54prior_color.png"
    use = joined[
        joined["cmodel_mag_r"].ge(16)
        & joined["cmodel_mag_r"].lt(26)
        & np.isfinite(joined["psf_minus_cmodel_r"])
        & np.isfinite(joined["pS_r_eq54prior_color"])
    ].copy()
    plot_df = downsample_frame(use, 120_000, random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    sc = ax.scatter(
        plot_df["cmodel_mag_r"],
        plot_df["psf_minus_cmodel_r"],
        c=plot_df["pS_r_eq54prior_color"],
        s=1.2,
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.75,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("pS_r, morphology + color")
    ax.set_xlim(16, 26)
    finite_y = use["psf_minus_cmodel_r"].dropna().to_numpy(float)
    if finite_y.size:
        lo, hi = np.nanpercentile(finite_y, [0.2, 99.8])
        ax.set_ylim(max(-0.4, lo), min(2.2, hi))
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("r PSF - CModel")
    ax.set_title("COSMOS r-band pS map, morphology + color")
    return save_figure(fig, out, write_pdf=True)


def plot_performance_grid(performance: pd.DataFrame, out_png: Path, methods: list[tuple[str, str, str]]) -> list[Path]:
    set_paper_style()
    metrics = ("completeness", "contamination", "purity")
    x = np.arange(len(PERFORMANCE_BINS))
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 10.4), sharex=True)
    for ax, metric in zip(axes, metrics):
        for method, label, color in methods:
            vals = []
            for lo, hi in PERFORMANCE_BINS:
                row = performance[performance["method"].eq(method) & performance["mag_low"].eq(lo) & performance["mag_high"].eq(hi)]
                vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ax.plot(x, vals, marker="o", ms=6.5, lw=2.0, color=color, label=label)
        ax.set_ylabel(metric)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(PERFORMANCE_BIN_LABELS)
    axes[-1].set_xlabel("r CModel magnitude bin")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 0.005), frameon=True)
    fig.subplots_adjust(bottom=0.13, hspace=0.18)
    return save_figure(fig, out_png, write_pdf=True)


def plot_fig2_5(repo_root: Path, performance: pd.DataFrame) -> list[Path]:
    return plot_performance_grid(
        performance,
        repo_root / FIGURE_DIR / "fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_color.png",
        [
            ("r_extendedness", "r extendedness", "#ff7f0e"),
            ("eq54_morphology_r", "r morphology only", "#222222"),
            ("eq54_color_r", "r morphology + color", "#1f77b4"),
        ],
    )


def plot_fig2_6(repo_root: Path, performance: pd.DataFrame) -> list[Path]:
    return plot_performance_grid(
        performance,
        repo_root / FIGURE_DIR / "fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_color.png",
        [
            ("eq54_color_r", "r morphology + color", "#1f77b4"),
            ("eq54_color_gri", "gri morphology + color", "#2ca02c"),
            ("eq54_color_ugrizy", "ugrizy morphology + color", "#9467bd"),
        ],
    )


def plot_fig2_7(repo_root: Path, joined: pd.DataFrame) -> list[Path]:
    set_paper_style()
    out = repo_root / FIGURE_DIR / "fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color.png"
    color_planes = [("ug", "gr", "u-g", "g-r"), ("gr", "ri", "g-r", "r-i"), ("ri", "iz", "r-i", "i-z"), ("iz", "zy", "i-z", "z-y")]
    mag_bins = [(16.0, 25.0), (25.0, 26.0)]
    rmag = pd.to_numeric(joined["cmodel_mag_r"], errors="coerce")
    score = pd.to_numeric(joined["pS_ugrizy_eq54prior_color"], errors="coerce")
    method_star = score.ge(OPERATING_THRESHOLD)
    fig, axes = plt.subplots(4, 2, figsize=FIG_SIZES["4x2"], sharex=False, sharey=False)
    rows: list[dict[str, object]] = []
    for row_idx, (xcol, ycol, xlabel, ylabel) in enumerate(color_planes):
        xlim, ylim = COLOR_COLOR_LIMITS[(xcol, ycol)]
        for col_idx, (lo, hi) in enumerate(mag_bins):
            ax = axes[row_idx, col_idx]
            finite = rmag.ge(lo) & rmag.lt(hi) & np.isfinite(score) & np.isfinite(joined[xcol]) & np.isfinite(joined[ycol])
            panel = joined.loc[finite, [xcol, ycol]].copy()
            star_panel = panel.loc[method_star[finite]]
            gal_panel = panel.loc[~method_star[finite]]
            gal_plot = downsample_frame(gal_panel, 80_000, random_state=RANDOM_SEED)
            star_plot = downsample_frame(star_panel, 45_000, random_state=RANDOM_SEED)
            ax.scatter(gal_plot[xcol], gal_plot[ycol], s=1.0, c=COLORS["galaxy"], alpha=0.08, linewidths=0, rasterized=True)
            ax.scatter(star_plot[xcol], star_plot[ycol], s=1.3, c=COLORS["star"], alpha=0.28, linewidths=0, rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{lo:g} < r < {hi:g}\nN_unres={len(star_panel):,}, N_res={len(gal_panel):,}")
            rows.append(
                {
                    "x_color": xcol,
                    "y_color": ycol,
                    "mag_low": lo,
                    "mag_high": hi,
                    "N_finite": int(len(panel)),
                    "N_classified_star": int(len(star_panel)),
                    "N_classified_galaxy": int(len(gal_panel)),
                    "score_column": "pS_ugrizy_eq54prior_color",
                    "threshold": OPERATING_THRESHOLD,
                }
            )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="resolved"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="unresolved"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.07, hspace=0.58, wspace=0.30)
    saved = save_figure(fig, out, write_pdf=True)
    summary = repo_root / RESULT_DIR / "fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return [*saved, summary]


def plot_fig5_10(repo_root: Path, joined: pd.DataFrame) -> list[Path]:
    set_paper_style()
    out = repo_root / FIGURE5_DIR / "fig5_10_cosmos_eq54_vs_eq54_color_pS_hist_by_rmag.png"
    scores = [
        ("pS_r_eq54prior", "r morphology only", "#222222"),
        ("pS_color", "pS_color", "#ff7f0e"),
        ("pS_r_eq54prior_color", "r morphology + color", "#1f77b4"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14.6, 10.2), sharex=True, sharey=False)
    hist_bins = np.linspace(0, 1, 41)
    for row_idx, (lo, hi) in enumerate(PERFORMANCE_BINS):
        in_bin = joined["cmodel_mag_r"].ge(lo) & joined["cmodel_mag_r"].lt(hi)
        for col_idx, (score_col, label, color) in enumerate(scores):
            ax = axes[row_idx, col_idx]
            for truth_value, ls, suffix in [(0, "-", "galaxy"), (1, "--", "star")]:
                vals = pd.to_numeric(joined.loc[in_bin & joined["truth_binary"].eq(truth_value), score_col], errors="coerce").dropna()
                if len(vals):
                    ax.hist(vals, bins=hist_bins, density=True, histtype="step", lw=1.7, ls=ls, color=color, label=suffix)
            ax.axvline(0.5, color="0.35", lw=1.0, ls=":")
            ax.set_xlim(0, 1)
            ax.set_title(f"{label}\n{lo:g} < r < {hi:g}")
            if col_idx == 0:
                ax.set_ylabel("density")
            if row_idx == 2:
                ax.set_xlabel("pS")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.22)
    return save_figure(fig, out, write_pdf=True)


def write_report(repo_root: Path, output_files: list[Path]) -> Path:
    report = repo_root / DOC_DIR / "section2_eq54_color_integration_report.md"
    lines = [
        "# Section 2 Eq.54 + pS_color Integration Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "`pS_color` was previously a diagnostic color-only star probability. This run integrates it into the final likelihood/posterior calculation as an additional color likelihood-ratio factor.",
        "",
        "## Implementation",
        "",
        "- `pS_color` is converted to `logLR_color = logit(pS_color)` with clipping at `1e-6` and `1 - 1e-6`.",
        "- Morphology/model log-likelihood ratios are taken from `outputs/dp2_cosmos_ps_v9_eq54prior.parquet`.",
        "- Combined log-likelihood ratios use `logLR_total = logLR_morphology + logLR_color`.",
        "- The Eq.54 magnitude prior is applied once: `pS = sigmoid(logLR_total + log_prior_odds_eq54)`.",
        "- No stored Eq.54 or pS_color input parquet was modified.",
        "- The color classifier was not retrained.",
        "",
        "## Caveats",
        "",
        "- This is a first-pass post-hoc likelihood-factor integration.",
        "- It assumes the exploratory Random Forest `pS_color` is calibrated enough for logit conversion.",
        "- A cleaner cross-validated color classifier may be needed before final paper claims.",
        "- `pS_color` is color-only and does not use morphology.",
        "",
        "## Outputs",
        "",
    ]
    for path in output_files:
        lines.append(f"- `{path.relative_to(repo_root)}`")
    report.write_text("\n".join(lines) + "\n")
    return report


def run(repo_root: Path) -> dict[str, list[Path]]:
    eq, color, matched, analysis = read_inputs(repo_root)
    joined = build_color_integrated_table(eq, color, analysis)
    output_parquet = write_output_parquet(repo_root, joined)
    eval_df = joined[pd.to_numeric(joined["truth_binary"], errors="coerce").isin([0, 1])].copy()
    performance = build_performance_rows(eval_df, matched)
    generated: dict[str, list[Path]] = {"data": [output_parquet]}
    generated["summary"] = write_summary(repo_root, eq, color, joined, performance)
    generated["fig2_4"] = plot_fig2_4(repo_root, joined)
    generated["fig2_5"] = plot_fig2_5(repo_root, performance)
    generated["fig2_6"] = plot_fig2_6(repo_root, performance)
    generated["fig2_7"] = plot_fig2_7(repo_root, joined)
    generated["fig5_10"] = plot_fig5_10(repo_root, joined)
    all_outputs = [p for values in generated.values() for p in values]
    generated["docs"] = [write_report(repo_root, all_outputs)]
    return generated


if __name__ == "__main__":
    root = repo_root_from_file()
    outputs = run(root)
    for group, paths in outputs.items():
        print(group)
        for path in paths:
            print(" ", path.relative_to(root))

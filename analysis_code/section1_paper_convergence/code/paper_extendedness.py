"""Extendedness baseline figures for COSMOS paper convergence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style
from paper_sample_selection import truth_masks

MAG_SPLITS = ((16.0, 25.0), (25.0, 26.0))
PERFORMANCE_BINS = ((16.0, 24.0), (24.0, 25.0), (25.0, 26.0))


def class_masks(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return star-like, galaxy-like, and valid masks for extendedness-like columns."""

    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.isin([0, 1])
    star_like = values.eq(0) & valid
    galaxy_like = values.eq(1) & valid
    return star_like, galaxy_like, valid


def verify_extendedness_convention(matched: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    stars, galaxies = truth_masks(matched)
    rows = []
    for col in columns:
        star_like, galaxy_like, valid = class_masks(matched, col)
        rows.extend(
            [
                {
                    "column": col,
                    "group": "external_star",
                    "N_total": int(stars.sum()),
                    "N_value_0_unresolved": int((stars & star_like).sum()),
                    "N_value_1_resolved": int((stars & galaxy_like).sum()),
                    "N_nan_or_other": int((stars & ~valid).sum()),
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                },
                {
                    "column": col,
                    "group": "external_galaxy",
                    "N_total": int(galaxies.sum()),
                    "N_value_0_unresolved": int((galaxies & star_like).sum()),
                    "N_value_1_resolved": int((galaxies & galaxy_like).sum()),
                    "N_nan_or_other": int((galaxies & ~valid).sum()),
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                },
            ]
        )
    return pd.DataFrame(rows)


def _limit_string(limits) -> str:
    return str(tuple(float(x) for x in limits))


def _smooth_hist2d(hist: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return hist
    return gaussian_filter(hist, sigma=sigma)


def _draw_density_contours(
    ax,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlim,
    ylim,
    color: str,
    *,
    bins: int = 120,
    sigma: float = 1.0,
    linewidth: float = 1.0,
    alpha: float = 0.78,
) -> bool:
    x_range = (float(min(xlim)), float(max(xlim)))
    y_range = (float(min(ylim)), float(max(ylim)))
    use = data[[x_col, y_col]].copy()
    use[x_col] = pd.to_numeric(use[x_col], errors="coerce")
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use = use[
        np.isfinite(use[x_col])
        & np.isfinite(use[y_col])
        & use[x_col].between(*x_range)
        & use[y_col].between(*y_range)
    ]
    if len(use) < 100:
        return False
    hist, x_edges, y_edges = np.histogram2d(
        use[x_col],
        use[y_col],
        bins=bins,
        range=[x_range, y_range],
    )
    smooth = _smooth_hist2d(hist.T, sigma=sigma)
    positive = smooth[smooth > 0]
    if positive.size < 8:
        return False
    levels = np.nanpercentile(positive, [50, 68, 82, 92, 97])
    levels = np.unique(levels[levels > 0])
    if levels.size < 2:
        return False
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers,
        y_centers,
        smooth,
        levels=levels,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )
    return True


def _scatter_points(ax, data: pd.DataFrame, x_col: str, y_col: str, color: str, *, s: float = 10.0, alpha: float = 0.58) -> None:
    use = data[[x_col, y_col]].copy()
    use[x_col] = pd.to_numeric(use[x_col], errors="coerce")
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use = use[np.isfinite(use[x_col]) & np.isfinite(use[y_col])]
    ax.scatter(
        use[x_col],
        use[y_col],
        s=s,
        c=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )


def _scatter_class(ax, df: pd.DataFrame, class_col: str, x_col: str, y_col: str, xlim, ylim) -> dict:
    star_like, galaxy_like, valid = class_masks(df, class_col)
    finite = valid & np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    gal = df.loc[galaxy_like & finite]
    star = df.loc[star_like & finite]
    drew_contours = _draw_density_contours(ax, gal, x_col, y_col, xlim, ylim, COLORS["galaxy"])
    if not drew_contours:
        _scatter_points(ax, gal, x_col, y_col, COLORS["galaxy"], s=2.0, alpha=0.18)
    _scatter_points(ax, star, x_col, y_col, COLORS["star"], s=10.0, alpha=0.58)
    return {
        "N_unresolved_finite": int(len(star)),
        "N_resolved_finite": int(len(gal)),
        "N_nan_or_other": int((~valid).sum()),
        "resolved_display": "density_contours" if drew_contours else "scatter_fallback",
        "unresolved_display": "scatter_points",
    }


def plot_extendedness_color_color(
    df: pd.DataFrame,
    class_col: str,
    title_label: str,
    output_png: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Create Fig 1.4-style color-color diagram using extendedness labels."""

    set_paper_style()
    fig, axes = plt.subplots(4, 2, figsize=FIG_SIZES["4x2"])
    fig.subplots_adjust(left=0.110, right=0.985, bottom=0.075, top=0.975, hspace=0.62, wspace=0.30)
    rowspecs = [
        ("color_ug", "color_gr", "u-g", "g-r", ("ug", "gr")),
        ("color_gr", "color_ri", "g-r", "r-i", ("gr", "ri")),
        ("color_ri", "color_iz", "r-i", "i-z", ("ri", "iz")),
        ("color_iz", "color_zy", "i-z", "z-y", ("iz", "zy")),
    ]
    rows = []
    rmag = pd.to_numeric(df["cmodel_mag_r"], errors="coerce")
    for row_idx, (x_col, y_col, x_label, y_label, limit_key) in enumerate(rowspecs):
        for col_idx, (lo, hi) in enumerate(MAG_SPLITS):
            ax = axes[row_idx, col_idx]
            use = df.loc[rmag.gt(lo) & rmag.lt(hi)]
            xlim, ylim = COLOR_COLOR_LIMITS[limit_key]
            counts = _scatter_class(ax, use, class_col, x_col, y_col, xlim, ylim)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_unres={counts['N_unresolved_finite']:,}, N_res={counts['N_resolved_finite']:,}")
            rows.append(
                {
                    "classification_column": class_col,
                    "x_column": x_col,
                    "y_column": y_col,
                    "mag_low": lo,
                    "mag_high": hi,
                    **counts,
                    "xlim": _limit_string(xlim),
                    "ylim": _limit_string(ylim),
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                }
            )
    handles = [
        Line2D([0], [0], color=COLORS["galaxy"], lw=1.4, alpha=0.78, label="resolved"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=7, alpha=0.75, label="unresolved"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.012))
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return save_figure(fig, output_png), pd.DataFrame(rows)


def plot_confusion_cmd(
    matched: pd.DataFrame,
    class_col: str,
    title_label: str,
    output_png: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Create Fig 1.5-style CMD confusion panels against COSMOS2020 truth labels."""

    set_paper_style()
    truth_star, truth_gal = truth_masks(matched)
    pred_star, pred_gal, valid = class_masks(matched, class_col)
    finite = valid & np.isfinite(matched["color_gi"]) & np.isfinite(matched["dp2_cmodel_mag_r"])
    categories = [
        ("SS", truth_star & pred_star & finite, COLORS["star"], "truth star, class star"),
        ("SG", truth_star & pred_gal & finite, COLORS["star"], "truth star, class galaxy"),
        ("GS", truth_gal & pred_star & finite, COLORS["galaxy"], "truth galaxy, class star"),
        ("GG", truth_gal & pred_gal & finite, COLORS["galaxy"], "truth galaxy, class galaxy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["2x2"], sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.095, top=0.965, hspace=0.28, wspace=0.12)
    rows = []
    for idx, (ax, (label, mask, color, subtitle)) in enumerate(zip(axes.flat, categories)):
        data = matched.loc[mask]
        ax.set_xlim(-0.8, 4.0)
        ax.set_ylim(26, 16)
        if label.startswith("G"):
            drew_contours = _draw_density_contours(
                ax,
                data,
                "color_gi",
                "dp2_cmodel_mag_r",
                (-0.8, 4.0),
                (16.0, 26.0),
                COLORS["galaxy"],
                bins=100,
            )
            if not drew_contours:
                _scatter_points(ax, data, "color_gi", "dp2_cmodel_mag_r", COLORS["galaxy"], s=2.0, alpha=0.18)
            display_style = "density_contours" if drew_contours else "scatter_fallback"
            main_style_reason = "truth-galaxy panels use red density contours"
        else:
            _scatter_points(ax, data, "color_gi", "dp2_cmodel_mag_r", COLORS["star"], s=10.0, alpha=0.58)
            display_style = "scatter_points"
            main_style_reason = "truth-star panels use larger blue scatter points"
        ax.set_title(f"{label}: {subtitle}\nN={len(data):,}")
        if idx // 2 == 1:
            ax.set_xlabel("g-i")
        else:
            ax.set_xlabel("")
        if idx % 2 == 0:
            ax.set_ylabel("r CModel magnitude")
        else:
            ax.set_ylabel("")
        rows.append(
            {
                "classification_column": class_col,
                "panel": label,
                "description": subtitle,
                "N": int(len(data)),
                "x_column": "color_gi",
                "y_column": "dp2_cmodel_mag_r",
                "xlim": "(-0.8, 4.0)",
                "ylim": "(26.0, 16.0)",
                "display_style": display_style,
                "main_style_reason": main_style_reason,
                "convention": "first letter=COSMOS2020 truth, second letter=extendedness classification",
            }
        )
    return save_figure(fig, output_png), pd.DataFrame(rows)


def _binary_metrics(truth_positive: pd.Series, pred_positive: pd.Series, valid: pd.Series) -> dict:
    truth_positive = truth_positive & valid
    truth_negative = (~truth_positive) & valid
    pred_positive = pred_positive & valid
    pred_negative = (~pred_positive) & valid
    tp = int((truth_positive & pred_positive).sum())
    fn = int((truth_positive & pred_negative).sum())
    fp = int((truth_negative & pred_positive).sum())
    tn = int((truth_negative & pred_negative).sum())
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    contamination = fp / (tp + fp) if (tp + fp) else np.nan
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "completeness_TPR": tpr,
        "false_positive_rate": fpr,
        "purity_precision": precision,
        "contamination": contamination,
    }


def extendedness_metrics_by_mag(matched: pd.DataFrame, class_col: str) -> pd.DataFrame:
    truth_star, truth_gal = truth_masks(matched)
    pred_star, pred_gal, valid_class = class_masks(matched, class_col)
    rmag = pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce")
    rows = []
    for lo, hi in PERFORMANCE_BINS:
        in_bin = rmag.gt(lo) & rmag.lt(hi)
        valid = in_bin & valid_class & (truth_star | truth_gal)
        for positive_class, truth_positive, pred_positive in [
            ("star", truth_star, pred_star),
            ("galaxy", truth_gal, pred_gal),
        ]:
            metrics = _binary_metrics(truth_positive, pred_positive, valid)
            rows.append(
                {
                    "classification_column": class_col,
                    "mag_low": lo,
                    "mag_high": hi,
                    "positive_class": positive_class,
                    "N_valid": int(valid.sum()),
                    "N_star": int((truth_star & valid).sum()),
                    "N_galaxy": int((truth_gal & valid).sum()),
                    "N_nan_or_other_classification": int((in_bin & ~valid_class).sum()),
                    **metrics,
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                    "metric_definitions": (
                        "TP=truth positive classified positive; FP=truth negative classified positive; "
                        "TN=truth negative classified negative; FN=truth positive classified negative; "
                        "completeness = TP / (TP + FN); contamination = FP / (TP + FP); "
                        "purity = TP / (TP + FP)"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_extendedness_performance(
    matched: pd.DataFrame,
    class_col: str,
    title_label: str,
    output_png: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Create Fig 1.6 binary completeness-contamination operating-point panels."""

    set_paper_style()
    metrics = extendedness_metrics_by_mag(matched, class_col)
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"])
    fig.subplots_adjust(left=0.060, right=0.990, bottom=0.18, top=0.935, wspace=0.26)
    for ax, (lo, hi) in zip(axes, PERFORMANCE_BINS):
        panel = metrics[(metrics["mag_low"] == lo) & (metrics["mag_high"] == hi)]
        for _, row in panel.iterrows():
            color = COLORS["star"] if row["positive_class"] == "star" else COLORS["galaxy"]
            label = f"{row['positive_class']}-positive"
            contamination = row["contamination"]
            tpr = row["completeness_TPR"]
            ax.scatter([contamination], [tpr], color=color, s=72, label=label, zorder=5)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("contamination")
        ax.set_ylabel("completeness")
        ax.set_title(f"{lo:g} < rmag < {hi:g}")
        ax.legend(loc="lower right", frameon=True, fontsize=9)
    return save_figure(fig, output_png), metrics


def compare_extendedness_metrics(r_metrics: pd.DataFrame, ref_metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    merged = r_metrics.merge(
        ref_metrics,
        on=["mag_low", "mag_high", "positive_class"],
        suffixes=("_r_extendedness", "_refExtendedness"),
    )
    keep = [
        "mag_low",
        "mag_high",
        "positive_class",
        "N_valid_r_extendedness",
        "N_valid_refExtendedness",
        "completeness_TPR_r_extendedness",
        "completeness_TPR_refExtendedness",
        "contamination_r_extendedness",
        "contamination_refExtendedness",
        "false_positive_rate_r_extendedness",
        "false_positive_rate_refExtendedness",
        "purity_precision_r_extendedness",
        "purity_precision_refExtendedness",
    ]
    out = merged[keep].copy()
    out["delta_completeness_ref_minus_r"] = out["completeness_TPR_refExtendedness"] - out["completeness_TPR_r_extendedness"]
    out["delta_contamination_ref_minus_r"] = out["contamination_refExtendedness"] - out["contamination_r_extendedness"]
    md = ["# Extendedness vs refExtendedness Comparison", ""]
    md.append("Conventions: `0=unresolved/star-like`, `1=resolved/galaxy-like`; NaN/other values are excluded from operating-point metrics and counted in the CSV summaries.")
    md.append("Metrics: completeness = TP / (TP + FN); contamination = FP / (TP + FP); purity = TP / (TP + FP).")
    md.append("")
    for _, row in out.iterrows():
        md.append(
            f"- {row['mag_low']:g}-{row['mag_high']:g}, {row['positive_class']} positive: "
            f"completeness r={row['completeness_TPR_r_extendedness']:.3f}, "
            f"ref={row['completeness_TPR_refExtendedness']:.3f}; "
            f"contamination r={row['contamination_r_extendedness']:.3f}, "
            f"ref={row['contamination_refExtendedness']:.3f}."
        )
    return out, "\n".join(md) + "\n"

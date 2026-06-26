"""Section 5 discussion diagnostics for COSMOS paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_bayesian_method import DISPLAY_BINS
from paper_plot_style import COLORS, FIG_SIZES, save_figure, set_paper_style


def _safe_density_hist(ax, values: pd.Series, *, bins, color: str, label: str, lw: float = 1.2) -> None:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    ax.hist(vals, bins=bins, range=(0, 1), density=True, histtype="step", color=color, lw=lw, label=label)


def plot_fig5_1_red_sources(dp2_ps: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    """Compare r-band pS distributions for all COSMOS sources and red sources.

    Red sources are defined using dust-corrected CModel color:
    color_ri = r-i > 1.4. The pS distribution uses existing v8 pS_r
    values without Eq.54 prior calibration.
    """

    set_paper_style()
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZES["2x4"])
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.84, hspace=0.46, wspace=0.30)
    rmag = pd.to_numeric(dp2_ps["cmodel_mag_r"], errors="coerce")
    pS = pd.to_numeric(dp2_ps["pS_r"], errors="coerce")
    color_ri = pd.to_numeric(dp2_ps["color_ri"], errors="coerce")
    hist_bins = np.linspace(0, 1, 51)
    rows = []

    for ax, (lo, hi) in zip(axes.flat, DISPLAY_BINS):
        in_bin = rmag.gt(lo) & rmag.lt(hi) & np.isfinite(pS)
        red = in_bin & color_ri.gt(1.4) & np.isfinite(color_ri)
        all_vals = pS[in_bin]
        red_vals = pS[red]
        n_all = int(len(all_vals))
        n_red = int(len(red_vals))

        _safe_density_hist(ax, all_vals, bins=hist_bins, color="black", label="all sources", lw=1.1)
        _safe_density_hist(ax, red_vals, bins=hist_bins, color=COLORS["galaxy"], label="red sources: r-i > 1.4", lw=1.5)
        ax.axvline(0.5, color=COLORS["threshold"], ls="--", lw=1.0, label="pS_r = 0.5")
        if n_all == 0:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            med_all = med_red = np.nan
        else:
            med_all = float(np.nanmedian(all_vals))
            med_red = float(np.nanmedian(red_vals)) if n_red else np.nan
            ax.axvline(med_all, color="black", lw=0.9, alpha=0.7)
            if n_red:
                ax.axvline(med_red, color=COLORS["galaxy"], lw=0.9, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("pS_r")
        ax.set_ylabel("normalized density")
        ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_all={n_all:,}, N_red={n_red:,}")
        rows.append(
            {
                "mag_low": lo,
                "mag_high": hi,
                "N_all": n_all,
                "N_red_ri_gt_1p4": n_red,
                "red_fraction": n_red / n_all if n_all else np.nan,
                "median_pS_all": med_all,
                "median_pS_red": med_red,
                "p16_pS_all": np.nanpercentile(all_vals, 16) if n_all else np.nan,
                "p84_pS_all": np.nanpercentile(all_vals, 84) if n_all else np.nan,
                "p16_pS_red": np.nanpercentile(red_vals, 16) if n_red else np.nan,
                "p84_pS_red": np.nanpercentile(red_vals, 84) if n_red else np.nan,
                "red_definition": "dust-corrected r-i > 1.4",
                "notes": "Existing v8 pS_r; no Eq.54 prior calibration yet.",
            }
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Fig 5.1 COSMOS red-source r-band pS diagnostic\nno Eq. 54 prior calibration yet", y=0.965, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def _fraction_above(values: pd.Series, threshold: float) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals > threshold))


def _percentile(values: pd.Series, percentile: float) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.nanpercentile(vals, percentile))


def plot_fig5_1_red_sources_eq54prior(
    dp2_ps: pd.DataFrame,
    output_png: Path,
    *,
    rmag_col: str = "cmodel_mag_r",
    color_col: str = "ri",
    ps_col: str = "pS_r_eq54prior",
) -> tuple[list[Path], pd.DataFrame]:
    """Compare Eq.54-prior r-band pS distributions for all and red sources.

    The red-source selection uses the existing dust-corrected CModel color
    column ``ri``. This function intentionally writes a separate Eq.54-prior
    figure so the older stored-pS diagnostic remains available unchanged.
    """

    required = [rmag_col, color_col, ps_col]
    missing = [col for col in required if col not in dp2_ps.columns]
    if missing:
        raise KeyError(f"Missing required columns for Fig 5.1 Eq.54 diagnostic: {missing}")

    set_paper_style()
    mag_bins = [(16.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    hist_bins = np.linspace(0, 1, 51)
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"], sharex=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.23, top=0.82, wspace=0.28)

    rmag = pd.to_numeric(dp2_ps[rmag_col], errors="coerce")
    color_ri = pd.to_numeric(dp2_ps[color_col], errors="coerce")
    pS = pd.to_numeric(dp2_ps[ps_col], errors="coerce")
    paper_sample = rmag.gt(16.0) & rmag.lt(26.0)
    red_sample = paper_sample & color_ri.gt(1.4) & np.isfinite(color_ri)
    rows = []

    for ax, (lo, hi) in zip(axes.flat, mag_bins):
        in_bin = paper_sample & rmag.gt(lo) & rmag.lt(hi)
        red = red_sample & rmag.gt(lo) & rmag.lt(hi)
        all_vals = pS[in_bin & np.isfinite(pS)]
        red_vals = pS[red & np.isfinite(pS)]
        n_all = int(in_bin.sum())
        n_red = int(red.sum())
        n_all_finite = int(len(all_vals))
        n_red_finite = int(len(red_vals))

        _safe_density_hist(ax, all_vals, bins=hist_bins, color="black", label="all sources", lw=1.6)
        _safe_density_hist(
            ax,
            red_vals,
            bins=hist_bins,
            color=COLORS["galaxy"],
            label="red sources: r-i > 1.4",
            lw=2.0,
        )
        ax.axvline(0.5, color=COLORS["threshold"], ls="--", lw=1.4, label="pS = 0.5")
        ax.set_xlim(0, 1)
        ax.set_xlabel("pS_r Eq.54 prior")
        ax.set_ylabel("normalized density")
        ax.set_title(f"{lo:g} < r < {hi:g}\nN_all={n_all:,}, N_red={n_red:,}")
        if n_all_finite == 0:
            ax.text(0.5, 0.5, "no finite pS", ha="center", va="center", transform=ax.transAxes)

        row = {
            "mag_low": lo,
            "mag_high": hi,
            "N_all_sources": n_all,
            "N_red_sources_ri_gt_1p4": n_red,
            "red_fraction": n_red / n_all if n_all else np.nan,
            "finite_pS_all_count": n_all_finite,
            "finite_pS_red_count": n_red_finite,
            "median_pS_all": float(np.nanmedian(all_vals)) if n_all_finite else np.nan,
            "median_pS_red": float(np.nanmedian(red_vals)) if n_red_finite else np.nan,
            "p10_pS_all": _percentile(all_vals, 10),
            "p90_pS_all": _percentile(all_vals, 90),
            "p10_pS_red": _percentile(red_vals, 10),
            "p90_pS_red": _percentile(red_vals, 90),
            "fraction_all_pS_gt_0p5": _fraction_above(all_vals, 0.5),
            "fraction_red_pS_gt_0p5": _fraction_above(red_vals, 0.5),
            "fraction_all_pS_gt_0p1": _fraction_above(all_vals, 0.1),
            "fraction_red_pS_gt_0p1": _fraction_above(red_vals, 0.1),
            "fraction_all_pS_gt_0p01": _fraction_above(all_vals, 0.01),
            "fraction_red_pS_gt_0p01": _fraction_above(red_vals, 0.01),
            "fraction_all_pS_gt_0p001": _fraction_above(all_vals, 0.001),
            "fraction_red_pS_gt_0p001": _fraction_above(red_vals, 0.001),
            "rmag_column": rmag_col,
            "color_column": color_col,
            "ps_column": ps_col,
            "red_definition": "dust-corrected r-i > 1.4",
            "notes": "Eq.54-prior-calibrated r-band pS from derived v9 prior output.",
        }
        rows.append(row)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("COSMOS red-source r-band pS diagnostic, Eq.54 prior", y=0.965, fontsize=16)
    return save_figure(fig, output_png), pd.DataFrame(rows)

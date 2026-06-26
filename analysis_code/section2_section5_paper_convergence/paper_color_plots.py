"""Section 1 COSMOS paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style
from paper_sample_selection import truth_masks


def _scatter_truth(ax, df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, *, max_gal=90000):
    stars, galaxies = truth_masks(df)
    gal = df.loc[galaxies & np.isfinite(df[x_col]) & np.isfinite(df[y_col])]
    star = df.loc[stars & np.isfinite(df[x_col]) & np.isfinite(df[y_col])]
    gal_plot = downsample_frame(gal, max_gal)
    star_plot = downsample_frame(star, 25000)
    ax.scatter(gal_plot[x_col], gal_plot[y_col], s=2.0, c=COLORS["galaxy"], alpha=0.10, linewidths=0, label="galaxy")
    ax.scatter(star_plot[x_col], star_plot[y_col], s=3.0, c=COLORS["star"], alpha=0.45, linewidths=0, label="star")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _limit_string(limits) -> str:
    return str(tuple(float(x) for x in limits))


def _hist_log_counts(ax, rows: list[dict], bins: np.ndarray) -> None:
    for row in rows:
        data = pd.to_numeric(row["data"], errors="coerce").dropna()
        counts, edges = np.histogram(data, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        y = np.full(counts.shape, np.nan, dtype=float)
        positive = counts > 0
        y[positive] = np.log10(counts[positive])
        ax.plot(mids, y, drawstyle="steps-mid", color=row["color"], lw=row.get("lw", 1.8), label=row["label"])
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("log10(counts per 0.5 mag)")
    ax.set_xlim(16, 26)
    ax.legend(loc="best", frameon=True)


def _binned_ratio(df: pd.DataFrame, category: str, bins: np.ndarray, mag_col: str, flux_col: str, err_col: str) -> pd.DataFrame:
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    flux = pd.to_numeric(df[flux_col], errors="coerce")
    err = pd.to_numeric(df[err_col], errors="coerce")
    ratio = err / flux
    use = pd.DataFrame({"mag": mag, "ratio": ratio})
    use = use[np.isfinite(use["mag"]) & np.isfinite(use["ratio"]) & (flux > 0) & (err > 0)]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = use.loc[use["mag"].ge(lo) & use["mag"].lt(hi), "ratio"]
        rows.append(
            {
                "panel": "flux_uncertainty_ratio",
                "category": category,
                "mag_low": lo,
                "mag_high": hi,
                "count": int(vals.size),
                "median_flux_err_over_flux": vals.median() if vals.size else np.nan,
                "p16_flux_err_over_flux": vals.quantile(0.16) if vals.size else np.nan,
                "p84_flux_err_over_flux": vals.quantile(0.84) if vals.size else np.nan,
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def plot_fig1_1(samples: dict[str, pd.DataFrame], output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    matched = samples["matched_paper"]
    dp2_only = samples["dp2_only"]
    external_only = samples.get("external_only", pd.DataFrame())
    stars, galaxies = truth_masks(matched)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 19.4))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.055, top=0.975, hspace=0.74)

    ax = axes[0]
    gal = matched.loc[galaxies]
    star = matched.loc[stars]
    gal_plot = downsample_frame(gal, 120000)
    ax.scatter(gal_plot["dp2_ra"], gal_plot["dp2_dec"], s=1.2, c=COLORS["galaxy"], alpha=0.10, linewidths=0, label="galaxy")
    ax.scatter(star["dp2_ra"], star["dp2_dec"], s=2.2, c=COLORS["star"], alpha=0.45, linewidths=0, label="star")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Matched COSMOS2020-DP2 sample")
    ax.invert_xaxis()
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="galaxy"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="star"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        frameon=True,
        ncol=2,
        fontsize=12,
        handletextpad=0.8,
        columnspacing=1.6,
        borderpad=0.55,
    )

    bins = np.arange(16, 26.0001, 0.5)
    ax = axes[1]
    hist_rows = [
        {"data": dp2_only["cmodel_mag_r"], "color": COLORS["dp2_only"], "label": f"DP2-only N={len(dp2_only):,}"},
        {"data": matched.loc[stars, "dp2_cmodel_mag_r"], "color": COLORS["star"], "label": f"matched stars N={int(stars.sum()):,}"},
        {"data": matched.loc[galaxies, "dp2_cmodel_mag_r"], "color": COLORS["galaxy"], "label": f"matched galaxies N={int(galaxies.sum()):,}"},
    ]
    external_rmag_available = False
    ext_r = pd.Series(dtype=float)
    _hist_log_counts(ax, hist_rows, bins)
    ax.text(
        0.98,
        0.05,
        "COSMOS2020-only r CModel\nmagnitude unavailable",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0),
    )
    ax.set_title("Counts vs r magnitude")

    ax = axes[2]
    ratio_rows = []
    categories = [
        ("matched stars", matched.loc[stars], "dp2_cmodel_mag_r", "dp2_cmodel_flux_r", "dp2_cmodel_flux_err_r", COLORS["star"]),
        ("matched galaxies", matched.loc[galaxies], "dp2_cmodel_mag_r", "dp2_cmodel_flux_r", "dp2_cmodel_flux_err_r", COLORS["galaxy"]),
    ]
    for label, df, mag_col, flux_col, err_col, color in categories:
        stat = _binned_ratio(df, label, bins, mag_col, flux_col, err_col)
        ratio_rows.append(stat)
        x = 0.5 * (stat["mag_low"] + stat["mag_high"])
        ax.plot(x, stat["median_flux_err_over_flux"], marker="o", ms=3, lw=1.5, color=color, label=label)
        ax.fill_between(x, stat["p16_flux_err_over_flux"], stat["p84_flux_err_over_flux"], color=color, alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(16, 26)
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("r CModelFluxErr / CModelFlux")
    ax.set_title("Relative model flux uncertainty")
    ax.legend(loc="best", frameon=True)

    saved = save_figure(fig, output_png)

    count_rows = []
    count_notes = {
        "DP2-only": "DP2 paper-sample objects with no COSMOS2020 match within 0.5 arcsec.",
        "matched stars": "Matched objects whose COSMOS2020 truth label is star.",
        "matched galaxies": "Matched objects whose COSMOS2020 truth label is galaxy.",
    }
    for label, data in [
        ("DP2-only", dp2_only["cmodel_mag_r"]),
        ("matched stars", matched.loc[stars, "dp2_cmodel_mag_r"]),
        ("matched galaxies", matched.loc[galaxies, "dp2_cmodel_mag_r"]),
        ("COSMOS2020-only HSC r", ext_r),
    ]:
        if label.startswith("COSMOS2020-only") and not external_rmag_available:
            continue
        counts, edges = np.histogram(pd.to_numeric(data, errors="coerce").dropna(), bins=bins)
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            count_rows.append({"panel": "counts_vs_rmag", "category": label, "mag_low": lo, "mag_high": hi, "count": int(count), "notes": count_notes.get(label, "")})
    if not external_rmag_available:
        count_rows.append(
            {
                "panel": "counts_vs_rmag",
                "category": "COSMOS2020-only",
                "mag_low": np.nan,
                "mag_high": np.nan,
                "count": np.nan,
                "notes": "Unavailable for main panel: no external r-band magnitude directly comparable to DP2 r CModelMag was used.",
            }
        )
    else:
        count_rows.append(
            {
                "panel": "counts_vs_rmag",
                "category": "COSMOS2020-only HSC r",
                "mag_low": np.nan,
                "mag_high": np.nan,
                "count": int(ext_r.size),
                "notes": "External-only curve uses COSMOS2020 FARMER HSC_r_MAG from the local full FARMER FITS; it is not DP2 r CModelMag.",
            }
        )
    summary = pd.concat([pd.DataFrame(count_rows), *ratio_rows], ignore_index=True)
    return saved, summary


def plot_cosmos_radec_quicklooks(samples: dict[str, pd.DataFrame], output_dir: Path) -> tuple[list[Path], str]:
    """Write RA/Dec quicklooks for checking footprint holes by sample type."""

    set_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    quicklooks = [
        (
            "cosmos_radec_dp2_only_quicklook",
            samples["dp2_only"],
            "ra",
            "dec",
            COLORS["dp2_only"],
            "DP2-only paper-sample objects",
            "DP2 objects in the COSMOS paper sample without a COSMOS2020 match.",
        ),
        (
            "cosmos_radec_cosmos2020_only_quicklook",
            samples["external_only"],
            "ra",
            "dec",
            COLORS["external_only"],
            "COSMOS2020-only objects",
            "COSMOS2020 truth-catalog objects inside the coordinate cut without a DP2 match.",
        ),
        (
            "cosmos_radec_matched_quicklook",
            samples["matched_paper"],
            "dp2_ra",
            "dp2_dec",
            COLORS["matched"],
            "Matched COSMOS2020-DP2 paper sample",
            "Matched COSMOS2020-DP2 objects after the paper r-magnitude cut.",
        ),
    ]

    notes = [
        "# COSMOS RA/Dec Holes Quicklook Notes",
        "",
        "These diagnostics compare sky footprints for DP2-only, COSMOS2020-only, and matched samples.",
        "They are diagnostic-only outputs and are not listed as main paper figures.",
        "",
    ]
    for stem, df, ra_col, dec_col, color, title, description in quicklooks:
        use = df[np.isfinite(df[ra_col]) & np.isfinite(df[dec_col])]
        plot_use = downsample_frame(use, 180000)
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        ax.scatter(plot_use[ra_col], plot_use[dec_col], s=1.0, c=color, alpha=0.16, linewidths=0)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(title)
        ax.invert_xaxis()
        saved.extend(save_figure(fig, output_dir / f"{stem}.png"))
        notes.append(f"## {stem}")
        notes.append(f"- Description: {description}")
        notes.append(f"- Full N: {len(use):,}")
        notes.append(f"- Plotted N: {len(plot_use):,}")
        notes.append("")
    return saved, "\n".join(notes) + "\n"


def plot_fig1_2(samples: dict[str, pd.DataFrame], output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    matched = samples["matched_paper"]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["2x2"])
    fig.subplots_adjust(left=0.090, right=0.985, bottom=0.095, top=0.900, hspace=0.30, wspace=0.28)

    panels = [
        (axes[0, 0], "dp2_cmodel_mag_r", "dp2_psf_minus_cmodel_r", "r CModel magnitude", "r PSF - CModel", (18, 26), None, False),
        (axes[0, 1], "color_gi", "dp2_cmodel_mag_r", "g-i", "r CModel magnitude", (-0.8, 4.0), (26, 18), True),
        (axes[1, 0], "color_ug", "color_gr", "u-g", "g-r", *COLOR_COLOR_LIMITS[("ug", "gr")], False),
        (axes[1, 1], "color_gr", "color_ri", "g-r", "r-i", *COLOR_COLOR_LIMITS[("gr", "ri")], False),
    ]

    rows = []
    for ax, x_col, y_col, x_label, y_label, xlim, ylim, fixed_ylim in panels:
        _scatter_truth(ax, matched, x_col, y_col, x_label, y_label)
        ax.set_xlim(xlim)
        if ylim is None:
            vals = pd.to_numeric(matched[y_col], errors="coerce")
            lo, hi = np.nanpercentile(vals, [1, 99])
            pad = 0.08 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 0.1
            ylim = (lo - pad, hi + pad)
        ax.set_ylim(ylim)
        stars, galaxies = truth_masks(matched)
        finite = np.isfinite(matched[x_col]) & np.isfinite(matched[y_col])
        rows.append(
            {
                "panel": y_label,
                "x_column": x_col,
                "y_column": y_col,
                "N_star_finite": int((stars & finite).sum()),
                "N_galaxy_finite": int((galaxies & finite).sum()),
                "xlim": _limit_string(xlim),
                "ylim": _limit_string(ylim),
            }
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="galaxy"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="star"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.985))
    return save_figure(fig, output_png), pd.DataFrame(rows)


def plot_fig1_3(
    samples: dict[str, pd.DataFrame],
    output_png: Path,
    mag_bins: tuple[tuple[float, float], tuple[float, float]] = ((16.0, 25.0), (25.0, 26.0)),
) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    matched = samples["matched_paper"]
    fig, axes = plt.subplots(4, 2, figsize=FIG_SIZES["4x2"])
    fig.subplots_adjust(left=0.110, right=0.985, bottom=0.075, top=0.975, hspace=0.62, wspace=0.30)
    rowspecs = [
        ("color_ug", "color_gr", "u-g", "g-r", ("ug", "gr")),
        ("color_gr", "color_ri", "g-r", "r-i", ("gr", "ri")),
        ("color_ri", "color_iz", "r-i", "i-z", ("ri", "iz")),
        ("color_iz", "color_zy", "i-z", "z-y", ("iz", "zy")),
    ]
    rows = []
    rmag = pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce")
    for row_idx, (x_col, y_col, x_label, y_label, limit_key) in enumerate(rowspecs):
        for col_idx, (lo, hi) in enumerate(mag_bins):
            ax = axes[row_idx, col_idx]
            use = matched.loc[rmag.gt(lo) & rmag.lt(hi)]
            _scatter_truth(ax, use, x_col, y_col, x_label, y_label, max_gal=65000)
            xlim, ylim = COLOR_COLOR_LIMITS[limit_key]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            stars, galaxies = truth_masks(use)
            finite = np.isfinite(use[x_col]) & np.isfinite(use[y_col])
            ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_star={int((stars & finite).sum()):,}, N_gal={int((galaxies & finite).sum()):,}")
            rows.append(
                {
                    "x_column": x_col,
                    "y_column": y_col,
                    "mag_low": lo,
                    "mag_high": hi,
                    "N_star_finite": int((stars & finite).sum()),
                    "N_galaxy_finite": int((galaxies & finite).sum()),
                    "xlim": _limit_string(xlim),
                    "ylim": _limit_string(ylim),
                }
            )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="galaxy"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="star"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.012))
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return save_figure(fig, output_png), pd.DataFrame(rows)

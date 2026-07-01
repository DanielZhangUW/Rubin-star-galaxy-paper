"""Section 2 Bayesian-method paper figures for COSMOS."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from paper_extendedness import class_masks
from paper_metrics import binary_operating_metrics, compute_roc, density_score_train_apply
from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style
from paper_sample_selection import BANDS, paper_mag_mask, truth_masks
from paper_priors import star_galaxy_prior_table


DISPLAY_BINS = (
    (20.0, 21.0),
    (21.0, 22.0),
    (22.0, 23.0),
    (23.0, 24.0),
    (24.0, 24.5),
    (24.5, 25.0),
    (25.0, 25.5),
    (25.5, 26.0),
)
PERFORMANCE_BINS = ((16.0, 24.0), (24.0, 25.0), (25.0, 26.0))
COMBINATIONS = {
    "r": ("r",),
    "gri": ("g", "r", "i"),
    "ugrizy": ("u", "g", "r", "i", "z", "y"),
}


def merge_ps_scores(matched: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    keep = ["object_id"] + [f"pS_{b}" for b in BANDS if f"pS_{b}" in ps.columns]
    return matched.merge(ps[keep].drop_duplicates("object_id"), on="object_id", how="left")


def plot_fig2_1_prior(matched: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    bins = np.arange(16, 26.0001, 0.5)
    table = star_galaxy_prior_table(matched, bins)
    fig, ax1 = plt.subplots(figsize=(8.8, 5.6))
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.14, top=0.86)
    x = table["mag_center"]
    ax1.plot(x, table["star_to_galaxy_ratio"], color=COLORS["star"], marker="o", lw=2, label="matched stars / matched galaxies")
    ax1.set_yscale("log")
    ax1.set_xlabel("uncorrected r CModel magnitude")
    ax1.set_ylabel("star / galaxy count ratio")
    ax1.set_xlim(16, 26)
    ax2 = ax1.twinx()
    ax2.plot(x, table["p_star_prior"], color="#333333", ls="--", marker="s", lw=1.4, label="p_star prior")
    ax2.plot(x, table["p_gal_prior"], color=COLORS["galaxy"], ls=":", marker="^", lw=1.4, label="p_gal prior")
    ax2.set_ylabel("prior probability")
    ax2.set_ylim(0, 1)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
    fig.suptitle("Fig 2.1 COSMOS star/galaxy count-ratio prior vs r magnitude", y=0.965, fontsize=13)
    return save_figure(fig, output_png), table


def _slice_hist_stats(df: pd.DataFrame, mag_col: str, delta_col: str, bins=DISPLAY_BINS) -> pd.DataFrame:
    rows = []
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    delta = pd.to_numeric(df[delta_col], errors="coerce")
    for lo, hi in bins:
        vals = delta[mag.ge(lo) & mag.lt(hi) & np.isfinite(delta)]
        rows.append(
            {
                "mag_low": lo,
                "mag_high": hi,
                "N": int(vals.size),
                "median_delta": vals.median() if vals.size else np.nan,
                "mean_delta": vals.mean() if vals.size else np.nan,
                "mad_sigma_delta": 1.4826 * np.median(np.abs(vals - np.median(vals))) if vals.size else np.nan,
                "p16_delta": vals.quantile(0.16) if vals.size else np.nan,
                "p84_delta": vals.quantile(0.84) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_fig2_2_slice_fits(dp2: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    """Display r-band psfMag-CModelMag slice histograms with robust Gaussian overlays."""

    set_paper_style()
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZES["2x4"])
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.10, top=0.88, hspace=0.42, wspace=0.28)
    rows = []
    mag = pd.to_numeric(dp2["cmodel_mag_r"], errors="coerce")
    delta = pd.to_numeric(dp2["psf_minus_cmodel_r"], errors="coerce")
    x_grid = np.linspace(-0.25, 1.2, 500)
    for ax, (lo, hi) in zip(axes.flat, DISPLAY_BINS):
        vals = delta[mag.ge(lo) & mag.lt(hi) & delta.gt(-0.25) & delta.lt(1.2)]
        n = int(vals.size)
        if n:
            counts, edges, _ = ax.hist(vals, bins=120, range=(-0.25, 1.2), density=True, histtype="step", color="black", lw=0.9, label="data")
            med = float(vals.median())
            sig = float(1.4826 * np.median(np.abs(vals - med)))
            sig = max(sig, 1e-3)
            amp = 1 / (np.sqrt(2 * np.pi) * sig)
            ax.plot(x_grid, amp * np.exp(-0.5 * ((x_grid - med) / sig) ** 2), color=COLORS["star"], lw=1.2, ls="--", label="robust Gaussian guide")
            ax.axvline(med, color=COLORS["star"], lw=1.0)
        else:
            med = sig = np.nan
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{lo:g} < rmag < {hi:g}\nN={n:,}, med={med:.3f}")
        ax.set_xlim(-0.25, 1.2)
        ax.set_xlabel("r psfMag - CModelMag")
        ax.set_ylabel("density")
        rows.append({"mag_low": lo, "mag_high": hi, "N": n, "median_delta": med, "mad_sigma_delta": sig, "fit_status": "robust Gaussian display guide"})
    axes.flat[0].legend(loc="best", frameon=True, fontsize=8)
    fig.suptitle("Fig 2.2 COSMOS r-band psfMag - CModelMag slice distributions", y=0.975, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def _hist2d_density(df: pd.DataFrame, band: str, dm_range=(-0.25, 1.2), mag_range=(16, 26), bins=(110, 90)):
    x = pd.to_numeric(df[f"psf_minus_cmodel_{band}"], errors="coerce")
    y = pd.to_numeric(df[f"cmodel_mag_{band}"], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y) & x.gt(dm_range[0]) & x.lt(dm_range[1]) & y.gt(mag_range[0]) & y.lt(mag_range[1])
    h, xedges, yedges = np.histogram2d(x[valid], y[valid], bins=bins, range=[dm_range, mag_range])
    return h.T, xedges, yedges, int(valid.sum())


def _smooth_along_delta(h: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, h)
    return out


def plot_fig2_3_residuals(dp2: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    """Data/model/residual 3x3 display for u/r/y using smoothed row models."""

    set_paper_style()
    bands = ("u", "r", "y")
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZES["3x3"])
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.91, hspace=0.36, wspace=0.34)
    rows = []
    for row_idx, band in enumerate(bands):
        data, xedges, yedges, n = _hist2d_density(dp2, band)
        model = _smooth_along_delta(data)
        resid = data - model
        vmax = np.nanpercentile(data, 99.5) if np.isfinite(data).any() else 1
        rlim = np.nanpercentile(np.abs(resid), 99) if np.isfinite(resid).any() else 1
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        for col_idx, (arr, title, cmap, vmin, vmax_i) in enumerate(
            [
                (data, "data", "cividis", 0, vmax),
                (model, "smoothed row model", "cividis", 0, vmax),
                (resid, "data - model", "RdBu_r", -rlim, rlim),
            ]
        ):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax_i)
            ax.invert_yaxis()
            ax.set_title(f"{band}: {title}")
            ax.set_xlabel(f"{band} psfMag - CModelMag")
            ax.set_ylabel(f"{band} CModel mag")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        rows.append({"band": band, "N_used": n, "model": "5-bin smoothed histogram rows", "notes": "display residual diagnostic; no Eq.54 prior calibration"})
    fig.suptitle("Fig 2.3 COSMOS u/r/y data-model residual display", y=0.975, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def plot_fig2_4_ps_map(dp2_ps: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    use = dp2_ps.loc[paper_mag_mask(dp2_ps, "cmodel_mag_r")].copy()
    x = pd.to_numeric(use["psf_minus_cmodel_r"], errors="coerce")
    y = pd.to_numeric(use["cmodel_mag_r"], errors="coerce")
    z = pd.to_numeric(use["pS_r"], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & x.gt(-0.25) & x.lt(1.2)
    xbins = np.linspace(-0.25, 1.2, 111)
    ybins = np.linspace(16, 26, 91)
    ix = np.clip(np.digitize(x[valid], xbins) - 1, 0, len(xbins) - 2)
    iy = np.clip(np.digitize(y[valid], ybins) - 1, 0, len(ybins) - 2)
    pix = pd.DataFrame({"ix": ix, "iy": iy, "pS": z[valid].to_numpy()})
    img = np.full((len(ybins) - 1, len(xbins) - 1), np.nan)
    for (iyv, ixv), val in pix.groupby(["iy", "ix"])["pS"].median().items():
        img[iyv, ixv] = val
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.86)
    im = ax.imshow(img, origin="lower", extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect="auto", cmap="coolwarm_r", vmin=0, vmax=1)
    ax.invert_yaxis()
    ax.set_xlabel("r psfMag - CModelMag")
    ax.set_ylabel("uncorrected r CModel magnitude")
    ax.set_title("Fig 2.4 COSMOS r-band v8 pS map, 16 < rmag < 26\nno Eq. 54 prior calibration yet")
    fig.colorbar(im, ax=ax, label="median pS_r")
    summary = pd.DataFrame(
        [
            {
                "N_valid": int(valid.sum()),
                "pS_column": "pS_r",
                "x_column": "psf_minus_cmodel_r",
                "y_column": "cmodel_mag_r",
                "prior_calibrated": False,
                "notes": "Existing v8 parquet has pS columns only; no logL_star/logL_galaxy components found.",
            }
        ]
    )
    return save_figure(fig, output_png), summary


def plot_fig2_5_ps_vs_extendedness(matched_ps: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    truth = pd.to_numeric(matched_ps["truth_binary"], errors="coerce")
    rmag = pd.to_numeric(matched_ps["dp2_cmodel_mag_r"], errors="coerce")
    pS = pd.to_numeric(matched_ps["pS_r"], errors="coerce")
    pred_star, _, valid_ext = class_masks(matched_ps, "dp2_extendedness_r")
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.80, wspace=0.24)
    rows = []
    for ax, (lo, hi) in zip(axes, PERFORMANCE_BINS):
        in_bin = rmag.gt(lo) & rmag.lt(hi) & truth.isin([0, 1])
        y = truth[in_bin].astype(int)
        score = pS[in_bin]
        ax.plot([0, 1], [0, 1], color="0.65", ls="--", lw=1.0)
        n_star = int((y == 1).sum())
        n_gal = int((y == 0).sum())
        if n_star and n_gal and np.isfinite(score).any():
            fpr, tpr, auc = compute_roc(y, score)
            ax.plot(fpr, tpr, color=COLORS["star"], lw=2, label=f"pS_r AUC={auc:.3f}")
        else:
            auc = np.nan
        ext_valid = in_bin & valid_ext
        ext_metrics = binary_operating_metrics(truth[ext_valid], pred_star[ext_valid])
        ax.scatter([ext_metrics["galaxy_false_positive_rate"]], [ext_metrics["star_completeness"]], color=COLORS["galaxy"], s=45, label=f"r extendedness point")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_star={n_star:,}, N_gal={n_gal:,}")
        ax.legend(loc="lower right", frameon=True)
        rows.append(
            {
                "mag_low": lo,
                "mag_high": hi,
                "N_star": n_star,
                "N_galaxy": n_gal,
                "pS_auc": auc,
                "extendedness_star_completeness": ext_metrics["star_completeness"],
                "extendedness_galaxy_false_positive_rate": ext_metrics["galaxy_false_positive_rate"],
                "extendedness_star_purity": ext_metrics["star_purity"],
                "threshold_pS": "ROC all thresholds",
                "notes": "draft/no Eq. 54 prior calibration yet",
            }
        )
    fig.suptitle("Fig 2.5 COSMOS r-band pS vs r extendedness performance\nno Eq. 54 prior calibration yet", y=0.965, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def compute_combined_scores_by_bin(matched_ps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = matched_ps.copy()
    rmag = pd.to_numeric(out["dp2_cmodel_mag_r"], errors="coerce")
    density_rows = []
    for combo, bands in COMBINATIONS.items():
        out[f"score_{combo}"] = np.nan
        out[f"n_bands_{combo}"] = 0
    for lo, hi in PERFORMANCE_BINS:
        in_bin = rmag.gt(lo) & rmag.lt(hi)
        train = out.loc[in_bin]
        for combo, bands in COMBINATIONS.items():
            score, n_used, diagnostics = density_score_train_apply(train, train, bands)
            out.loc[train.index, f"score_{combo}"] = score
            out.loc[train.index, f"n_bands_{combo}"] = n_used
            for row in diagnostics:
                row.update({"combination": combo, "mag_low": lo, "mag_high": hi, "cross_validation_used": False, "notes": "same-bin empirical likelihood draft/no Eq.54 prior calibration yet"})
                density_rows.append(row)
    return out, pd.DataFrame(density_rows)


def plot_fig2_6_multiband(matched_ps: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    set_paper_style()
    scored, density = compute_combined_scores_by_bin(matched_ps)
    truth = pd.to_numeric(scored["truth_binary"], errors="coerce")
    rmag = pd.to_numeric(scored["dp2_cmodel_mag_r"], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.80, wspace=0.24)
    colors = {"r": COLORS["star"], "gri": "#ff7f0e", "ugrizy": "#2ca02c"}
    rows = []
    for ax, (lo, hi) in zip(axes, PERFORMANCE_BINS):
        in_bin = rmag.gt(lo) & rmag.lt(hi) & truth.isin([0, 1])
        y = truth[in_bin].astype(int)
        ax.plot([0, 1], [0, 1], color="0.65", ls="--", lw=1.0)
        for combo, bands in COMBINATIONS.items():
            score = pd.to_numeric(scored.loc[in_bin, f"score_{combo}"], errors="coerce")
            valid = np.isfinite(score)
            n_star = int((y[valid] == 1).sum())
            n_gal = int((y[valid] == 0).sum())
            if n_star and n_gal:
                fpr, tpr, auc = compute_roc(y[valid], score[valid])
                ax.plot(fpr, tpr, color=colors[combo], lw=2, label=f"{'+'.join(bands)} AUC={auc:.3f}")
            else:
                auc = np.nan
            rows.append(
                {
                    "method": "empirical_likelihood_no_eq54_prior",
                    "combination": combo,
                    "bands": "+".join(bands),
                    "mag_low": lo,
                    "mag_high": hi,
                    "N_star": n_star,
                    "N_galaxy": n_gal,
                    "AUC": auc,
                    "notes": "same-bin empirical likelihood; no Eq. 54 prior calibration yet",
                }
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"{lo:g} < rmag < {hi:g}")
        ax.legend(loc="lower right", frameon=True)
    fig.suptitle("Fig 2.6 COSMOS multiband empirical likelihood performance\nno Eq. 54 prior calibration yet", y=0.965, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows), density


def plot_fig2_7_method_color_color(scored: pd.DataFrame, output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZES["2x4"])
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.84, hspace=0.50, wspace=0.30)
    specs = [
        ("color_ug", "color_gr", "dust-corrected u-g", "dust-corrected g-r", (16, 25), ("ug", "gr")),
        ("color_ug", "color_gr", "dust-corrected u-g", "dust-corrected g-r", (25, 26), ("ug", "gr")),
        ("color_gr", "color_ri", "dust-corrected g-r", "dust-corrected r-i", (16, 25), ("gr", "ri")),
        ("color_gr", "color_ri", "dust-corrected g-r", "dust-corrected r-i", (25, 26), ("gr", "ri")),
        ("color_ri", "color_iz", "dust-corrected r-i", "dust-corrected i-z", (16, 25), ("ri", "iz")),
        ("color_ri", "color_iz", "dust-corrected r-i", "dust-corrected i-z", (25, 26), ("ri", "iz")),
        ("color_iz", "color_zy", "dust-corrected i-z", "dust-corrected z-y", (16, 25), ("iz", "zy")),
        ("color_iz", "color_zy", "dust-corrected i-z", "dust-corrected z-y", (25, 26), ("iz", "zy")),
    ]
    rmag = pd.to_numeric(scored["dp2_cmodel_mag_r"], errors="coerce")
    score = pd.to_numeric(scored["score_ugrizy"], errors="coerce")
    method_star = score.ge(0)
    method_gal = score.lt(0)
    rows = []
    for ax, (x_col, y_col, x_label, y_label, mag_range, limit_key) in zip(axes.flat, specs):
        lo, hi = mag_range
        finite = rmag.gt(lo) & rmag.lt(hi) & np.isfinite(scored[x_col]) & np.isfinite(scored[y_col]) & np.isfinite(score)
        gal = scored.loc[finite & method_gal]
        star = scored.loc[finite & method_star]
        gal_plot = downsample_frame(gal, 80000)
        star_plot = downsample_frame(star, 30000)
        ax.scatter(gal_plot[x_col], gal_plot[y_col], s=2, c=COLORS["galaxy"], alpha=0.10, linewidths=0)
        ax.scatter(star_plot[x_col], star_plot[y_col], s=3, c=COLORS["star"], alpha=0.40, linewidths=0)
        xlim, ylim = COLOR_COLOR_LIMITS[limit_key]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_unres={len(star):,}, N_res={len(gal):,}")
        rows.append(
            {
                "x_column": x_col,
                "y_column": y_col,
                "mag_low": lo,
                "mag_high": hi,
                "N_method_star": int(len(star)),
                "N_method_galaxy": int(len(gal)),
                "method": "ugrizy empirical likelihood, star if logL_star-logL_galaxy >= 0",
                "notes": "no Eq. 54 prior calibration yet",
            }
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="resolved"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="unresolved"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.suptitle("Fig 2.7 COSMOS ugrizy empirical-likelihood method color-color diagrams\nno Eq. 54 prior calibration yet", y=0.965, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)

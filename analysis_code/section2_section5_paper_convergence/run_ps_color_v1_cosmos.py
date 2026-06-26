"""Create first-pass COSMOS pS_color_v1 output and diagnostics.

This uses the existing trained color-only Random Forest pickle.  The model
features are the dust-corrected colors [ug, gr, ri, iz, zy] in that exact order.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, save_figure, set_paper_style


FEATURES = ["ug", "gr", "ri", "iz", "zy"]
RMAG_COL = "cmodel_mag_r"
PS_COL = "pS_color"
VERSION = "v1"
MAG_BINS = [(16.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
METHOD_MAG_BINS = [(16.0, 25.0), (25.0, 26.0)]
COLOR_PLANES = [
    ("ug", "gr", "u-g", "g-r", ("ug", "gr")),
    ("gr", "ri", "g-r", "r-i", ("gr", "ri")),
    ("ri", "iz", "r-i", "i-z", ("ri", "iz")),
    ("iz", "zy", "i-z", "z-y", ("iz", "zy")),
]


def _ensure_missing(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing pS_color_v1 outputs:\n{joined}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _truth_star_mask(df: pd.DataFrame) -> pd.Series:
    label = df["truth_label"].astype("string").str.lower()
    return label.eq("star")


def _truth_gal_mask(df: pd.DataFrame) -> pd.Series:
    label = df["truth_label"].astype("string").str.lower()
    return label.eq("galaxy")


def _compute_star_metrics(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float | int | str]:
    valid_truth = _truth_star_mask(df) | _truth_gal_mask(df)
    valid = df.loc[valid_truth & np.isfinite(df[PS_COL])].copy()
    if valid.empty:
        return {
            "N_eval": 0,
            "N_star": 0,
            "N_galaxy": 0,
            "TP_star": 0,
            "FP_star": 0,
            "TN_star": 0,
            "FN_star": 0,
            "AUC_star": np.nan,
            "completeness_star": np.nan,
            "contamination_star": np.nan,
            "purity_star": np.nan,
            "notes": "No finite truth-labeled pS_color rows.",
        }
    y_true = _truth_star_mask(valid).astype(int).to_numpy()
    score = pd.to_numeric(valid[PS_COL], errors="coerce").to_numpy()
    pred_star = score > threshold
    truth_star = y_true == 1
    tp = int(np.sum(truth_star & pred_star))
    fp = int(np.sum(~truth_star & pred_star))
    tn = int(np.sum(~truth_star & ~pred_star))
    fn = int(np.sum(truth_star & ~pred_star))
    n_star = int(np.sum(truth_star))
    n_gal = int(np.sum(~truth_star))
    auc = float(roc_auc_score(y_true, score)) if n_star > 0 and n_gal > 0 else np.nan
    selected = tp + fp
    return {
        "N_eval": int(len(valid)),
        "N_star": n_star,
        "N_galaxy": n_gal,
        "TP_star": tp,
        "FP_star": fp,
        "TN_star": tn,
        "FN_star": fn,
        "AUC_star": auc,
        "completeness_star": tp / (tp + fn) if (tp + fn) else np.nan,
        "contamination_star": fp / selected if selected else np.nan,
        "purity_star": tp / selected if selected else np.nan,
        "notes": "Star-positive metrics at pS_color > 0.5.",
    }


def _summary_rows(ps_color: pd.DataFrame, classifier_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    paper = ps_color
    finite = paper[paper["valid_color_features"] & np.isfinite(paper[PS_COL])]
    truth_count = int((paper["truth_label"].astype("string").str.lower().isin(["star", "galaxy"])).sum())
    base = {
        "ps_color_version": VERSION,
        "classifier_path": str(classifier_path),
        "feature_list": ",".join(FEATURES),
        "label_convention": "star=1, galaxy=0; pS_color = predict_proba(X)[:, 1]",
    }
    overall = {
        **base,
        "bin_label": "16 < r < 26",
        "mag_low": 16.0,
        "mag_high": 26.0,
        "N_paper_sample": int(len(paper)),
        "N_finite_color_features": int(len(finite)),
        "N_truth_labeled": truth_count,
        **_compute_star_metrics(paper),
    }
    rows.append(overall)
    for lo, hi in MAG_BINS:
        sub = paper[paper[RMAG_COL].gt(lo) & paper[RMAG_COL].lt(hi)]
        finite_sub = sub[sub["valid_color_features"] & np.isfinite(sub[PS_COL])]
        truth_sub = int((sub["truth_label"].astype("string").str.lower().isin(["star", "galaxy"])).sum())
        rows.append(
            {
                **base,
                "bin_label": f"{lo:g} < r < {hi:g}",
                "mag_low": lo,
                "mag_high": hi,
                "N_paper_sample": int(len(sub)),
                "N_finite_color_features": int(len(finite_sub)),
                "N_truth_labeled": truth_sub,
                **_compute_star_metrics(sub),
            }
        )
    return pd.DataFrame(rows)


def _plot_truth_hist(ps_color: pd.DataFrame, output_png: Path) -> list[Path]:
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"], sharex=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.23, top=0.82, wspace=0.28)
    hist_bins = np.linspace(0, 1, 51)
    stars = _truth_star_mask(ps_color)
    gals = _truth_gal_mask(ps_color)
    for ax, (lo, hi) in zip(axes.flat, MAG_BINS):
        in_bin = ps_color[RMAG_COL].gt(lo) & ps_color[RMAG_COL].lt(hi)
        star_vals = pd.to_numeric(ps_color.loc[in_bin & stars, PS_COL], errors="coerce").dropna()
        gal_vals = pd.to_numeric(ps_color.loc[in_bin & gals, PS_COL], errors="coerce").dropna()
        if len(gal_vals):
            ax.hist(gal_vals, bins=hist_bins, density=True, histtype="step", lw=1.8, color=COLORS["galaxy"], label="COSMOS2020 galaxy")
        if len(star_vals):
            ax.hist(star_vals, bins=hist_bins, density=True, histtype="step", lw=1.8, color=COLORS["star"], label="COSMOS2020 star")
        ax.axvline(0.5, color=COLORS["threshold"], ls="--", lw=1.2, label="pS_color = 0.5")
        ax.set_xlim(0, 1)
        ax.set_xlabel("pS_color")
        ax.set_ylabel("normalized density")
        ax.set_title(f"{lo:g} < r < {hi:g}\nN_S={len(star_vals):,}, N_G={len(gal_vals):,}")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("COSMOS color-only RF star probability by truth label", y=0.965, fontsize=16)
    return save_figure(fig, output_png)


def _plot_performance(summary: pd.DataFrame, output_png: Path) -> list[Path]:
    set_paper_style()
    bins = summary[summary["bin_label"] != "16 < r < 26"].copy()
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.plot(x, bins["AUC_star"], marker="o", lw=2.0, label="AUC")
    ax.plot(x, bins["completeness_star"], marker="o", lw=2.0, label="completeness")
    ax.plot(x, bins["contamination_star"], marker="o", lw=2.0, label="contamination")
    ax.plot(x, bins["purity_star"], marker="o", lw=2.0, label="purity")
    ax.set_xticks(x, bins["bin_label"].tolist())
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("metric value")
    ax.set_xlabel("r CModel magnitude bin")
    ax.set_title("pS_color v1 star-positive performance")
    ax.legend(loc="best", frameon=True)
    return save_figure(fig, output_png)


def _plot_method_color_color(ps_color: pd.DataFrame, output_png: Path) -> list[Path]:
    set_paper_style()
    fig, axes = plt.subplots(4, 2, figsize=FIG_SIZES["4x2"])
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.075, top=0.97, hspace=0.64, wspace=0.30)
    class_star = pd.to_numeric(ps_color[PS_COL], errors="coerce").ge(0.5)
    for row, (xcol, ycol, xlabel, ylabel, limit_key) in enumerate(COLOR_PLANES):
        xlim, ylim = COLOR_COLOR_LIMITS[limit_key]
        for col, (lo, hi) in enumerate(METHOD_MAG_BINS):
            ax = axes[row, col]
            in_bin = ps_color[RMAG_COL].gt(lo) & ps_color[RMAG_COL].lt(hi)
            finite = in_bin & np.isfinite(ps_color[xcol]) & np.isfinite(ps_color[ycol]) & np.isfinite(ps_color[PS_COL])
            galaxy_like = ps_color.loc[finite & ~class_star]
            star_like = ps_color.loc[finite & class_star]
            if len(galaxy_like):
                plot_gal = galaxy_like.sample(min(len(galaxy_like), 90000), random_state=42)
                ax.scatter(plot_gal[xcol], plot_gal[ycol], s=1.8, c=COLORS["galaxy"], alpha=0.10, linewidths=0, label="pS_color < 0.5")
            if len(star_like):
                plot_star = star_like.sample(min(len(star_like), 40000), random_state=42)
                ax.scatter(plot_star[xcol], plot_star[ycol], s=2.2, c=COLORS["star"], alpha=0.35, linewidths=0, label="pS_color >= 0.5")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel, labelpad=8)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_title(f"{lo:g} < r < {hi:g}\nN_starlike={len(star_like):,}, N_gallike={len(galaxy_like):,}", pad=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.018))
    return save_figure(fig, output_png)


def run(repo_root: Path) -> dict[str, list[Path]]:
    classifier_path = repo_root / "paper_convergence/notebooks/SG-COSMOS-HST-ColorRFclassifier_all.pkl"
    analysis_path = repo_root / "outputs/dp2_cosmos_analysis_table.parquet"
    matched_path = repo_root / "outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet"
    output_parquet = repo_root / "outputs/dp2_cosmos_ps_color_v1.parquet"
    figures_dir = repo_root / "paper_convergence/figures/section5_discussion"
    results_dir = repo_root / "paper_convergence/results/section5_discussion"
    docs_dir = repo_root / "paper_convergence/docs"
    summary_csv = results_dir / "ps_color_v1_summary.csv"
    summary_md = results_dir / "ps_color_v1_summary.md"
    report = docs_dir / "section5_ps_color_report.md"
    fig_hist_png = figures_dir / "fig5_2_cosmos_ps_color_truth_hist_by_rmag.png"
    fig_perf_png = figures_dir / "fig5_3_cosmos_ps_color_performance_by_rmag.png"
    fig_cc_png = figures_dir / "fig5_4_cosmos_ps_color_method_color_color_2x4.png"
    output_paths = [
        output_parquet,
        fig_hist_png,
        fig_hist_png.with_suffix(".pdf"),
        fig_perf_png,
        fig_perf_png.with_suffix(".pdf"),
        fig_cc_png,
        fig_cc_png.with_suffix(".pdf"),
        summary_csv,
        summary_md,
        report,
    ]
    for path in [classifier_path, analysis_path, matched_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    if output_parquet.exists() or summary_csv.exists() or summary_md.exists():
        if not (output_parquet.exists() and summary_csv.exists() and summary_md.exists()):
            raise FileExistsError("Found a partial pS_color_v1 output set; refusing to overwrite or guess.")
        _ensure_missing([fig_hist_png, fig_hist_png.with_suffix(".pdf"), fig_perf_png, fig_perf_png.with_suffix(".pdf"), fig_cc_png, fig_cc_png.with_suffix(".pdf"), report])
        paper = pd.read_parquet(output_parquet)
        summary = pd.read_csv(summary_csv)
        overall = summary.iloc[0]
        fig_hist = _plot_truth_hist(paper, fig_hist_png)
        fig_perf = _plot_performance(summary, fig_perf_png)
        fig_cc = _plot_method_color_color(paper, fig_cc_png)
        _write_report(repo_root, report, classifier_path, analysis_path, matched_path, output_parquet, fig_hist, fig_perf, fig_cc, summary_csv, overall)
        return {
            "outputs": [output_parquet],
            "figures": [*fig_hist, *fig_perf, *fig_cc],
            "summaries": [summary_csv, summary_md],
            "docs": [report],
        }

    _ensure_missing(output_paths)

    cols = ["object_id", "ra", "dec", RMAG_COL, *FEATURES]
    dp2 = pd.read_parquet(analysis_path, columns=cols)
    rmag = pd.to_numeric(dp2[RMAG_COL], errors="coerce")
    paper = dp2.loc[rmag.gt(16.0) & rmag.lt(26.0)].copy()
    for col in [RMAG_COL, *FEATURES]:
        paper[col] = pd.to_numeric(paper[col], errors="coerce")
    paper["valid_color_features"] = np.isfinite(paper[FEATURES]).all(axis=1)
    paper[PS_COL] = np.nan

    clf = joblib.load(classifier_path)
    valid = paper["valid_color_features"]
    x = paper.loc[valid, FEATURES].to_numpy()
    paper.loc[valid, PS_COL] = clf.predict_proba(x)[:, 1]
    paper["ps_color_version"] = VERSION

    truth = pd.read_parquet(matched_path, columns=["object_id", "truth_label", "truth_binary"])
    truth = truth.drop_duplicates("object_id", keep="first")
    paper = paper.merge(truth, on="object_id", how="left")

    out_cols = ["object_id", "ra", "dec", RMAG_COL, *FEATURES, "truth_label", "truth_binary", PS_COL, "valid_color_features", "ps_color_version"]
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    paper[out_cols].to_parquet(output_parquet, index=False)

    summary = _summary_rows(paper, classifier_path.relative_to(repo_root))
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    overall = summary.iloc[0]
    _write_text(
        summary_md,
        "\n".join(
            [
                "# pS_color v1 Summary",
                "",
                f"- Derived output: `{output_parquet.relative_to(repo_root)}`",
                f"- Classifier: `{classifier_path.relative_to(repo_root)}`",
                f"- Features: `{FEATURES}`",
                "- Feature order is fixed as `ug, gr, ri, iz, zy`.",
                "- Classifier is color-only; no morphology features are used.",
                "- Label convention: star = 1, galaxy = 0; `pS_color = predict_proba(X)[:, 1]`.",
                f"- Total paper sample: {int(overall['N_paper_sample']):,}",
                f"- Finite color-feature rows: {int(overall['N_finite_color_features']):,}",
                f"- Matched truth-label rows: {int(overall['N_truth_labeled']):,}",
                f"- Overall AUC where truth labels are available: {overall['AUC_star']:.4f}",
                "",
                "Star-positive metrics at `pS_color > 0.5` are recorded in the CSV for each r-magnitude bin.",
            ]
        )
        + "\n",
    )

    fig_hist = _plot_truth_hist(paper, fig_hist_png)
    fig_perf = _plot_performance(summary, fig_perf_png)
    fig_cc = _plot_method_color_color(paper, fig_cc_png)

    _write_report(repo_root, report, classifier_path, analysis_path, matched_path, output_parquet, fig_hist, fig_perf, fig_cc, summary_csv, overall)

    return {
        "outputs": [output_parquet],
        "figures": [*fig_hist, *fig_perf, *fig_cc],
        "summaries": [summary_csv, summary_md],
        "docs": [report],
    }


def _write_report(
    repo_root: Path,
    report: Path,
    classifier_path: Path,
    analysis_path: Path,
    matched_path: Path,
    output_parquet: Path,
    fig_hist: list[Path],
    fig_perf: list[Path],
    fig_cc: list[Path],
    summary_csv: Path,
    overall: pd.Series,
) -> None:
    _write_text(
        report,
        "\n".join(
            [
                "# Section 5 pS_color v1 Diagnostic Report",
                "",
                "`pS_color` is a color-only Random Forest star probability.",
                "",
                "Inputs and provenance:",
                f"- Classifier pickle: `{classifier_path.relative_to(repo_root)}`",
                f"- DP2 analysis table: `{analysis_path.relative_to(repo_root)}`",
                f"- COSMOS2020 matched labels for evaluation only: `{matched_path.relative_to(repo_root)}`",
                f"- Features: `{', '.join(FEATURES)}`",
                "- The classifier uses only dust-corrected colors and no morphology features.",
                "- Label convention: star = 1, galaxy = 0; the score is `predict_proba(X)[:, 1]`.",
                "",
                "Outputs:",
                f"- Derived pS_color table: `{output_parquet.relative_to(repo_root)}`",
                f"- Truth-label histogram: `{fig_hist[0].relative_to(repo_root)}`",
                f"- Performance diagnostic: `{fig_perf[0].relative_to(repo_root)}`",
                f"- Method color-color diagnostic: `{fig_cc[0].relative_to(repo_root)}`",
                f"- Summary CSV: `{summary_csv.relative_to(repo_root)}`",
                "",
                "Performance summary:",
                f"- Total paper sample: {int(overall['N_paper_sample']):,}",
                f"- Finite color-feature rows: {int(overall['N_finite_color_features']):,}",
                f"- Matched truth-label rows: {int(overall['N_truth_labeled']):,}",
                f"- Overall AUC: {overall['AUC_star']:.4f}",
                f"- Overall completeness at pS_color > 0.5: {overall['completeness_star']:.4f}",
                f"- Overall contamination at pS_color > 0.5: {overall['contamination_star']:.4f}",
                f"- Overall purity at pS_color > 0.5: {overall['purity_star']:.4f}",
                "",
                "Caveats:",
                "- This is an exploratory notebook-level classifier made reproducible as a first-pass derived output.",
                "- This is a color-only score, not morphology pS.",
                "- This is not an Eq.54 morphology posterior.",
                "- COSMOS2020 truth labels are used only for evaluation in this output.",
                "- The pickle was saved with a different scikit-learn version than the current runtime; treat this as a reproducibility caveat until rerun in the original environment or retrained.",
                "- Treat these figures as Section 5 / discussion diagnostics unless further validated.",
            ]
        )
        + "\n",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    outputs = run(root)
    for group, paths in outputs.items():
        print(group)
        for path in paths:
            print(" ", path.relative_to(root))

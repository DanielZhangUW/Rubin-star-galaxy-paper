"""Generate ROC/AUC diagnostics for Eq.54 + pS_color Section 2 scores.

This runner reads only existing derived outputs. It does not modify source
parquet files and writes new Fig 2.8 / Fig 2.9 files with distinct names.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import auc, roc_curve
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for ROC/AUC diagnostics") from exc


BINS = [
    ("16 < r < 24", 16.0, 24.0),
    ("24 < r < 25", 24.0, 25.0),
    ("25 < r < 26", 25.0, 26.0),
]

COLOR_SCORES = [
    ("r + color", "pS_r_eq54prior_color", "#1f77b4"),
    ("gri + color", "pS_gri_eq54prior_color", "#ff7f0e"),
    ("ugrizy + color", "pS_ugrizy_eq54prior_color", "#2ca02c"),
]

R_COMPARISON_SCORES = [
    ("Eq.54 r + color", "pS_r_eq54prior_color", "#1f77b4"),
    ("Eq.54 morphology r", "pS_r_eq54prior", "#222222"),
]

MORPHOLOGY_REFERENCE_SCORES = [
    ("r morphology", "pS_r_eq54prior"),
    ("gri morphology", "pS_gri_eq54prior"),
    ("ugrizy morphology", "pS_ugrizy_eq54prior"),
]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    input_parquet: Path
    previous_performance_csv: Path
    figure_dir: Path
    result_dir: Path
    doc_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "Paths":
        return cls(
            repo_root=repo_root,
            input_parquet=repo_root / "outputs" / "dp2_cosmos_ps_v9_eq54prior_color.parquet",
            previous_performance_csv=repo_root
            / "paper_convergence"
            / "results"
            / "section2_bayesian_method"
            / "eq54_color_integration_performance_by_rmag.csv",
            figure_dir=repo_root
            / "paper_convergence"
            / "figures"
            / "section2_bayesian_method",
            result_dir=repo_root
            / "paper_convergence"
            / "results"
            / "section2_bayesian_method",
            doc_dir=repo_root / "paper_convergence" / "docs",
        )


def sigmoid_safe(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_v = np.exp(values[~positive])
    out[~positive] = exp_v / (1.0 + exp_v)
    return out


def threshold_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    pred = score >= threshold
    truth = y_true == 1
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    completeness = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    contamination = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    purity = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "completeness_at_pS_ge_0p5": completeness,
        "contamination_at_pS_ge_0p5": contamination,
        "purity_at_pS_ge_0p5": purity,
    }


def compute_roc_summary(df: pd.DataFrame, score_specs: Iterable[tuple[str, str]], include_curves: bool = False):
    rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for bin_label, low, high in BINS:
        in_bin = (df["cmodel_mag_r"] > low) & (df["cmodel_mag_r"] < high)
        for score_label, score_col in score_specs:
            sub = df.loc[in_bin, ["truth_binary", score_col]].copy()
            sub = sub[np.isfinite(sub["truth_binary"]) & np.isfinite(sub[score_col])]
            sub = sub[sub["truth_binary"].isin([0, 1])]
            y_true = sub["truth_binary"].to_numpy(dtype=int)
            score = sub[score_col].to_numpy(dtype=float)
            n_star = int(np.sum(y_true == 1))
            n_galaxy = int(np.sum(y_true == 0))
            row: dict[str, object] = {
                "magnitude_bin": bin_label,
                "mag_low": low,
                "mag_high": high,
                "score_label": score_label,
                "score_column": score_col,
                "N_valid": len(sub),
                "N_star": n_star,
                "N_galaxy": n_galaxy,
            }
            if n_star > 0 and n_galaxy > 0:
                fpr, tpr, _ = roc_curve(y_true, score, pos_label=1)
                row["AUC"] = auc(fpr, tpr)
                row["roc_computed"] = True
                row["not_computed_reason"] = ""
                row.update(threshold_metrics(y_true, score))
                if include_curves:
                    curves[(bin_label, score_col)] = (fpr, tpr)
            else:
                row["AUC"] = np.nan
                row["roc_computed"] = False
                row["not_computed_reason"] = "one class missing"
                row.update(
                    {
                        "TP": np.nan,
                        "FP": np.nan,
                        "TN": np.nan,
                        "FN": np.nan,
                        "completeness_at_pS_ge_0p5": np.nan,
                        "contamination_at_pS_ge_0p5": np.nan,
                        "purity_at_pS_ge_0p5": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), curves


def setup_ax(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], color="0.65", lw=1.2, ls="--", label="random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("false positive rate / galaxy contamination", fontsize=13)
    ax.set_ylabel("true positive rate / star completeness", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, color="0.85", lw=0.8, alpha=0.7)


def make_fig2_8(summary: pd.DataFrame, curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], paths: Paths) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (bin_label, _, _) in zip(axes, BINS):
        setup_ax(ax, bin_label)
        for score_label, score_col, color in COLOR_SCORES:
            row = summary[(summary["magnitude_bin"] == bin_label) & (summary["score_column"] == score_col)]
            if row.empty or not bool(row.iloc[0]["roc_computed"]):
                continue
            fpr, tpr = curves[(bin_label, score_col)]
            auc_value = float(row.iloc[0]["AUC"])
            n_star = int(row.iloc[0]["N_star"])
            n_gal = int(row.iloc[0]["N_galaxy"])
            ax.plot(fpr, tpr, lw=2.0, color=color, label=f"{score_label} AUC={auc_value:.3f} (S={n_star}, G={n_gal})")
        ax.legend(fontsize=8, loc="lower right", frameon=True)
    for suffix in ["png", "pdf"]:
        out = paths.figure_dir / f"fig2_8_cosmos_eq54_color_roc_3bins.{suffix}"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {out}")
        fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_fig2_9(summary: pd.DataFrame, curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], paths: Paths) -> pd.DataFrame:
    extendedness_rows = pd.DataFrame()
    if paths.previous_performance_csv.exists():
        previous = pd.read_csv(paths.previous_performance_csv)
        extendedness_rows = previous[previous["method"] == "r_extendedness"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ext_summary_rows: list[dict[str, object]] = []
    for ax, (bin_label, low, high) in zip(axes, BINS):
        setup_ax(ax, bin_label)
        for score_label, score_col, color in R_COMPARISON_SCORES:
            row = summary[(summary["magnitude_bin"] == bin_label) & (summary["score_column"] == score_col)]
            if row.empty or not bool(row.iloc[0]["roc_computed"]):
                continue
            fpr, tpr = curves[(bin_label, score_col)]
            auc_value = float(row.iloc[0]["AUC"])
            ax.plot(fpr, tpr, lw=2.0, color=color, label=f"{score_label} AUC={auc_value:.3f}")

        if {"mag_low", "mag_high"}.issubset(extendedness_rows.columns):
            ext = extendedness_rows[
                np.isclose(extendedness_rows["mag_low"].astype(float), low)
                & np.isclose(extendedness_rows["mag_high"].astype(float), high)
            ]
        else:
            ext = extendedness_rows[extendedness_rows["magnitude_bin"] == bin_label]
        if not ext.empty:
            erow = ext.iloc[0]
            tp = float(erow["TP"])
            fp = float(erow["FP"])
            fn = float(erow["FN"])
            tn = float(erow["TN"])
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            contamination = fp / (tp + fp) if (tp + fp) > 0 else np.nan
            purity = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            ax.plot(
                [0.0, fpr, 1.0],
                [0.0, tpr, 1.0],
                color="#d62728",
                lw=2.0,
                ls="-.",
                label="r extendedness",
            )
            ext_summary_rows.append(
                {
                    "magnitude_bin": bin_label,
                    "score_label": "r extendedness operating point",
                    "score_column": "dp2_extendedness_r",
                    "N_valid": int(erow["N_valid"]),
                    "N_star": int(erow["N_star"]),
                    "N_galaxy": int(erow["N_galaxy"]),
                    "FPR": fpr,
                    "TPR": tpr,
                    "contamination": contamination,
                    "purity": purity,
                    "TP": int(tp),
                    "FP": int(fp),
                    "TN": int(tn),
                    "FN": int(fn),
                }
            )
        ax.legend(fontsize=8, loc="lower right", frameon=True)
    for suffix in ["png", "pdf"]:
        out = paths.figure_dir / f"fig2_9_cosmos_pS_vs_extendedness_roc_3bins_eq54prior_color.{suffix}"
        if out.exists() and os.environ.get("OVERWRITE_CURRENT_ROC_OUTPUTS") != "1":
            raise FileExistsError(f"Refusing to overwrite existing file: {out}")
        fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(ext_summary_rows)


def write_report(paths: Paths, fig2_8_summary: pd.DataFrame, fig2_9_summary: pd.DataFrame, used_extendedness: bool) -> None:
    lines = [
        "# Section 2 ROC/AUC Report",
        "",
        "This report accompanies the Fig 2.8 and Fig 2.9 ROC/AUC diagnostics for the current Eq.54 + pS_color outputs.",
        "",
        "## Purpose",
        "",
        "Existing Fig 2.5 and Fig 2.6 summarize fixed-threshold performance at `pS > 0.5`. ROC/AUC is threshold-independent and is useful because the Eq.54 magnitude prior makes `pS > 0.5` very conservative at the faint end.",
        "",
        "AUC should be used to assess ranking performance. Threshold metrics show one operating point.",
        "",
        "## Inputs",
        "",
        f"- Derived pS table: `{paths.input_parquet.relative_to(paths.repo_root)}`",
        "- Positive class: COSMOS2020 truth star (`truth_binary = 1`).",
        "- Negative class: COSMOS2020 truth galaxy (`truth_binary = 0`).",
        "- Magnitude column: `cmodel_mag_r`.",
        "",
        "## Figures",
        "",
        "- Fig 2.8: ROC curves for `pS_r_eq54prior_color`, `pS_gri_eq54prior_color`, and `pS_ugrizy_eq54prior_color`.",
        "- Fig 2.9: ROC curves for `pS_r_eq54prior_color` and morphology-only `pS_r_eq54prior`, with the r-band extendedness operating point where available.",
        "",
        "## Extendedness Operating Point",
        "",
        (
            "The r-band extendedness operating point was read from the existing derived performance summary "
            f"`{paths.previous_performance_csv.relative_to(paths.repo_root)}`."
            if used_extendedness
            else "The r-band extendedness operating point was not included because no derived operating-point summary was available."
        ),
        "",
        "## Caveats",
        "",
        "- These diagnostics do not retrain or recalibrate `pS_color`.",
        "- Existing Eq.54 and color-integrated parquet inputs were not modified.",
        "- Fig 2.8/2.9 use FPR on the x-axis for ROC consistency; fixed-threshold contamination/purity are recorded in the CSV summaries.",
        "",
        "## Summary Rows",
        "",
        f"- Fig 2.8 summary rows: {len(fig2_8_summary)}",
        f"- Fig 2.9 summary rows, including extendedness rows if present: {len(fig2_9_summary)}",
    ]
    out = paths.doc_dir / "section2_roc_auc_report.md"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {out}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(repo_root: Path | None = None) -> dict[str, Path]:
    repo_root = (repo_root or Path.cwd()).resolve()
    paths = Paths.from_repo_root(repo_root)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    paths.doc_dir.mkdir(parents=True, exist_ok=True)

    required_columns = [
        "truth_binary",
        "cmodel_mag_r",
        "pS_r_eq54prior_color",
        "pS_gri_eq54prior_color",
        "pS_ugrizy_eq54prior_color",
        "pS_r_eq54prior",
        "pS_gri_eq54prior",
        "pS_ugrizy_eq54prior",
    ]
    df = pd.read_parquet(paths.input_parquet, columns=required_columns)

    fig2_8_scores = [(label, col) for label, col, _ in COLOR_SCORES] + MORPHOLOGY_REFERENCE_SCORES
    fig2_8_summary_all, fig2_8_curves_all = compute_roc_summary(df, fig2_8_scores, include_curves=True)
    fig2_8_summary = fig2_8_summary_all[fig2_8_summary_all["score_column"].isin([col for _, col, _ in COLOR_SCORES])].copy()

    fig2_8_summary_out = paths.result_dir / "fig2_8_cosmos_eq54_color_roc_3bins_summary.csv"
    fig2_9_summary_out = paths.result_dir / "fig2_9_cosmos_pS_vs_extendedness_roc_3bins_eq54prior_color_summary.csv"
    for out in [
        fig2_8_summary_out,
        fig2_9_summary_out,
        paths.doc_dir / "section2_roc_auc_report.md",
    ]:
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {out}")

    make_fig2_8(fig2_8_summary_all, fig2_8_curves_all, paths)
    fig2_8_summary.to_csv(fig2_8_summary_out, index=False)

    fig2_9_scores = [(label, col) for label, col, _ in R_COMPARISON_SCORES]
    fig2_9_summary, fig2_9_curves = compute_roc_summary(df, fig2_9_scores, include_curves=True)
    ext_summary = make_fig2_9(fig2_9_summary, fig2_9_curves, paths)
    if not ext_summary.empty:
        ext_summary = ext_summary.assign(AUC=np.nan, roc_computed=False, not_computed_reason="binary operating point")
        fig2_9_summary = pd.concat([fig2_9_summary, ext_summary], ignore_index=True, sort=False)
    fig2_9_summary.to_csv(fig2_9_summary_out, index=False)

    write_report(paths, fig2_8_summary, fig2_9_summary, used_extendedness=not ext_summary.empty)

    return {
        "fig2_8_png": paths.figure_dir / "fig2_8_cosmos_eq54_color_roc_3bins.png",
        "fig2_8_pdf": paths.figure_dir / "fig2_8_cosmos_eq54_color_roc_3bins.pdf",
        "fig2_9_png": paths.figure_dir / "fig2_9_cosmos_pS_vs_extendedness_roc_3bins_eq54prior_color.png",
        "fig2_9_pdf": paths.figure_dir / "fig2_9_cosmos_pS_vs_extendedness_roc_3bins_eq54prior_color.pdf",
        "fig2_8_summary": fig2_8_summary_out,
        "fig2_9_summary": fig2_9_summary_out,
        "report": paths.doc_dir / "section2_roc_auc_report.md",
    }


if __name__ == "__main__":
    outputs = main()
    for key, path in outputs.items():
        print(f"{key}: {path}")

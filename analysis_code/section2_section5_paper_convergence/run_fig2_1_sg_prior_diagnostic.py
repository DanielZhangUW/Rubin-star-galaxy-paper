"""Generate the lightweight Fig 2.1 S/G prior-vs-rmag diagnostic."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_priors import log10_ng_over_ns_prior_from_rmag


BINS = np.arange(16.0, 26.0001, 0.5)
V9_RATIO_RELATIVE_PATH = Path("paper_convergence/tables/v9_ratio_data.csv")


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_required_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _first_existing(columns: set[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"none of these columns exist: {candidates}")


def _ratio_row(
    source_name: str,
    metric_type: str,
    mag_low: float,
    mag_high: float,
    mag_center: float,
    magnitude_column: str,
    label_column: str,
    n_star: int | float,
    n_galaxy: int | float,
    notes: str = "",
) -> dict:
    if n_star and n_galaxy and n_star > 0 and n_galaxy > 0:
        log10_ratio = float(np.log10(n_galaxy / n_star))
        skipped_reason = ""
    else:
        log10_ratio = np.nan
        skipped_reason = "N_star=0 or N_galaxy=0"
    return {
        "source_name": source_name,
        "metric_type": metric_type,
        "mag_low": mag_low,
        "mag_high": mag_high,
        "mag_center": mag_center,
        "magnitude_column": magnitude_column,
        "label_column": label_column,
        "N_star": int(n_star) if np.isfinite(n_star) else np.nan,
        "N_galaxy": int(n_galaxy) if np.isfinite(n_galaxy) else np.nan,
        "N_total": int(n_star + n_galaxy) if np.isfinite(n_star) and np.isfinite(n_galaxy) else np.nan,
        "log10_NG_over_NS": log10_ratio,
        "fraction_value": np.nan,
        "fraction_definition": "",
        "skipped_reason": skipped_reason,
        "notes": notes,
    }


def _coverage_row(
    source_name: str,
    mag_low: float,
    mag_high: float,
    mag_center: float,
    magnitude_column: str,
    n_matched: int,
    n_dp2_only: int,
    notes: str = "",
) -> dict:
    denom = n_matched + n_dp2_only
    fraction = n_matched / denom if denom else np.nan
    return {
        "source_name": source_name,
        "metric_type": "coverage_proxy",
        "mag_low": mag_low,
        "mag_high": mag_high,
        "mag_center": mag_center,
        "magnitude_column": magnitude_column,
        "label_column": "object_id matched to COSMOS2020",
        "N_star": np.nan,
        "N_galaxy": np.nan,
        "N_total": denom,
        "log10_NG_over_NS": np.nan,
        "fraction_value": fraction,
        "fraction_definition": "N_matched_DP2_to_COSMOS2020 / (N_matched_DP2_to_COSMOS2020 + N_DP2_only)",
        "skipped_reason": "" if denom else "empty magnitude bin",
        "notes": notes,
    }


def _parse_mag_bin(mag_bin: str) -> tuple[float, float, float]:
    low_text, high_text = str(mag_bin).split("-", maxsplit=1)
    low = float(low_text)
    high = float(high_text)
    return low, high, 0.5 * (low + high)


def _v9_mixture_rows(repo_root: Path) -> tuple[list[dict], dict[str, object]]:
    """Load COSMOS r-band v9 mixture ratios, validating formula conventions."""
    path = repo_root / V9_RATIO_RELATIVE_PATH
    validation: dict[str, object] = {
        "source_path": str(V9_RATIO_RELATIVE_PATH),
        "accepted_for_fig2_1": False,
        "strict_1e_minus_6_log_formula_check_passed": False,
        "notes": "",
    }
    if not path.exists():
        validation["notes"] = "v9 ratio table missing"
        return [], validation

    table = pd.read_csv(path)
    required = {
        "field",
        "band",
        "mag_bin",
        "star_fraction_wU",
        "galaxy_fraction",
        "log10_NG_NS",
    }
    missing = sorted(required - set(table.columns))
    validation["row_count_total"] = int(len(table))
    validation["columns"] = ", ".join(table.columns)
    if missing:
        validation["notes"] = f"missing required columns: {missing}"
        return [], validation

    selected = table[table["field"].eq("COSMOS") & table["band"].eq("r")].copy()
    validation["selected_field"] = "COSMOS"
    validation["selected_band"] = "r"
    validation["selected_row_count"] = int(len(selected))
    if selected.empty:
        validation["notes"] = "no COSMOS r-band rows found"
        return [], validation

    selected["star_fraction_wU"] = pd.to_numeric(selected["star_fraction_wU"], errors="coerce")
    selected["galaxy_fraction"] = pd.to_numeric(selected["galaxy_fraction"], errors="coerce")
    selected["log10_NG_NS"] = pd.to_numeric(selected["log10_NG_NS"], errors="coerce")
    finite = selected.dropna(subset=["star_fraction_wU", "galaxy_fraction", "log10_NG_NS"]).copy()
    validation["finite_selected_rows"] = int(len(finite))
    if finite.empty:
        validation["notes"] = "COSMOS r-band rows exist but all ratio values are non-finite"
        return [], validation

    fractions_in_bounds = (
        finite["star_fraction_wU"].between(0, 1)
        & finite["galaxy_fraction"].between(0, 1)
    ).all()
    positive_fractions = (finite["star_fraction_wU"].gt(0) & finite["galaxy_fraction"].gt(0)).all()
    galaxy_from_star = 1.0 - finite["star_fraction_wU"]
    max_galaxy_resid = float(np.nanmax(np.abs(finite["galaxy_fraction"] - galaxy_from_star)))
    recomputed_log = np.log10(finite["galaxy_fraction"] / finite["star_fraction_wU"])
    max_log_resid = float(np.nanmax(np.abs(finite["log10_NG_NS"] - recomputed_log)))

    validation.update(
        {
            "fractions_in_bounds": bool(fractions_in_bounds),
            "positive_fractions": bool(positive_fractions),
            "max_abs_galaxy_fraction_minus_1_minus_wU": max_galaxy_resid,
            "max_abs_log10_formula_residual": max_log_resid,
            "strict_1e_minus_6_galaxy_fraction_check_passed": bool(max_galaxy_resid < 1e-6),
            "strict_1e_minus_6_log_formula_check_passed": bool(max_log_resid < 1e-6),
        }
    )

    if not fractions_in_bounds or not positive_fractions or max_galaxy_resid >= 1e-6:
        validation["notes"] = "rejected: invalid fractions or galaxy_fraction != 1 - star_fraction_wU"
        return [], validation

    rows: list[dict] = []
    for _, row in selected.iterrows():
        lo, hi, center = _parse_mag_bin(row["mag_bin"])
        star_fraction = row["star_fraction_wU"]
        galaxy_fraction = row["galaxy_fraction"]
        if np.isfinite(star_fraction) and np.isfinite(galaxy_fraction) and star_fraction > 0 and galaxy_fraction > 0:
            log10_ratio = float(np.log10(galaxy_fraction / star_fraction))
            skipped_reason = ""
        else:
            log10_ratio = np.nan
            skipped_reason = "non-finite or non-positive v9 mixture fraction"
        rows.append(
            {
                "source_name": "v9 mixture-model ratio",
                "metric_type": "ratio_curve",
                "mag_low": lo,
                "mag_high": hi,
                "mag_center": center,
                "magnitude_column": "r CModel magnitude bin from v9_ratio_data.csv",
                "label_column": "star_fraction_wU; galaxy_fraction=1-star_fraction_wU",
                "N_star": np.nan,
                "N_galaxy": np.nan,
                "N_total": np.nan,
                "log10_NG_over_NS": log10_ratio,
                "fraction_value": np.nan,
                "fraction_definition": "star_fraction=wU; galaxy_fraction=1-wU; plotted ratio recomputed as log10(galaxy_fraction/star_fraction)",
                "skipped_reason": skipped_reason,
                "notes": "v9 mixture-model ratio from paper_convergence/tables/v9_ratio_data.csv",
            }
        )

    validation["accepted_for_fig2_1"] = True
    validation["notes"] = (
        "accepted for Fig 2.1; plotted values are recomputed from star_fraction_wU "
        "and galaxy_fraction. The stored log10_NG_NS column is consistent to CSV "
        "rounding, but the strict 1e-6 log-column check may fail because the table "
        "stores rounded decimal values."
    )
    return rows, validation


def build_summary(repo_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    matched_path = repo_root / "outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet"
    dp2_path = repo_root / "outputs/dp2_cosmos_analysis_table.parquet"
    matched_cols = ["dp2_object_id", "object_id", "dp2_cmodel_mag_r", "truth_binary", "truth_label"]
    dp2_cols = ["object_id", "cmodel_mag_r", "extendedness_r", "r_extendedness"]

    matched = _read_required_columns(matched_path, matched_cols)
    dp2 = _read_required_columns(dp2_path, dp2_cols)

    ext_col = _first_existing(set(dp2.columns), ["extendedness_r", "r_extendedness"])
    matched_id_col = "dp2_object_id" if "dp2_object_id" in matched.columns else "object_id"

    matched_mag = pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce")
    truth = pd.to_numeric(matched["truth_binary"], errors="coerce")
    dp2_mag = pd.to_numeric(dp2["cmodel_mag_r"], errors="coerce")
    ext = pd.to_numeric(dp2[ext_col], errors="coerce")

    matched_ids = set(pd.to_numeric(matched[matched_id_col], errors="coerce").dropna().astype("int64"))
    dp2_ids = pd.to_numeric(dp2["object_id"], errors="coerce")
    dp2_only = ~dp2_ids.isin(matched_ids)

    rows: list[dict] = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        center = 0.5 * (lo + hi)

        rows.append(
            {
                "source_name": "by-eye prior",
                "metric_type": "ratio_curve",
                "mag_low": lo,
                "mag_high": hi,
                "mag_center": center,
                "magnitude_column": "r CModel magnitude",
                "label_column": "piecewise analytic function",
                "N_star": np.nan,
                "N_galaxy": np.nan,
                "N_total": np.nan,
                "log10_NG_over_NS": log10_ng_over_ns_prior_from_rmag(center, bounds_action="ignore"),
                "fraction_value": np.nan,
                "fraction_definition": "",
                "skipped_reason": "",
                "notes": "by-eye piecewise fit; higher values mean more galaxies relative to stars",
            }
        )

        in_matched_bin = matched_mag.ge(lo) & matched_mag.lt(hi)
        n_star = int((in_matched_bin & truth.eq(1)).sum())
        n_galaxy = int((in_matched_bin & truth.eq(0)).sum())
        rows.append(
            _ratio_row(
                "COSMOS2020 matched labels",
                "ratio_curve",
                lo,
                hi,
                center,
                "dp2_cmodel_mag_r",
                "truth_binary; 1=star, 0=galaxy",
                n_star,
                n_galaxy,
                "computed from existing matched COSMOS2020-DP2 table",
            )
        )

        in_dp2_bin = dp2_mag.ge(lo) & dp2_mag.lt(hi)
        n_ext_star = int((in_dp2_bin & ext.eq(0)).sum())
        n_ext_gal = int((in_dp2_bin & ext.eq(1)).sum())
        rows.append(
            _ratio_row(
                "r-band extendedness",
                "ratio_curve",
                lo,
                hi,
                center,
                "cmodel_mag_r",
                f"{ext_col}; 0=star/unresolved, 1=galaxy/resolved",
                n_ext_star,
                n_ext_gal,
                "computed from existing full COSMOS DP2 analysis table",
            )
        )

        n_matched = int((in_dp2_bin & ~dp2_only).sum())
        n_dp2_only = int((in_dp2_bin & dp2_only).sum())
        rows.append(
            _coverage_row(
                "DP2-to-COSMOS2020 matched fraction proxy",
                lo,
                hi,
                center,
                "cmodel_mag_r",
                n_matched,
                n_dp2_only,
                "COSMOS2020-only external-r curve excluded from main prior plot because HSC/FARMER r is not directly comparable to DP2 r CModel magnitude",
            )
        )

    v9_rows, v9_validation = _v9_mixture_rows(repo_root)
    rows.extend(v9_rows)
    metadata = {
        "matched_table": str(matched_path.relative_to(repo_root)),
        "dp2_table": str(dp2_path.relative_to(repo_root)),
        "v9_ratio_table": str(V9_RATIO_RELATIVE_PATH),
        "v9_ratio_accepted_for_fig2_1": str(v9_validation["accepted_for_fig2_1"]),
        "v9_ratio_max_log10_formula_residual": str(v9_validation.get("max_abs_log10_formula_residual", "")),
        "matched_magnitude_column": "dp2_cmodel_mag_r",
        "matched_label_column": "truth_binary",
        "dp2_magnitude_column": "cmodel_mag_r",
        "extendedness_column": ext_col,
        "matched_id_column": matched_id_col,
    }
    return pd.DataFrame(rows), metadata


def plot_summary(summary: pd.DataFrame, output_png: Path) -> list[Path]:
    ratio = summary[summary["metric_type"].eq("ratio_curve")].copy()
    coverage = summary[summary["metric_type"].eq("coverage_proxy")].copy()
    fig, (ax_ratio, ax_cov) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )

    styles = {
        "by-eye prior": {"color": "black", "lw": 2.6, "ls": "-", "marker": None},
        "COSMOS2020 matched labels": {"color": "#2ca02c", "lw": 1.8, "ls": "--", "marker": "o"},
        "r-band extendedness": {"color": "#9467bd", "lw": 1.6, "ls": ":", "marker": "s"},
        "v9 mixture-model ratio": {"color": "#ff7f0e", "lw": 2.0, "ls": "-.", "marker": "D"},
    }
    for source, group in ratio.groupby("source_name", sort=False):
        style = styles.get(source, {"color": "0.3", "lw": 1.3, "ls": "-", "marker": "o"})
        ax_ratio.plot(
            group["mag_center"],
            group["log10_NG_over_NS"],
            label=source,
            color=style["color"],
            lw=style["lw"],
            ls=style["ls"],
            marker=style["marker"],
            ms=4.5 if style["marker"] else 0,
        )
    ax_ratio.axhline(0, color="0.65", lw=0.9, ls="--")
    ax_ratio.set_ylabel("log10(NG/NS)")
    ax_ratio.set_xlim(16, 26)
    ax_ratio.grid(True, color="0.88", lw=0.7)
    ax_ratio.legend(loc="best", frameon=True, fontsize=9)
    ax_ratio.text(
        0.02,
        0.04,
        "higher y = more galaxies relative to stars",
        transform=ax_ratio.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.25",
    )

    ax_cov.plot(
        coverage["mag_center"],
        coverage["fraction_value"],
        color="0.25",
        marker="o",
        lw=1.5,
        ms=4.0,
        label="DP2 matched fraction proxy",
    )
    ax_cov.set_ylabel("matched fraction")
    ax_cov.set_xlabel("r CModel magnitude")
    ax_cov.set_ylim(0, 1.05)
    ax_cov.grid(True, color="0.88", lw=0.7)
    ax_cov.legend(loc="best", frameon=True, fontsize=8.5)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    pdf_path = output_png.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [output_png, pdf_path]


def write_notes(path: Path, summary: pd.DataFrame, metadata: dict[str, str]) -> None:
    availability = {
        "by-eye prior": "available from paper_priors.py",
        "v9 mixture-model ratio": "loaded from paper_convergence/tables/v9_ratio_data.csv and plotted for COSMOS r band if validation passes",
        "COSMOS2020 matched-label ratio": "computed from existing matched table",
        "r-band extendedness ratio": "computed from existing full DP2 analysis table",
        "coverage proxy": "computed from full DP2 object IDs minus matched DP2 object IDs",
        "COSMOS2020-only external-r proxy": "excluded from main plot because HSC/FARMER r is not directly comparable to DP2 r CModel magnitude",
    }
    sanity_values = pd.DataFrame(
        {
            "rmag": [23.0, 24.0, 25.0],
            "log10_NG_over_NS": log10_ng_over_ns_prior_from_rmag(
                np.array([23.0, 24.0, 25.0])
            ),
        }
    )
    lines = [
        "# Fig 2.1 S/G Prior vs r Magnitude Notes",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This lightweight diagnostic compares `log10(NG/NS)` against r-band CModel magnitude.",
        "Higher y values mean more galaxies relative to stars.",
        "",
        "## Inputs And Columns",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Available Curves",
        "",
    ]
    for key, value in availability.items():
        lines.append(f"- {key}: {value}.")
    lines += [
        "",
        "## Sanity Values From By-Eye Prior",
        "",
        sanity_values.to_csv(index=False).strip(),
        "",
        "## Coverage Proxy Definition",
        "",
        "`matched fraction = N_matched_DP2_to_COSMOS2020 / (N_matched_DP2_to_COSMOS2020 + N_DP2_only)` in each r CModel magnitude bin.",
        "",
        "The coverage proxy is recorded separately from `log10(NG/NS)` because it is a fraction, not a star/galaxy count ratio.",
        "",
        "## Scope",
        "",
        "- v8/v9 pS parquet files are not modified.",
        "- Stored `pS` values are not recalibrated.",
        "- Fig 2.4-2.7 are not regenerated.",
        "- No large output parquet is created.",
        "",
        "## v9 Mixture Ratio Convention",
        "",
        "For the v9 table, `star_fraction = star_fraction_wU = wU`, `galaxy_fraction = 1 - wU`, and the plotted value is recomputed as `log10(galaxy_fraction / star_fraction)`.",
        "The source table also contains `log10_NG_NS`; that column is used for validation only because it is stored with rounded decimal precision.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v9_validation(path: Path, repo_root: Path) -> None:
    rows, validation = _v9_mixture_rows(repo_root)
    lines = [
        "# v9 Mixture Ratio Validation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Source table: `{validation.get('source_path')}`",
        f"- Selected field: `{validation.get('selected_field', '')}`",
        f"- Selected band: `{validation.get('selected_band', '')}`",
        f"- Total rows: `{validation.get('row_count_total', '')}`",
        f"- Selected rows: `{validation.get('selected_row_count', '')}`",
        f"- Finite selected rows: `{validation.get('finite_selected_rows', '')}`",
        f"- Fractions in [0, 1]: `{validation.get('fractions_in_bounds', '')}`",
        f"- Positive finite fractions: `{validation.get('positive_fractions', '')}`",
        f"- Max |galaxy_fraction - (1 - star_fraction_wU)|: `{validation.get('max_abs_galaxy_fraction_minus_1_minus_wU', '')}`",
        f"- Max |stored log10_NG_NS - log10(galaxy_fraction/star_fraction)|: `{validation.get('max_abs_log10_formula_residual', '')}`",
        f"- Strict 1e-6 galaxy-fraction check passed: `{validation.get('strict_1e_minus_6_galaxy_fraction_check_passed', '')}`",
        f"- Strict 1e-6 stored-log-column check passed: `{validation.get('strict_1e_minus_6_log_formula_check_passed', '')}`",
        f"- Accepted for Fig 2.1: `{validation.get('accepted_for_fig2_1')}`",
        "",
        "## Convention",
        "",
        "`star_fraction_wU` is interpreted as the unresolved/star mixture weight `wU`.",
        "`galaxy_fraction = 1 - star_fraction_wU`.",
        "The ratio convention is `log10(NG/NS) = log10(galaxy_fraction / star_fraction_wU)`, so larger values mean more galaxies relative to stars.",
        "",
        "## Notes",
        "",
        str(validation.get("notes", "")),
        "",
        "The Fig 2.1 runner recomputes the plotted v9 ratio from the fraction columns instead of using the rounded stored log column.",
        f"Rows emitted for Fig 2.1: `{len(rows)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = repo_root_from_file()
    fig_dir = repo_root / "paper_convergence/figures/section2_bayesian_method"
    result_dir = repo_root / "paper_convergence/results/section2_bayesian_method"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary, metadata = build_summary(repo_root)
    summary_path = result_dir / "fig2_1_sg_prior_vs_rmag_summary.csv"
    notes_path = result_dir / "fig2_1_sg_prior_vs_rmag_notes.md"
    validation_path = result_dir / "v9_mixture_ratio_validation.md"
    fig_path = fig_dir / "fig2_1_sg_prior_vs_rmag.png"

    summary.to_csv(summary_path, index=False)
    write_notes(notes_path, summary, metadata)
    write_v9_validation(validation_path, repo_root)
    outputs = plot_summary(summary, fig_path)

    for path in [*outputs, summary_path, notes_path, validation_path]:
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    main()

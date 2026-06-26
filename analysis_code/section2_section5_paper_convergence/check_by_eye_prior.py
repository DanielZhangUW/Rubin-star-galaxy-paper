"""Write lightweight sanity checks for the by-eye S/G count prior."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_priors import (
    apply_sg_prior_to_log_likelihood_ratio,
    log10_ng_over_ns_prior_from_rmag,
    log10_ns_over_ng_prior_from_rmag,
    log_prior_odds_star_over_gal,
)


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def build_sanity_table() -> pd.DataFrame:
    rmags = np.array([23.0, 24.0, 25.0], dtype=float)
    log10_ng_ns = log10_ng_over_ns_prior_from_rmag(rmags, bounds_action="raise")
    log10_ns_ng = log10_ns_over_ng_prior_from_rmag(rmags, bounds_action="raise")
    ln_prior = log_prior_odds_star_over_gal(rmags, bounds_action="raise")
    posterior_from_zero = apply_sg_prior_to_log_likelihood_ratio(
        np.zeros_like(rmags), rmags, bounds_action="raise"
    )
    return pd.DataFrame(
        {
            "r_cmodel_mag": rmags,
            "log10_NG_over_NS": log10_ng_ns,
            "log10_NS_over_NG": log10_ns_ng,
            "ln_prior_odds_star_over_gal": ln_prior,
            "posterior_log_odds_if_logLR_zero": posterior_from_zero,
            "expected_log10_NG_over_NS": [1.12, 1.40, 1.60],
            "expected_log10_NS_over_NG": [-1.12, -1.40, -1.60],
        }
    )


def write_notes(path: Path, table: pd.DataFrame) -> None:
    markdown_table = table.to_csv(index=False).strip().splitlines()
    header = markdown_table[0].split(",")
    body = [row.split(",") for row in markdown_table[1:]]
    table_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    table_lines.extend("| " + " | ".join(row) + " |" for row in body)
    lines = [
        "# By-Eye S/G Prior Sanity Notes",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This is a lightweight sanity check for the reusable r-band CModel magnitude prior.",
        "",
        "## Prior Definition",
        "",
        "- For `rmag < 24`: `log10(NG/NS) = 1.40 + 0.28 * (rmag - 24)`.",
        "- For `rmag >= 24`: `log10(NG/NS) = 1.40 + 0.20 * (rmag - 24)`.",
        "- Bayes star-over-galaxy prior uses the inverse: `log10(NS/NG) = -log10(NG/NS)`.",
        "- Natural-log prior odds: `ln[P(S|m)/P(G|m)] = ln(10) * log10(NS/NG)`.",
        "",
        "## Sanity Values",
        "",
        "\n".join(table_lines),
        "",
        "The `posterior_log_odds_if_logLR_zero` column verifies that applying the prior to a zero morphology log-likelihood ratio returns the natural-log prior odds.",
        "",
        "## Scope",
        "",
        "- No stored v8/v9 pS parquet files are modified.",
        "- This prior is not applied directly to stored `pS` values.",
        "- Future prior-calibrated pS should be computed from morphology likelihood ratios, with this prior applied once after combining bands.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_morphology_design_note(path: Path) -> None:
    lines = [
        "# Morphology Log-Likelihood Ratio Design Note",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Stored v8/v9 pS values should not be recalibrated by directly multiplying by the new magnitude prior.",
        "",
        "Reason: `compute_pS()` currently returns `p_unres / p_total`, where `p_unres` includes the fitted unresolved/star mixture weight and `p_total` includes all fitted mixture weights. Those weights encode a magnitude-dependent normalization that acts like an implicit prior.",
        "",
        "Recommended next implementation:",
        "",
        "1. Compute morphology-only likelihood terms per band from the fitted component shapes.",
        "2. Form per-band `logLR_morph = logL_star_morph - logL_galaxy_morph`.",
        "3. Combine morphology evidence across requested bands by summing logLR values for valid bands.",
        "4. Apply the r-band CModel magnitude prior once:",
        "   `posterior_log_odds = sum(logLR_morph_bands) + ln[P(S|rmag)/P(G|rmag)]`.",
        "5. Convert to posterior star probability with a sigmoid only after the prior is added.",
        "",
        "This keeps the S/G count prior separate from morphology evidence and avoids double-counting the fitted mixture normalization.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_curve_plot(path: Path) -> None:
    rmag = np.linspace(16.0, 27.0, 300)
    log10_ng_ns = log10_ng_over_ns_prior_from_rmag(rmag, bounds_action="ignore")
    log10_ns_ng = -log10_ng_ns
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(rmag, log10_ng_ns, color="#d62728", lw=2.0, label="log10(NG/NS)")
    ax.plot(rmag, log10_ns_ng, color="#1f77b4", lw=1.6, ls="--", label="log10(NS/NG)")
    ax.axvline(24.0, color="0.45", lw=1.0, ls=":", label="rmag = 24")
    ax.axhline(-1.5, color="0.75", lw=0.8, ls=":")
    ax.axhline(2.0, color="0.75", lw=0.8, ls=":")
    ax.set_xlabel("r CModel magnitude")
    ax.set_ylabel("log10 count ratio")
    ax.set_xlim(16.0, 27.0)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = repo_root_from_file()
    result_dir = repo_root / "paper_convergence/results/section2_bayesian_method"
    fig_dir = repo_root / "paper_convergence/figures/section2_bayesian_method"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    table = build_sanity_table()
    table_path = result_dir / "by_eye_prior_sanity_values.csv"
    notes_path = result_dir / "by_eye_prior_notes.md"
    design_path = result_dir / "morphology_loglr_design_notes.md"
    plot_path = fig_dir / "by_eye_prior_curve_sanity.png"

    table.to_csv(table_path, index=False)
    write_notes(notes_path, table)
    write_morphology_design_note(design_path)
    write_curve_plot(plot_path)

    for path in (table_path, notes_path, design_path, plot_path):
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    main()

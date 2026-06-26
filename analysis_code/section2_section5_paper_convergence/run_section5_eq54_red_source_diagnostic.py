"""Generate the Section 5.1 Eq.54-prior red-source diagnostic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from paper_discussion import plot_fig5_1_red_sources_eq54prior


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_count(value: int | float) -> str:
    return f"{int(value):,}"


def run(repo_root: Path) -> dict[str, list[Path]]:
    eq54_path = repo_root / "outputs/dp2_cosmos_ps_v9_eq54prior.parquet"
    analysis_path = repo_root / "outputs/dp2_cosmos_analysis_table.parquet"
    if not eq54_path.exists():
        raise FileNotFoundError(eq54_path)
    if not analysis_path.exists():
        raise FileNotFoundError(analysis_path)

    eq54_cols = ["object_id", "cmodel_mag_r", "pS_r_eq54prior"]
    optional_cols = ["pS_gri_eq54prior", "pS_ugrizy_eq54prior"]
    available_eq54_cols = set(pq.read_schema(eq54_path).names)
    eq54_cols.extend([col for col in optional_cols if col in available_eq54_cols])
    eq54 = pd.read_parquet(eq54_path, columns=eq54_cols)

    analysis_cols = ["object_id", "ri"]
    analysis = pd.read_parquet(analysis_path, columns=analysis_cols)
    merged = eq54.merge(analysis, on="object_id", how="left", validate="one_to_one")

    figures_dir = repo_root / "paper_convergence/figures/section5_discussion"
    results_dir = repo_root / "paper_convergence/results/section5_discussion"
    docs_dir = repo_root / "paper_convergence/docs"
    output_png = figures_dir / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior.png"
    saved_figures, summary = plot_fig5_1_red_sources_eq54prior(merged, output_png)

    summary_csv = results_dir / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    rmag = pd.to_numeric(merged["cmodel_mag_r"], errors="coerce")
    ri = pd.to_numeric(merged["ri"], errors="coerce")
    ps = pd.to_numeric(merged["pS_r_eq54prior"], errors="coerce")
    paper_sample = rmag.gt(16.0) & rmag.lt(26.0)
    red_sample = paper_sample & ri.gt(1.4) & ri.notna()
    finite_ps = paper_sample & ps.notna()
    finite_red_ps = red_sample & ps.notna()
    old_png = figures_dir / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all.png"
    old_pdf = figures_dir / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all.pdf"

    summary_md = results_dir / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior_summary.md"
    _write_text(
        summary_md,
        "\n".join(
            [
                "# Fig 5.1 Eq.54-Prior Red-Source Summary",
                "",
                f"- Source pS table: `{eq54_path.relative_to(repo_root)}`",
                f"- Color source table: `{analysis_path.relative_to(repo_root)}`",
                "- Paper sample: COSMOS objects with `16 < cmodel_mag_r < 26`.",
                "- Red-source definition: dust-corrected `ri > 1.4`.",
                "- Score: `pS_r_eq54prior`.",
                f"- Paper-sample rows: {_format_count(paper_sample.sum())}",
                f"- Red-source rows: {_format_count(red_sample.sum())}",
                f"- Rows with finite Eq.54 pS in paper sample: {_format_count(finite_ps.sum())}",
                f"- Red-source rows with finite Eq.54 pS: {_format_count(finite_red_ps.sum())}",
                "",
                "The CSV table records per-bin counts, pS medians, 10th/90th percentiles, "
                "and fractions above pS thresholds 0.5, 0.1, 0.01, and 0.001.",
            ]
        )
        + "\n",
    )

    report = docs_dir / "section5_red_source_diagnostic_report.md"
    _write_text(
        report,
        "\n".join(
            [
                "# Section 5.1 Red-Source Diagnostic",
                "",
                "This is the Section 5.1 red-source diagnostic for the COSMOS paper sample.",
                "",
                "Definitions:",
                "- Paper sample: `16 < cmodel_mag_r < 26`.",
                "- Red sources: dust-corrected `r-i > 1.4`, using the existing `ri` column from the COSMOS analysis table.",
                "- Eq.54-prior score: `pS_r_eq54prior` from `outputs/dp2_cosmos_ps_v9_eq54prior.parquet`.",
                "",
                "Outputs:",
                f"- Figure PNG: `{saved_figures[0].relative_to(repo_root)}`",
                f"- Figure PDF: `{saved_figures[1].relative_to(repo_root)}`",
                f"- Summary CSV: `{summary_csv.relative_to(repo_root)}`",
                f"- Summary MD: `{summary_md.relative_to(repo_root)}`",
                "",
                "Status notes:",
                "- The old v8/stored-pS Section 5.1 figure remains diagnostic only and was not overwritten.",
                f"- Old PNG exists: `{old_png.relative_to(repo_root)}` = {old_png.exists()}",
                f"- Old PDF exists: `{old_pdf.relative_to(repo_root)}` = {old_pdf.exists()}",
                "- Because the Eq.54 prior strongly suppresses faint-end star probabilities, red-source pS distributions should be interpreted with that prior effect in mind.",
                "",
                "Sample counts:",
                f"- Paper-sample rows: {_format_count(paper_sample.sum())}",
                f"- Red-source rows: {_format_count(red_sample.sum())}",
                f"- Rows with finite Eq.54 pS in paper sample: {_format_count(finite_ps.sum())}",
                f"- Red-source rows with finite Eq.54 pS: {_format_count(finite_red_ps.sum())}",
            ]
        )
        + "\n",
    )

    return {
        "figures": saved_figures,
        "summaries": [summary_csv, summary_md],
        "docs": [report],
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    outputs = run(root)
    for group, paths in outputs.items():
        print(group)
        for path in paths:
            print(" ", path.relative_to(root))

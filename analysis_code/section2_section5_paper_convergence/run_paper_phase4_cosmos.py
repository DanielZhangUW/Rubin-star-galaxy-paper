"""Generate Phase 4 COSMOS discussion figure and final paper-convergence docs."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from paper_discussion import plot_fig5_1_red_sources
from paper_sample_selection import PaperPaths, prepare_phase1_samples


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _branch_name(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _all_figure_records(repo_root: Path) -> list[dict]:
    records = [
        {
            "section": "1.1",
            "figure_id": "fig1_1",
            "title": "COSMOS dataset overview",
            "path": "paper_convergence/figures/section1_dataset/fig1_1_cosmos_dataset_overview.png",
            "pdf": "paper_convergence/figures/section1_dataset/fig1_1_cosmos_dataset_overview.pdf",
            "summary": "paper_convergence/results/section1_dataset/fig1_1_cosmos_dataset_overview_counts.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase1_cosmos.py",
            "caption": "COSMOS DP2 and COSMOS2020 matched sample overview: sky distribution, counts vs uncorrected r CModel magnitude, and r-band flux uncertainty ratio.",
            "sample": "COSMOS coordinate cut; 16 < uncorrected r CModelMag < 26 where applicable.",
            "caveats": "COSMOS2020-only r magnitude is unavailable in the truth-label CSV and is not plotted.",
            "status": "main paper candidate",
        },
        {
            "section": "diagnostics",
            "figure_id": "cosmos_matching_radius_sweep",
            "title": "Matching radius sweep",
            "path": "paper_convergence/figures/diagnostics/cosmos_matching_radius_sweep.png",
            "pdf": "paper_convergence/figures/diagnostics/cosmos_matching_radius_sweep.pdf",
            "summary": "paper_convergence/results/diagnostics/cosmos_matching_radius_sweep.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase1_cosmos.py",
            "caption": "Nearest-neighbor matching fraction as a function of matching radius.",
            "sample": "COSMOS coordinate cut; 16 < uncorrected r CModelMag < 26 where applicable.",
            "caveats": "DP2 denominator uses the rectangular COSMOS coordinate cut.",
            "status": "diagnostic only",
        },
        {
            "section": "1.2",
            "figure_id": "fig1_2",
            "title": "Truth-label morphology and color overview",
            "path": "paper_convergence/figures/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2.png",
            "pdf": "paper_convergence/figures/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2.pdf",
            "summary": "paper_convergence/results/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase1_cosmos.py",
            "caption": "COSMOS2020 truth-label stars and galaxies in morphology, color-magnitude, and color-color spaces.",
            "sample": "COSMOS2020-matched paper sample.",
            "caveats": "Colors use existing repo dust-correction coefficients and fixed Week10 color-color ranges.",
            "status": "main paper candidate",
        },
        {
            "section": "1.3",
            "figure_id": "fig1_3",
            "title": "Truth-label color-color diagrams by magnitude range",
            "path": "paper_convergence/figures/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4.png",
            "pdf": "paper_convergence/figures/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4.pdf",
            "summary": "paper_convergence/results/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase1_cosmos.py",
            "caption": "COSMOS2020 truth-label color-color diagrams split into 16<rmag<25 and 25<rmag<26.",
            "sample": "COSMOS2020-matched paper sample.",
            "caveats": "Uses fixed Week10 color-color axis ranges.",
            "status": "main paper candidate",
        },
        {
            "section": "1.4",
            "figure_id": "fig1_4_r_extendedness",
            "title": "r extendedness color-color diagrams",
            "path": "paper_convergence/figures/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4.png",
            "pdf": "paper_convergence/figures/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4.pdf",
            "summary": "paper_convergence/results/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase2_cosmos.py",
            "caption": "COSMOS color-color diagrams split by r-band extendedness classification.",
            "sample": "COSMOS DP2 paper sample.",
            "caveats": "NaN/other extendedness values are excluded from plotted classes and counted in the summary.",
            "status": "main paper candidate",
        },
        {
            "section": "1.4",
            "figure_id": "fig1_4_refExtendedness",
            "title": "refExtendedness color-color diagrams",
            "path": "paper_convergence/figures/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4.png",
            "pdf": "paper_convergence/figures/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4.pdf",
            "summary": "paper_convergence/results/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase2_cosmos.py",
            "caption": "COSMOS color-color diagrams split by refExtendedness classification.",
            "sample": "COSMOS DP2 paper sample.",
            "caveats": "NaN/other refExtendedness values are excluded from plotted classes and counted in the summary.",
            "status": "appendix candidate",
        },
        {
            "section": "1.5",
            "figure_id": "fig1_5_r_extendedness",
            "title": "r extendedness confusion CMD",
            "path": "paper_convergence/figures/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel.png",
            "pdf": "paper_convergence/figures/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel.pdf",
            "summary": "paper_convergence/results/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase2_cosmos.py",
            "caption": "COSMOS2020 truth versus r-band extendedness classification in a dust-corrected g-i CMD.",
            "sample": "COSMOS2020-matched paper sample.",
            "caveats": "y-axis uses uncorrected r CModel magnitude.",
            "status": "main paper candidate",
        },
        {
            "section": "1.5",
            "figure_id": "fig1_5_refExtendedness",
            "title": "refExtendedness confusion CMD",
            "path": "paper_convergence/figures/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel.png",
            "pdf": "paper_convergence/figures/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel.pdf",
            "summary": "paper_convergence/results/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase2_cosmos.py",
            "caption": "COSMOS2020 truth versus refExtendedness classification in a dust-corrected g-i CMD.",
            "sample": "COSMOS2020-matched paper sample.",
            "caveats": "y-axis uses uncorrected r CModel magnitude.",
            "status": "appendix candidate",
        },
        {
            "section": "1.6",
            "figure_id": "fig1_6_r_extendedness",
            "title": "r extendedness performance baseline",
            "path": "paper_convergence/figures/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins.png",
            "pdf": "paper_convergence/figures/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins.pdf",
            "summary": "paper_convergence/results/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase2_cosmos.py",
            "caption": "Binary r-band extendedness operating-point ROC metrics in three r-magnitude bins.",
            "sample": "COSMOS2020-matched paper sample.",
            "caveats": "Extendedness is binary; plotted curves are step/operating-point ROC diagnostics.",
            "status": "main paper candidate",
        },
        {
            "section": "1.6",
            "figure_id": "fig1_6_refExtendedness",
            "title": "refExtendedness performance baseline",
            "path": "paper_convergence/figures/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins.png",
            "pdf": "paper_convergence/figures/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins.pdf",
            "summary": "paper_convergence/results/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb",
            "script": "paper_convergence/code/run_paper_phase2_cosmos.py",
            "caption": "Binary refExtendedness operating-point ROC metrics in three r-magnitude bins.",
            "sample": "COSMOS2020-matched paper sample.",
            "caveats": "refExtendedness is binary; plotted curves are step/operating-point ROC diagnostics.",
            "status": "appendix candidate",
        },
    ]

    for section, fig_id, title, path, summary, caption, status, caveats in [
        ("2.1", "fig2_1", "Star/galaxy prior vs r magnitude", "fig2_1_cosmos_star_galaxy_prior_vs_rmag", "fig2_1_cosmos_star_galaxy_prior_vs_rmag_table.csv", "COSMOS matched star/galaxy count-ratio prior as a function of uncorrected r CModel magnitude.", "main paper candidate", "Prior is empirical from matched labels only."),
        ("2.2", "fig2_2", "r-band delta slice distributions", "fig2_2_cosmos_r_delta_slice_fits_2x4", "fig2_2_cosmos_r_delta_slice_fits_2x4_summary.csv", "COSMOS r-band psfMag-CModelMag slice distributions in display magnitude bins.", "main paper candidate", "First-pass display guide uses robust Gaussian overlays."),
        ("2.3", "fig2_3", "u/r/y data-model residuals", "fig2_3_cosmos_ury_data_model_residuals_3x3", "fig2_3_cosmos_ury_data_model_residuals_3x3_summary.csv", "COSMOS u/r/y data, smoothed row model, and residuals in morphology-magnitude space.", "diagnostic only", "First-pass residual display; full model2D fitting can replace the smoothed row model later."),
        ("2.4", "fig2_4", "r-band pS map", "fig2_4_cosmos_r_pS_map_16_26", "fig2_4_cosmos_r_pS_map_16_26_summary.csv", "COSMOS r-band v8 pS map in morphology-magnitude space.", "main paper candidate", "No Eq.54 prior calibration yet because log-likelihood components are unavailable."),
        ("2.5", "fig2_5", "single-band pS vs r extendedness performance", "fig2_5_cosmos_single_band_pS_vs_r_extendedness_performance", "fig2_5_cosmos_single_band_pS_vs_r_extendedness_performance_summary.csv", "COSMOS r-band pS ROC compared with r-band extendedness operating point.", "main paper candidate", "Draft/no Eq.54 prior calibration yet."),
        ("2.6", "fig2_6", "multiband r/gri/ugrizy performance", "fig2_6_cosmos_multiband_r_gri_ugrizy_performance", "fig2_6_cosmos_multiband_r_gri_ugrizy_performance_summary.csv", "COSMOS empirical likelihood performance for r, gri, and ugrizy band combinations.", "main paper candidate", "Same-bin empirical likelihood draft/no Eq.54 prior calibration yet."),
        ("2.7", "fig2_7", "ugrizy method color-color diagrams", "fig2_7_cosmos_ugrizy_method_color_color_2x4", "fig2_7_cosmos_ugrizy_method_color_color_2x4_summary.csv", "COSMOS color-color diagrams classified by ugrizy empirical likelihood method labels.", "appendix candidate", "Method labels use logL_star-logL_galaxy >= 0; no Eq.54 prior calibration yet."),
    ]:
        records.append(
            {
                "section": section,
                "figure_id": fig_id,
                "title": title,
                "path": f"paper_convergence/figures/section2_bayesian_method/{path}.png",
                "pdf": f"paper_convergence/figures/section2_bayesian_method/{path}.pdf",
                "summary": f"paper_convergence/results/section2_bayesian_method/{summary}",
                "notebook": "paper_convergence/notebooks/paper_section2_bayesian_method.ipynb",
                "script": "paper_convergence/code/run_paper_phase3_cosmos.py",
                "caption": caption,
                "sample": "COSMOS/COSMOS2020 paper sample.",
                "caveats": caveats,
                "status": status,
            }
        )

    records.append(
        {
            "section": "5.1",
            "figure_id": "fig5_1",
            "title": "Red-source r pS diagnostic",
            "path": "paper_convergence/figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all.png",
            "pdf": "paper_convergence/figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all.pdf",
            "summary": "paper_convergence/results/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all_summary.csv",
            "notebook": "paper_convergence/notebooks/paper_section5_discussion.ipynb",
            "script": "paper_convergence/code/run_paper_phase4_cosmos.py",
            "caption": "COSMOS r-band pS distributions for all sources compared with red sources selected by dust-corrected r-i > 1.4.",
            "sample": "COSMOS DP2 paper sample; 16 < uncorrected r CModelMag < 26.",
            "caveats": "Uses existing v8 pS_r; no Eq.54 prior calibration yet.",
            "status": "diagnostic only",
        }
    )

    for record in records:
        record["generated"] = Path(repo_root / record["path"]).exists()
        record["pdf_generated"] = Path(repo_root / record["pdf"]).exists()
        record["summary_generated"] = Path(repo_root / record["summary"]).exists()
    return records


def _write_manifest(repo_root: Path, records: list[dict]) -> Path:
    lines = [
        "# Figure Manifest",
        "",
        "This manifest tracks paper-convergence figures, generating code, sample definitions, and caveats.",
        "",
    ]
    current_section = None
    for record in records:
        if record["section"] != current_section:
            current_section = record["section"]
            lines.append(f"## Section {current_section}")
            lines.append("")
        lines += [
            f"### {record['figure_id']}: {record['title']}",
            f"- Filename: `{record['path']}`",
            f"- PDF: `{record['pdf']}`",
            f"- Generated by: `{record['script']}`",
            f"- Notebook: `{record['notebook']}`",
            f"- Caption draft: {record['caption']}",
            f"- Sample definition: {record['sample']}",
            f"- Caveats: {record['caveats']}",
            f"- Status: {record['status']}",
            f"- Generated: {record['generated']}",
            "",
        ]
    path = repo_root / "paper_convergence/docs/figure_manifest.md"
    _write_text(path, "\n".join(lines))
    return path


def _write_overleaf_manifest(repo_root: Path, records: list[dict]) -> Path:
    lines = [
        "# Overleaf Upload Manifest",
        "",
        "No local Overleaf/manuscript directory was updated. Upload these figures manually if needed.",
        "",
        "| Local figure path | Suggested Overleaf filename | Section | Placement | Draft caption |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        if record["status"] == "diagnostic only":
            placement = "diagnostic/appendix"
        elif record["status"] == "appendix candidate":
            placement = "appendix"
        else:
            placement = "main text"
        suggested = Path(record["path"]).name
        lines.append(
            f"| `{record['path']}` | `{suggested}` | {record['section']} | {placement} | {record['caption']} |"
        )
    path = repo_root / "paper_convergence/docs/overleaf_upload_manifest.md"
    _write_text(path, "\n".join(lines) + "\n")
    return path


def _write_checklist(repo_root: Path, records: list[dict]) -> Path:
    path = repo_root / "paper_convergence/docs/paper_figure_checklist.csv"
    rows = []
    for record in records:
        rows.append(
            {
                "section": record["section"],
                "figure_id": record["figure_id"],
                "expected_output": record["path"],
                "generated": record["generated"],
                "output_path": record["path"],
                "csv_summary_path": record["summary"],
                "notebook": record["notebook"],
                "script_or_function": record["script"],
                "status": record["status"],
                "notes": record["caveats"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_generation_summary(repo_root: Path, records: list[dict], phase4_outputs: list[Path]) -> Path:
    branch = _branch_name(repo_root)
    generated = [r for r in records if r["generated"]]
    missing = [r for r in records if not r["generated"]]
    lines = [
        "# Paper Figure Generation Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Branch: `{branch}`",
        "",
        "## Data Files Read",
        "- `outputs/dp2_cosmos_analysis_table.parquet`",
        "- `outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet`",
        "- `outputs/dp2_cosmos_ps_v8.parquet`",
        "- `data/cosmos2020_farmer_truth_catalog_github.csv`",
        "",
        "## Sample Definition",
        "- Field: COSMOS.",
        "- External labels: COSMOS2020.",
        "- Coordinate cut: 149.50413 < RA < 150.99170 and 1.48760 < Dec < 2.97521.",
        "- Main magnitude cut: 16 < uncorrected r-band CModel magnitude < 26.",
        "- Colors are CModel colors corrected with the existing repo extinction convention.",
        "",
        "## Generated Figures",
    ]
    for record in generated:
        lines.append(f"- `{record['path']}`")
    lines += ["", "## Generated CSV/MD Summaries"]
    for record in generated:
        lines.append(f"- `{record['summary']}`")
    lines += ["", "## Phase 4 Outputs"]
    for p in phase4_outputs:
        lines.append(f"- `{p.relative_to(repo_root)}`")
    lines += [
        "",
        "## Complete",
        "- Phase 1: COSMOS sample definition, dataset overview, truth-label morphology/color figures.",
        "- Phase 2: r extendedness and refExtendedness baseline figures.",
        "- Phase 3: Bayesian-method first-pass figures, pS map, empirical likelihood performance, and method-label color-color diagrams.",
        "- Phase 4: red-source discussion diagnostic and final documentation manifests.",
        "",
        "## Incomplete Or Deferred",
        "- Eq.54 prior-calibrated pS is not implemented because the available v8 pS parquet does not include `logL_star`/`logL_galaxy` components.",
        "- Fig 2.2 and Fig 2.3 are first-pass display/model diagnostics; full production mixture/model2D fitting can replace them.",
        "- No local Overleaf/manuscript directory was updated; use `paper_convergence/docs/overleaf_upload_manifest.md` for manual upload.",
    ]
    if missing:
        lines += ["", "## Missing Expected Figures"]
        for record in missing:
            lines.append(f"- `{record['path']}`")
    else:
        lines += ["", "## Missing Expected Figures", "- None."]
    lines += [
        "",
        "## Caveats",
        "- COSMOS/COSMOS2020 is the primary paper sample for this pass.",
        "- ECDFS/HST is secondary and can be rerun later.",
        "- r-band magnitude is uncorrected for sample selection.",
        "- color-color plots use dust-corrected CModel colors.",
        "- pS/prior-calibrated figures are labeled when they are still empirical/no Eq.54 prior calibration.",
    ]
    path = repo_root / "paper_convergence/docs/paper_figure_generation_summary.md"
    _write_text(path, "\n".join(lines) + "\n")
    return path


def _write_readme(repo_root: Path) -> Path:
    path = repo_root / "paper_convergence/README.md"
    text = """# Paper Convergence

This folder contains the paper-convergence pipeline for the Rubin star-galaxy separation project.

- Primary paper sample: COSMOS DP2 matched to COSMOS2020 external labels.
- ECDFS/HST is secondary and can be rerun later as a comparison.
- Figures are in `figures/`.
- Numerical summaries are in `results/`.
- Notebooks are in `notebooks/`.
- Reusable paper-specific code is in `code/`.
- Documentation and manifests are in `docs/`.
- Logs are in `logs/`.
- Private data, FITS files, parquet catalogs, and large Rubin catalog files are not stored here.
- Main sample selection uses uncorrected r-band CModel magnitude.
- Colors are computed from CModel magnitudes corrected for dust extinction with the existing repo extinction convention.
- Shared plot style is centralized in `paper_convergence/code/paper_plot_style.py`.

## Reproducibility

From the repository root, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase1_cosmos.py
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase2_cosmos.py
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase3_cosmos.py
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase4_cosmos.py
```
"""
    _write_text(path, text)
    return path


def run_phase4(repo_root: Path) -> dict[str, list[Path]]:
    paths = PaperPaths.from_repo_root(repo_root)
    samples = prepare_phase1_samples(repo_root)
    dp2 = samples["dp2_paper"]
    ps = pd.read_parquet(paths.repo_root / "outputs/dp2_cosmos_ps_v8.parquet", columns=["object_id", "pS_r"])
    dp2_ps = dp2.merge(ps.drop_duplicates("object_id"), on="object_id", how="left")

    out_fig = repo_root / "paper_convergence/figures/section5_discussion"
    out_res = repo_root / "paper_convergence/results/section5_discussion"
    out_log = repo_root / "paper_convergence/logs"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)
    out_log.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, list[Path]] = {}
    fig_paths, summary = plot_fig5_1_red_sources(
        dp2_ps,
        out_fig / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all.png",
    )
    summary_path = out_res / "fig5_1_cosmos_red_sources_r_pS_hist_vs_all_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs["fig5_1"] = fig_paths + [summary_path]

    records = _all_figure_records(repo_root)
    readme = _write_readme(repo_root)
    manifest = _write_manifest(repo_root, records)
    overleaf_manifest = _write_overleaf_manifest(repo_root, records)
    checklist = _write_checklist(repo_root, records)
    generation_summary = _write_generation_summary(repo_root, records, outputs["fig5_1"])
    outputs["docs"] = [readme, manifest, overleaf_manifest, checklist, generation_summary]

    lines = [
        "# Phase 4 Generation Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Completed",
    ]
    for key, paths_out in outputs.items():
        lines.append(f"- {key}:")
        for p in paths_out:
            lines.append(f"  - `{p.relative_to(repo_root)}`")
    lines += [
        "",
        "## Missing Or Deferred",
        "- Eq.54 prior-calibrated pS remains deferred; required log-likelihood components are unavailable.",
        "- No Overleaf directory was updated; use the upload manifest for manual copy.",
        "",
        "## Assumptions",
        "- Red sources are selected with dust-corrected r-i > 1.4.",
        "- Existing v8 pS_r is used without Eq.54 prior calibration.",
    ]
    log_path = out_log / "phase4_generation_summary.md"
    _write_text(log_path, "\n".join(lines) + "\n")
    outputs["logs"] = [log_path]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    outputs = run_phase4(args.repo_root.resolve())
    for key, paths_out in outputs.items():
        print(f"[{key}]")
        for p in paths_out:
            print(f"  {p}")


if __name__ == "__main__":
    main()

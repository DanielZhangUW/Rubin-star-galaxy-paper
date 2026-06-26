#!/usr/bin/env python3
"""Generate v9-fit-parameter diagnostic candidates for Fig 2.2 and Fig 2.3.

This script uses Nirav's exported v9 mixture-fit parameters for visualization
only. It does not compute or update pS values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc


BANDS_FIG23 = ("u", "r", "y")
DISPLAY_BINS_FIG22 = (
    (20.0, 21.0),
    (21.0, 22.0),
    (22.0, 23.0),
    (23.0, 24.0),
    (24.0, 24.5),
    (24.5, 25.0),
    (25.0, 25.5),
    (25.5, 26.0),
)
MODEL_MAG_EDGES = np.arange(16.0, 26.0 + 0.5, 0.5)
DELTA_RANGE = (-0.5, 1.5)
DELTA_EDGES_1D = np.linspace(DELTA_RANGE[0], DELTA_RANGE[1], 201)
DELTA_EDGES_2D = np.linspace(DELTA_RANGE[0], DELTA_RANGE[1], 121)
RESIDUAL_EPSILON = 1.0


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    fit_table: Path
    cosmos_table: Path
    fig_dir: Path
    result_dir: Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "paper_convergence").exists() and (candidate / "outputs").exists():
            return candidate
    raise RuntimeError("Could not find repository root containing paper_convergence and outputs")


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = abs(float(sigma)) + 1e-12
    return np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def skewnorm_pdf(x: np.ndarray, mu: float, sigma: float, alpha: float) -> np.ndarray:
    """Skew-normal convention matched to src/psf_cmodel_fit.py."""
    sigma = abs(float(sigma)) + 1e-12
    t = (x - float(mu)) / sigma
    return gaussian_pdf(x, mu, sigma) * erfc(-float(alpha) * t / np.sqrt(2.0))


def mixture_components(delta_centers: np.ndarray, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    star = float(row["wU"]) * gaussian_pdf(delta_centers, float(row["muU"]), float(row["sigmaU"]))
    resolved = np.zeros_like(delta_centers, dtype=float)
    for idx in (1, 2, 3):
        resolved += float(row[f"wR{idx}"]) * skewnorm_pdf(
            delta_centers,
            float(row[f"muR{idx}"]),
            float(row[f"sigmaR{idx}"]),
            float(row[f"alphaR{idx}"]),
        )
    total = star + resolved
    return star, resolved, total


def parse_mag_bin(label: str) -> tuple[float, float]:
    low, high = str(label).split("-")
    return float(low), float(high)


def load_fit_parameters(path: Path) -> pd.DataFrame:
    required = {
        "field",
        "band",
        "mag_bin",
        "muU",
        "sigmaU",
        "wU",
        "wR1",
        "muR1",
        "sigmaR1",
        "alphaR1",
        "wR2",
        "muR2",
        "sigmaR2",
        "alphaR2",
        "wR3",
        "muR3",
        "sigmaR3",
        "alphaR3",
    }
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required fit-parameter columns: {missing}")
    bins = df["mag_bin"].map(parse_mag_bin)
    df = df.copy()
    df["mag_low"] = [b[0] for b in bins]
    df["mag_high"] = [b[1] for b in bins]
    return df


def inspect_inputs(paths: Paths, fit_df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append("# v9 Fit Diagnostic Input Inspection\n")
    lines.append(f"- Fit-parameter table: `{paths.fit_table.relative_to(paths.repo_root)}`")
    lines.append(f"- COSMOS analysis table: `{paths.cosmos_table.relative_to(paths.repo_root)}`")
    lines.append(f"- Fit rows: {len(fit_df)}")
    lines.append(f"- Available fields: {', '.join(sorted(fit_df['field'].dropna().unique()))}")
    lines.append(f"- Available bands: {', '.join(sorted(fit_df['band'].dropna().unique()))}")
    lines.append("- Magnitude bin column: `mag_bin`, parsed into `mag_low` and `mag_high`")
    lines.append("- Unresolved parameters: `muU`, `sigmaU`, `wU`")
    lines.append("- Resolved parameters: `wR1/muR1/sigmaR1/alphaR1`, `wR2/...`, `wR3/...`")
    count_cols = [c for c in fit_df.columns if c.lower() in {"n", "n_objects", "count", "counts"}]
    status_cols = [c for c in fit_df.columns if "status" in c.lower() or "flag" in c.lower()]
    lines.append(f"- Count columns in fit table: {count_cols if count_cols else 'none'}")
    lines.append(f"- Fit status/fallback columns: {status_cols if status_cols else 'none'}")
    lines.append("- Fit table has no `n_objects`, so display-bin model averaging uses actual object counts from the COSMOS analysis table.")
    lines.append("")
    return lines


def read_cosmos_columns(path: Path) -> pd.DataFrame:
    cols = ["object_id"]
    for band in BANDS_FIG23:
        cols.extend([f"cmodel_mag_{band}", f"psf_minus_cmodel_{band}"])
    return pd.read_parquet(path, columns=cols)


def finite_values(df: pd.DataFrame, mag_col: str, delta_col: str, mag_low: float, mag_high: float) -> pd.DataFrame:
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    delta = pd.to_numeric(df[delta_col], errors="coerce")
    mask = (
        np.isfinite(mag)
        & np.isfinite(delta)
        & (mag >= mag_low)
        & (mag < mag_high)
        & (delta >= DELTA_RANGE[0])
        & (delta <= DELTA_RANGE[1])
    )
    return pd.DataFrame({"mag": mag[mask].to_numpy(), "delta": delta[mask].to_numpy()})


def fit_rows_for_display_bin(fit_df: pd.DataFrame, band: str, low: float, high: float) -> pd.DataFrame:
    sub = fit_df[(fit_df["field"] == "COSMOS") & (fit_df["band"] == band)].copy()
    return sub[(sub["mag_low"] >= low) & (sub["mag_high"] <= high)].sort_values("mag_low")


def weighted_model_for_display_bin(
    values: pd.DataFrame,
    fit_rows: pd.DataFrame,
    delta_centers: np.ndarray,
    delta_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    star_counts = np.zeros_like(delta_centers, dtype=float)
    resolved_counts = np.zeros_like(delta_centers, dtype=float)
    total_counts = np.zeros_like(delta_centers, dtype=float)
    notes: list[str] = []
    for _, row in fit_rows.iterrows():
        sub_mask = (values["mag"] >= row["mag_low"]) & (values["mag"] < row["mag_high"])
        n_sub = int(sub_mask.sum())
        if n_sub == 0:
            notes.append(f"no data in fit bin {row['mag_bin']}")
            continue
        star_pdf, resolved_pdf, total_pdf = mixture_components(delta_centers, row)
        star_counts += n_sub * star_pdf * delta_width
        resolved_counts += n_sub * resolved_pdf * delta_width
        total_counts += n_sub * total_pdf * delta_width

    observed_total = len(values)
    model_total = float(np.sum(total_counts))
    if observed_total > 0 and model_total > 0:
        scale = observed_total / model_total
        star_counts *= scale
        resolved_counts *= scale
        total_counts *= scale
    elif observed_total > 0:
        notes.append("model normalization failed because total model integral is zero")
    return star_counts, resolved_counts, total_counts, notes


def plot_fig2_2(paths: Paths, fit_df: pd.DataFrame, cosmos: pd.DataFrame) -> tuple[list[Path], pd.DataFrame, str]:
    mag_col = "cmodel_mag_r"
    delta_col = "psf_minus_cmodel_r"
    centers = 0.5 * (DELTA_EDGES_1D[:-1] + DELTA_EDGES_1D[1:])
    width = float(np.diff(DELTA_EDGES_1D)[0])

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), constrained_layout=True)
    summary_rows = []

    for ax, (low, high) in zip(axes.ravel(), DISPLAY_BINS_FIG22):
        values = finite_values(cosmos, mag_col, delta_col, low, high)
        fit_rows = fit_rows_for_display_bin(fit_df, "r", low, high)
        missing_note = ""
        if fit_rows.empty:
            missing_note = "missing v9 fit rows for display bin"
            star_counts = resolved_counts = total_counts = np.zeros_like(centers)
            model_notes = [missing_note]
        else:
            star_counts, resolved_counts, total_counts, model_notes = weighted_model_for_display_bin(
                values, fit_rows, centers, width
            )

        ax.hist(values["delta"], bins=DELTA_EDGES_1D, histtype="step", color="black", linewidth=1.2, label="data")
        if len(values) > 0 and np.any(total_counts > 0):
            ax.plot(centers, total_counts, color="crimson", lw=1.6, label="v9 total")
            ax.plot(centers, star_counts, color="tab:blue", lw=1.2, ls="--", label="unresolved")
            ax.plot(centers, resolved_counts, color="tab:orange", lw=1.2, ls=":", label="resolved")
        ax.set_xlim(*DELTA_RANGE)
        ax.set_title(f"{low:g} < rmag < {high:g}\nN={len(values):,}", fontsize=12)
        ax.grid(True, color="0.85", linewidth=0.6, alpha=0.7)
        ax.set_xlabel("r PSF - CModel", fontsize=11)
        ax.set_ylabel("counts", fontsize=11)
        ax.tick_params(labelsize=10)

        summary_rows.append(
            {
                "figure": "fig2_2_cosmos_r_slice_fits_v9fit",
                "field": "COSMOS",
                "band": "r",
                "mag_col": mag_col,
                "delta_col": delta_col,
                "display_mag_low": low,
                "display_mag_high": high,
                "N_objects": int(len(values)),
                "fit_bins_used": ";".join(fit_rows["mag_bin"].astype(str).tolist()),
                "model_weighting": "actual object counts per v9 0.5-mag fit bin",
                "model_normalized_to_observed_counts": bool(len(values) > 0 and np.any(total_counts > 0)),
                "notes": "; ".join(model_notes),
            }
        )

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=True, fontsize=11)

    png = paths.fig_dir / "fig2_2_cosmos_r_slice_fits_v9fit.png"
    pdf = paths.fig_dir / "fig2_2_cosmos_r_slice_fits_v9fit.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    notes = "\n".join(
        [
            "# Fig 2.2 v9-fit Candidate Notes",
            "",
            "This figure is a COSMOS r-band diagnostic candidate generated from `paper_convergence/tables/v9_fit_parameters.csv`.",
            "",
            "It uses the observed `psf_minus_cmodel_r` distribution from `outputs/dp2_cosmos_analysis_table.parquet` and overlays the exported v9 mixture model.",
            "",
            "Model definitions:",
            "- unresolved/star component: Gaussian with `muU`, `sigmaU`, and weight `wU`.",
            "- resolved/galaxy components: skew-normal PDFs with `muRk`, `sigmaRk`, `alphaRk`, and weights `wRk` for k=1..3.",
            "- skew-normal convention matches `src/psf_cmodel_fit.py`: `Gaussian(x; mu, sigma) * erfc(-alpha * t / sqrt(2))` where `t=(x-mu)/sigma`.",
            "- total model: `wU * star_pdf + sum_k wRk * resolved_pdf_k`.",
            "",
            "Display bins are the requested 20-21, 21-22, 22-23, 23-24, 24-24.5, 24.5-25, 25-25.5, and 25.5-26 r-band CModel magnitude bins.",
            "",
            "When a display bin contains multiple 0.5-mag v9 fit bins, the model curves are combined using actual object counts from the COSMOS analysis table in each fit bin. The combined model is then normalized to the observed histogram counts in the display bin.",
            "",
            "This script does not recompute pS and does not apply priors.",
        ]
    )
    return [png, pdf], pd.DataFrame(summary_rows), notes


def model_image_for_band(
    fit_df: pd.DataFrame,
    data: pd.DataFrame,
    band: str,
    mag_edges: np.ndarray,
    delta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mag_col = f"cmodel_mag_{band}"
    delta_col = f"psf_minus_cmodel_{band}"
    values = finite_values(data, mag_col, delta_col, float(mag_edges[0]), float(mag_edges[-1]))
    hist, _, _ = np.histogram2d(values["mag"], values["delta"], bins=[mag_edges, delta_edges])

    delta_centers = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    delta_width = float(np.diff(delta_edges)[0])
    model = np.zeros_like(hist, dtype=float)
    notes: list[str] = []

    band_fit = fit_df[(fit_df["field"] == "COSMOS") & (fit_df["band"] == band)].copy()
    for i, (low, high) in enumerate(zip(mag_edges[:-1], mag_edges[1:])):
        rows = band_fit[(band_fit["mag_low"] == low) & (band_fit["mag_high"] == high)]
        n_bin = int(((values["mag"] >= low) & (values["mag"] < high)).sum())
        if rows.empty:
            notes.append(f"{band}: missing fit row for {low:g}-{high:g}")
            continue
        if n_bin == 0:
            continue
        row = rows.iloc[0]
        _, _, total_pdf = mixture_components(delta_centers, row)
        counts = n_bin * total_pdf * delta_width
        total = float(np.sum(counts))
        if total > 0:
            counts *= n_bin / total
        model[i, :] = counts
    return hist, model, notes


def plot_fig2_3(paths: Paths, fit_df: pd.DataFrame, cosmos: pd.DataFrame) -> tuple[list[Path], pd.DataFrame, str]:
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 14.5), constrained_layout=True)
    summary_rows = []
    all_notes: list[str] = []
    extent = [DELTA_EDGES_2D[0], DELTA_EDGES_2D[-1], MODEL_MAG_EDGES[-1], MODEL_MAG_EDGES[0]]

    for row_idx, band in enumerate(BANDS_FIG23):
        hist, model, notes = model_image_for_band(fit_df, cosmos, band, MODEL_MAG_EDGES, DELTA_EDGES_2D)
        all_notes.extend(notes)
        residual = (hist - model) / np.sqrt(model + RESIDUAL_EPSILON)
        max_count = max(float(np.nanmax(hist)), float(np.nanmax(model)), 1.0)
        finite_resid = residual[np.isfinite(residual)]
        resid_lim = float(np.nanpercentile(np.abs(finite_resid), 99.0)) if finite_resid.size else 1.0
        resid_lim = max(resid_lim, 1.0)
        images = [
            (hist, f"{band}: data", "magma", 0.0, max_count),
            (model, f"{band}: v9 model", "magma", 0.0, max_count),
            (residual, f"{band}: (data - model) / sqrt(model + 1)", "coolwarm", -resid_lim, resid_lim),
        ]
        for col_idx, (image, title, cmap, vmin, vmax) in enumerate(images):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                image,
                origin="upper",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xlim(*DELTA_RANGE)
            ax.set_ylim(MODEL_MAG_EDGES[-1], MODEL_MAG_EDGES[0])
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(f"{band} PSF - CModel", fontsize=11)
            ax.set_ylabel(f"{band} CModel mag", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(True, color="0.85", linewidth=0.5, alpha=0.45)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=9)

        mag_col = f"cmodel_mag_{band}"
        delta_col = f"psf_minus_cmodel_{band}"
        values = finite_values(cosmos, mag_col, delta_col, float(MODEL_MAG_EDGES[0]), float(MODEL_MAG_EDGES[-1]))
        summary_rows.append(
            {
                "figure": "fig2_3_cosmos_ury_model2d_residuals_v9fit",
                "field": "COSMOS",
                "band": band,
                "mag_col": mag_col,
                "delta_col": delta_col,
                "N_objects": int(len(values)),
                "mag_binning": "16 <= mag < 26 in 0.5 mag bins",
                "delta_range": f"{DELTA_RANGE[0]} to {DELTA_RANGE[1]}",
                "delta_bins": len(DELTA_EDGES_2D) - 1,
                "model_source": "paper_convergence/tables/v9_fit_parameters.csv",
                "model_scaling": "each v9 0.5-mag bin normalized to actual object count in that bin",
                "residual_definition": "(data - model) / sqrt(model + 1)",
                "notes": "; ".join([n for n in notes if n.startswith(f"{band}:")]),
            }
        )

    png = paths.fig_dir / "fig2_3_cosmos_ury_model2d_residuals_v9fit.png"
    pdf = paths.fig_dir / "fig2_3_cosmos_ury_model2d_residuals_v9fit.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    notes_text = "\n".join(
        [
            "# Fig 2.3 v9-fit Candidate Notes",
            "",
            "This figure is a COSMOS u/r/y model2D residual diagnostic candidate generated from `paper_convergence/tables/v9_fit_parameters.csv`.",
            "",
            "Rows are bands u, r, y. Columns are observed data, exported v9 mixture model, and residual.",
            "",
            "Residual definition: `(data - model) / sqrt(model + 1)`. The +1 stabilizes bins where the exported model count is close to zero.",
            "",
            "For each 0.5-mag bin, the v9 mixture PDF is evaluated across PSF-CModel delta and scaled to the actual number of COSMOS objects in that magnitude bin.",
            "",
            "This script does not recompute pS and does not apply priors.",
            "",
            "Missing fit-bin notes:",
            *(f"- {note}" for note in all_notes),
        ]
    )
    return [png, pdf], pd.DataFrame(summary_rows), notes_text


def write_generation_notes(paths: Paths, fig22_paths: Iterable[Path], fig23_paths: Iterable[Path]) -> Path:
    path = paths.result_dir / "fig2_2_fig2_3_v9fit_generation_notes.md"
    lines = [
        "# Fig 2.2 / Fig 2.3 v9-fit Candidate Generation Notes",
        "",
        "Generated candidate versions using `paper_convergence/tables/v9_fit_parameters.csv` for model visualization only.",
        "",
        "## Outputs",
        "",
        "Fig 2.2 v9-fit candidates:",
        *(f"- `{p.relative_to(paths.repo_root)}`" for p in fig22_paths),
        "- `paper_convergence/results/section2_bayesian_method/fig2_2_cosmos_r_slice_fits_v9fit_summary.csv`",
        "- `paper_convergence/results/section2_bayesian_method/fig2_2_cosmos_r_slice_fits_v9fit_notes.md`",
        "",
        "Fig 2.3 v9-fit candidates:",
        *(f"- `{p.relative_to(paths.repo_root)}`" for p in fig23_paths),
        "- `paper_convergence/results/section2_bayesian_method/fig2_3_cosmos_ury_model2d_residuals_v9fit_summary.csv`",
        "- `paper_convergence/results/section2_bayesian_method/fig2_3_cosmos_ury_model2d_residuals_v9fit_notes.md`",
        "",
        "## Relationship to Previous Candidates",
        "",
        "The new v9-fit candidates are stronger model-visualization diagnostics than the previous robust-Gaussian and smoothed-histogram candidates because they use the exported v9 fit-parameter table.",
        "",
        "They do not automatically supersede the previous candidates until visual QC confirms that the exported fit parameters are the intended production display model for the manuscript.",
        "",
        "## Limitations",
        "",
        "- `v9_fit_parameters.csv` does not contain `n_objects`, so model-bin weighting uses actual object counts from `outputs/dp2_cosmos_analysis_table.parquet`.",
        "- These figures do not recompute pS.",
        "- These figures do not apply any star/galaxy prior.",
        "- These figures are diagnostics for visual QC and review.",
        "",
        "## Next Visual Checks",
        "",
        "- Confirm Fig 2.2 component curves track observed bright-bin and faint-bin structure well enough for paper use.",
        "- Confirm Fig 2.3 residual color scaling is readable and the residual definition is acceptable.",
        "- Confirm whether top titles should be removed before final paper insertion.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    repo_root = find_repo_root(Path.cwd().resolve())
    paths = Paths(
        repo_root=repo_root,
        fit_table=repo_root / "paper_convergence" / "tables" / "v9_fit_parameters.csv",
        cosmos_table=repo_root / "outputs" / "dp2_cosmos_analysis_table.parquet",
        fig_dir=repo_root / "paper_convergence" / "figures" / "section2_bayesian_method",
        result_dir=repo_root / "paper_convergence" / "results" / "section2_bayesian_method",
    )
    paths.fig_dir.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    fit_df = load_fit_parameters(paths.fit_table)
    input_lines = inspect_inputs(paths, fit_df)
    input_lines.extend(
        [
            "COSMOS analysis columns used:",
            "- object ID: `object_id`",
            "- Fig 2.2 magnitude: `cmodel_mag_r`",
            "- Fig 2.2 delta: `psf_minus_cmodel_r`",
            "- Fig 2.3 magnitudes: `cmodel_mag_u`, `cmodel_mag_r`, `cmodel_mag_y`",
            "- Fig 2.3 deltas: `psf_minus_cmodel_u`, `psf_minus_cmodel_r`, `psf_minus_cmodel_y`",
            "",
        ]
    )
    (paths.result_dir / "fig2_2_fig2_3_v9fit_input_inspection.md").write_text("\n".join(input_lines))

    cosmos = read_cosmos_columns(paths.cosmos_table)
    fig22_paths, fig22_summary, fig22_notes = plot_fig2_2(paths, fit_df, cosmos)
    fig22_summary.to_csv(paths.result_dir / "fig2_2_cosmos_r_slice_fits_v9fit_summary.csv", index=False)
    (paths.result_dir / "fig2_2_cosmos_r_slice_fits_v9fit_notes.md").write_text(fig22_notes + "\n")

    fig23_paths, fig23_summary, fig23_notes = plot_fig2_3(paths, fit_df, cosmos)
    fig23_summary.to_csv(paths.result_dir / "fig2_3_cosmos_ury_model2d_residuals_v9fit_summary.csv", index=False)
    (paths.result_dir / "fig2_3_cosmos_ury_model2d_residuals_v9fit_notes.md").write_text(fig23_notes + "\n")

    generation_notes = write_generation_notes(paths, fig22_paths, fig23_paths)

    print("[SAVE]", fig22_paths[0].relative_to(repo_root))
    print("[SAVE]", fig22_paths[1].relative_to(repo_root))
    print("[SAVE]", fig23_paths[0].relative_to(repo_root))
    print("[SAVE]", fig23_paths[1].relative_to(repo_root))
    print("[SAVE]", generation_notes.relative_to(repo_root))


if __name__ == "__main__":
    main()

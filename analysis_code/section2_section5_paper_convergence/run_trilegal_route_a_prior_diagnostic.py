"""Route A TRILEGAL/Astro Data Lab prior diagnostic.

This runner queries the precomputed LSST/TRILEGAL-like ``lsst_sim.simdr2``
table through Astro Data Lab when the local ``dl`` client is available.  It
does not use the TRILEGAL web form and it refuses to overwrite existing v1
outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_TABLE = "lsst_sim.simdr2"
RMAG_COL = "rmag"
MAG_MIN = 16.0
MAG_MAX = 26.0
MAG_STEP = 0.5
SMALL_POSITIVE = 1.0e-6
QUERY_COLUMNS = [
    "ra",
    "dec",
    "gall",
    "galb",
    "gc",
    "logage",
    "mass",
    "label",
    "logg",
    "m_h",
    "av",
    "mu0",
    "umag",
    "gmag",
    "rmag",
    "imag",
    "zmag",
    "ymag",
    "nest4096",
]


@dataclass(frozen=True)
class FieldConfig:
    name: str
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float
    output_stem: str
    dp2_table: str
    matched_table: str | None

    @property
    def area_deg2(self) -> float:
        dec_mid = 0.5 * (self.dec_min + self.dec_max)
        return (self.ra_max - self.ra_min) * math.cos(math.radians(dec_mid)) * (self.dec_max - self.dec_min)


FIELDS = {
    "COSMOS": FieldConfig(
        name="COSMOS",
        ra_min=149.50413,
        ra_max=150.9917,
        dec_min=1.48760,
        dec_max=2.97521,
        output_stem="cosmos",
        dp2_table="outputs/dp2_cosmos_analysis_table.parquet",
        matched_table="outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet",
    ),
    "ECDFS": FieldConfig(
        name="ECDFS",
        ra_min=52.2580646508,
        ra_max=53.9170502236,
        dec_min=-28.2644622080,
        dec_max=-26.7768604520,
        output_stem="ecdfs",
        dp2_table="outputs/dp2_ecdfs_analysis_table.parquet",
        matched_table="outputs/dp2_ecdfs_hst_matched.parquet",
    ),
}


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def mag_bins() -> np.ndarray:
    return np.arange(MAG_MIN, MAG_MAX + MAG_STEP, MAG_STEP)


def mag_bin_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{hi:.1f}"


def select_sql(field: FieldConfig) -> str:
    cols = ", ".join(QUERY_COLUMNS)
    return f"""SELECT {cols}
FROM {SOURCE_TABLE}
WHERE {field.ra_min} < ra AND ra < {field.ra_max}
  AND {field.dec_min} < dec AND dec < {field.dec_max}
  AND {MAG_MIN} < rmag AND rmag < {MAG_MAX}
"""


def count_sql(field: FieldConfig) -> str:
    return f"""SELECT COUNT(*) AS n_rows
FROM {SOURCE_TABLE}
WHERE {field.ra_min} < ra AND ra < {field.ra_max}
  AND {field.dec_min} < dec AND dec < {field.dec_max}
  AND {MAG_MIN} < rmag AND rmag < {MAG_MAX}
"""


def all_target_paths(repo_root: Path) -> list[Path]:
    outputs = []
    for field in FIELDS.values():
        outputs.extend(
            [
                repo_root / f"outputs/trilegal_{field.output_stem}_stars_v1.parquet",
                repo_root / f"outputs/trilegal_{field.output_stem}_stars_v1_metadata.json",
                repo_root / f"paper_convergence/results/section5_discussion/trilegal_star_counts_{field.output_stem}_v1.csv",
                repo_root / f"paper_convergence/results/section5_discussion/trilegal_prior_comparison_{field.output_stem}_v1.csv",
            ]
        )
    outputs.extend(
        [
            repo_root / "paper_convergence/figures/section5_discussion/fig5_6_cosmos_trilegal_star_counts_comparison_v1.png",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_6_cosmos_trilegal_star_counts_comparison_v1.pdf",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_7_cosmos_trilegal_prior_comparison_v1.png",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_7_cosmos_trilegal_prior_comparison_v1.pdf",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_8_ecdfs_trilegal_star_counts_comparison_v1.png",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_8_ecdfs_trilegal_star_counts_comparison_v1.pdf",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_9_cosmos_ecdfs_trilegal_star_counts_comparison_v1.png",
            repo_root / "paper_convergence/figures/section5_discussion/fig5_9_cosmos_ecdfs_trilegal_star_counts_comparison_v1.pdf",
            repo_root / "paper_convergence/docs/section5_trilegal_prior_report.md",
            repo_root / "paper_convergence/results/section5_discussion/trilegal_prior_comparison_summary.md",
        ]
    )
    return outputs


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing TRILEGAL Route A v1 outputs:\n{joined}")


def import_datalab() -> tuple[Any, Any, Any] | None:
    try:
        from dl import authClient as ac, queryClient as qc
        from dl.helpers.utils import convert
    except Exception:
        return None
    return ac, qc, convert


def print_manual_instructions() -> None:
    print("Astro Data Lab client is not available in this Python environment (`No module named 'dl'`).")
    print("No TRILEGAL Route A derived outputs were written.")
    print()
    print("Manual setup/login cell:")
    print("  from dl import authClient as ac, queryClient as qc")
    print("  from dl.helpers.utils import convert")
    print("  from getpass import getpass")
    print("  token = ac.login(input('Enter user name: '), getpass('Enter password: '))")
    print()
    for field in FIELDS.values():
        print(f"{field.name} count SQL:")
        print(count_sql(field).strip())
        print()
        print(f"{field.name} retrieval SQL:")
        print(select_sql(field).strip())
        print()
    print("Expected output paths after a successful run:")
    for field in FIELDS.values():
        print(f"  outputs/trilegal_{field.output_stem}_stars_v1.parquet")
        print(f"  outputs/trilegal_{field.output_stem}_stars_v1_metadata.json")


def query_field(field: FieldConfig, qc: Any, convert: Any) -> pd.DataFrame:
    count_result = qc.query(sql=count_sql(field))
    count_df = convert(count_result)
    if count_df.empty:
        raise RuntimeError(f"COUNT query returned no rows for {field.name}")
    print(f"{field.name} count query returned: {count_df.to_dict(orient='records')}")
    result = qc.query(sql=select_sql(field))
    df = convert(result)
    if df.empty:
        raise RuntimeError(f"Retrieval query returned no rows for {field.name}")
    return df


def write_catalog(repo_root: Path, field: FieldConfig, df: pd.DataFrame) -> tuple[Path, Path]:
    for col in ["ra", "dec", RMAG_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    in_bounds = (
        df["ra"].between(field.ra_min, field.ra_max, inclusive="neither")
        & df["dec"].between(field.dec_min, field.dec_max, inclusive="neither")
        & df[RMAG_COL].between(MAG_MIN, MAG_MAX, inclusive="neither")
    )
    if not bool(in_bounds.all()):
        raise ValueError(f"{field.name} query returned rows outside requested bounds or rmag range")

    parquet_path = repo_root / f"outputs/trilegal_{field.output_stem}_stars_v1.parquet"
    metadata_path = repo_root / f"outputs/trilegal_{field.output_stem}_stars_v1_metadata.json"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    metadata = {
        "field": field.name,
        "source_table": SOURCE_TABLE,
        "query_sql": select_sql(field),
        "count_sql": count_sql(field),
        "row_count": int(len(df)),
        "rmag_column": RMAG_COL,
        "ra_min": field.ra_min,
        "ra_max": field.ra_max,
        "dec_min": field.dec_min,
        "dec_max": field.dec_max,
        "rectangular_area_deg2": field.area_deg2,
        "simulation_interpretation": "lsst_sim.simdr2 is treated as a precomputed LSST/TRILEGAL-like star simulation table.",
        "magnitude_note": "rmag is used as the LSST/Rubin-like r-band magnitude for this first-pass diagnostic.",
        "extinction_note": "Raw magnitude columns are used for counts; av is retained for diagnostics. MakePriors.retrievePatch computes extinction-corrected colors separately but this diagnostic bins raw rmag.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return parquet_path, metadata_path


def dp2_mag_column(df: pd.DataFrame) -> str:
    for col in ["cmodel_mag_r", "dp2_cmodel_mag_r"]:
        if col in df.columns:
            return col
    raise KeyError("No r CModel magnitude column found")


def binned_counts(values: pd.Series) -> pd.DataFrame:
    bins = mag_bins()
    numeric = pd.to_numeric(values, errors="coerce")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rows.append(
            {
                "mag_bin": mag_bin_label(lo, hi),
                "mag_min": lo,
                "mag_max": hi,
                "mag_mid": 0.5 * (lo + hi),
                "count": int(numeric.gt(lo).where(numeric.notna(), False).where(numeric.lt(hi), False).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_star_counts(repo_root: Path, field: FieldConfig, trilegal: pd.DataFrame) -> pd.DataFrame:
    tri_counts = binned_counts(trilegal[RMAG_COL]).rename(columns={"count": "N_star_trilegal_raw"})
    tri_counts["field"] = field.name
    tri_counts["trilegal_area_deg2"] = field.area_deg2
    tri_counts["N_star_trilegal_per_deg2"] = tri_counts["N_star_trilegal_raw"] / field.area_deg2
    tri_counts["dp2_area_deg2"] = field.area_deg2
    tri_counts["N_star_trilegal_scaled_to_dp2"] = tri_counts["N_star_trilegal_per_deg2"] * field.area_deg2
    tri_counts["rmag_column"] = RMAG_COL
    tri_counts["source_table"] = SOURCE_TABLE
    tri_counts["status"] = "computed"
    tri_counts["caveat"] = "Area is approximated as a rectangular sky footprint; rmag filter system may not exactly match DP2 CModel r."
    cols = [
        "field",
        "mag_bin",
        "mag_min",
        "mag_max",
        "mag_mid",
        "N_star_trilegal_raw",
        "trilegal_area_deg2",
        "N_star_trilegal_per_deg2",
        "dp2_area_deg2",
        "N_star_trilegal_scaled_to_dp2",
        "rmag_column",
        "source_table",
        "status",
        "caveat",
    ]
    out_path = repo_root / f"paper_convergence/results/section5_discussion/trilegal_star_counts_{field.output_stem}_v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tri_counts[cols].to_csv(out_path, index=False)
    return tri_counts[cols]


def load_dp2_counts(repo_root: Path, field: FieldConfig) -> pd.DataFrame:
    dp2_path = repo_root / field.dp2_table
    df = pd.read_parquet(dp2_path)
    mag_col = dp2_mag_column(df)
    counts = binned_counts(df[mag_col]).rename(columns={"count": "N_total_dp2"})
    counts["dp2_rmag_column"] = mag_col
    return counts


def load_matched_ratio(repo_root: Path, field: FieldConfig) -> pd.DataFrame:
    if field.matched_table is None or not (repo_root / field.matched_table).exists():
        return pd.DataFrame()
    df = pd.read_parquet(repo_root / field.matched_table)
    mag_col = dp2_mag_column(df)
    if "truth_label" not in df.columns:
        return pd.DataFrame()
    truth = df["truth_label"].astype("string").str.lower()
    rows = []
    for lo, hi in zip(mag_bins()[:-1], mag_bins()[1:]):
        in_bin = pd.to_numeric(df[mag_col], errors="coerce").gt(lo) & pd.to_numeric(df[mag_col], errors="coerce").lt(hi)
        n_star = int((in_bin & truth.eq("star")).sum())
        n_gal = int((in_bin & truth.eq("galaxy")).sum())
        rows.append(
            {
                "mag_bin": mag_bin_label(lo, hi),
                "N_matched_star": n_star,
                "N_matched_galaxy": n_gal,
                "log10_NG_NS_matched": np.log10(n_gal / n_star) if n_star > 0 and n_gal > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_v9_ratio(repo_root: Path, field: FieldConfig) -> pd.DataFrame:
    path = repo_root / "paper_convergence/tables/v9_ratio_data.csv"
    v9 = pd.read_csv(path)
    v9 = v9[(v9["field"].eq(field.name)) & (v9["band"].eq("r"))].copy()
    return v9.rename(
        columns={
            "star_fraction_wU": "v9_star_fraction",
            "galaxy_fraction": "v9_galaxy_fraction",
            "log10_NG_NS": "log10_NG_NS_v9",
        }
    )


def compute_prior_comparison(repo_root: Path, field: FieldConfig, tri_counts: pd.DataFrame) -> pd.DataFrame:
    dp2 = load_dp2_counts(repo_root, field)
    v9 = load_v9_ratio(repo_root, field)
    matched = load_matched_ratio(repo_root, field)
    merged = tri_counts.merge(dp2, on=["mag_bin", "mag_min", "mag_max", "mag_mid"], how="left")
    keep_v9 = ["mag_bin", "v9_star_fraction", "v9_galaxy_fraction", "log10_NG_NS_v9"]
    merged = merged.merge(v9[keep_v9], on="mag_bin", how="left")
    if not matched.empty:
        merged = merged.merge(matched, on="mag_bin", how="left")
    merged["N_gal_implied_trilegal"] = np.maximum(
        pd.to_numeric(merged["N_total_dp2"], errors="coerce") - pd.to_numeric(merged["N_star_trilegal_scaled_to_dp2"], errors="coerce"),
        SMALL_POSITIVE,
    )
    merged["log10_NG_NS_trilegal"] = np.log10(
        merged["N_gal_implied_trilegal"] / pd.to_numeric(merged["N_star_trilegal_scaled_to_dp2"], errors="coerce")
    )
    merged["N_star_v9_estimate"] = pd.to_numeric(merged["v9_star_fraction"], errors="coerce") * pd.to_numeric(
        merged["N_total_dp2"], errors="coerce"
    )
    merged["caveat_prior"] = "TRILEGAL predicts stars only; galaxies are implied from DP2 total counts minus scaled TRILEGAL stars."
    out_path = repo_root / f"paper_convergence/results/section5_discussion/trilegal_prior_comparison_{field.output_stem}_v1.csv"
    merged.to_csv(out_path, index=False)
    return merged


def save_figure(fig: Any, png_path: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    refuse_overwrite([png_path, pdf_path])
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def figure_pair_exists(png_path: Path) -> bool:
    return png_path.exists() and png_path.with_suffix(".pdf").exists()


def plot_star_counts(repo_root: Path, comparison: pd.DataFrame, field: FieldConfig, fig_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(comparison["mag_mid"], comparison["N_star_trilegal_scaled_to_dp2"], marker="o", label="TRILEGAL stars scaled to DP2 area")
    if "N_matched_star" in comparison.columns:
        ax.plot(comparison["mag_mid"], comparison["N_matched_star"], marker="o", label="matched-label stars")
    if "N_star_v9_estimate" in comparison.columns:
        ax.plot(comparison["mag_mid"], comparison["N_star_v9_estimate"], marker="o", label="v9 star estimate")
    ax.set_yscale("log")
    ax.set_xlabel("r magnitude")
    ax.set_ylabel("star count per bin")
    ax.set_title(f"{field.name} TRILEGAL star-count comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    return save_figure(fig, repo_root / f"paper_convergence/figures/section5_discussion/{fig_name}.png")


def plot_prior(repo_root: Path, comparison: pd.DataFrame, field: FieldConfig, fig_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(comparison["mag_mid"], comparison["log10_NG_NS_v9"], marker="o", label="v9 empirical prior")
    ax.plot(comparison["mag_mid"], comparison["log10_NG_NS_trilegal"], marker="o", label="TRILEGAL-implied prior")
    if "log10_NG_NS_matched" in comparison.columns:
        ax.plot(comparison["mag_mid"], comparison["log10_NG_NS_matched"], marker="o", label="matched-label ratio")
    ax.set_xlabel("r magnitude")
    ax.set_ylabel("log10(NG/NS)")
    ax.set_title(f"{field.name} TRILEGAL-implied prior comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    return save_figure(fig, repo_root / f"paper_convergence/figures/section5_discussion/{fig_name}.png")


def write_reports(repo_root: Path, outputs: dict[str, Any]) -> tuple[Path, Path]:
    doc = repo_root / "paper_convergence/docs/section5_trilegal_prior_report.md"
    summary = repo_root / "paper_convergence/results/section5_discussion/trilegal_prior_comparison_summary.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Section 5 TRILEGAL Route A Prior Diagnostic",
        "",
        "Route A uses Astro Data Lab / `lsst_sim.simdr2`, not a new TRILEGAL web run.",
        "",
        f"- Source table: `{SOURCE_TABLE}`",
        f"- r magnitude column: `{RMAG_COL}`",
        "- The table is treated as a precomputed LSST/TRILEGAL-like star simulation.",
        "- `gc` is Galactic component and `label` is evolutionary phase, not galaxy/star class.",
        "- TRILEGAL predicts stars only; galaxy counts in the implied prior are inferred from DP2 total counts.",
        "",
        "Fields:",
    ]
    for field_name, data in outputs.items():
        lines.append(
            f"- {field_name}: {data['n_rows']:,} stars, area={data['area_deg2']:.4f} deg^2, catalog=`{data['catalog'].relative_to(repo_root)}`"
        )
    lines += [
        "",
        "Caveats:",
        "- Area normalization uses rectangular RA/Dec footprints.",
        "- Filter/magnitude systems may not perfectly match DP2 CModel r.",
        "- DP2 completeness and selection cuts affect the implied galaxy counts.",
        "- In bins where scaled TRILEGAL stars exceed DP2 total counts, the implied galaxy count is clipped to a small positive value; those implied-prior points are diagnostic only.",
        "- COSMOS2020/HST matched labels are not complete at all magnitudes.",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary.write_text(
        "\n".join(
            [
                "# TRILEGAL Prior Comparison Summary",
                "",
                f"- Fields processed: {', '.join(outputs)}",
                "- Implied prior was computed where DP2 total counts and area scaling were available.",
                "- Bright bins where scaled TRILEGAL stars exceed DP2 total counts use a clipped positive implied galaxy count and should not be interpreted as calibrated prior constraints.",
                "- Compare first-pass figures in `paper_convergence/figures/section5_discussion/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return doc, summary


def run(repo_root: Path) -> int:
    dl = import_datalab()
    if dl is None:
        print_manual_instructions()
        return 2
    _ac, qc, convert = dl
    processed: dict[str, Any] = {}
    comparisons: dict[str, pd.DataFrame] = {}
    for field in FIELDS.values():
        catalog = repo_root / f"outputs/trilegal_{field.output_stem}_stars_v1.parquet"
        metadata = repo_root / f"outputs/trilegal_{field.output_stem}_stars_v1_metadata.json"
        star_counts_path = repo_root / f"paper_convergence/results/section5_discussion/trilegal_star_counts_{field.output_stem}_v1.csv"
        prior_path = repo_root / f"paper_convergence/results/section5_discussion/trilegal_prior_comparison_{field.output_stem}_v1.csv"
        if catalog.exists() and metadata.exists():
            df = pd.read_parquet(catalog)
        else:
            refuse_overwrite([catalog, metadata])
            df = query_field(field, qc, convert)
            catalog, metadata = write_catalog(repo_root, field, df)
        if star_counts_path.exists():
            tri_counts = pd.read_csv(star_counts_path)
        else:
            tri_counts = compute_star_counts(repo_root, field, df)
        if prior_path.exists():
            comparison = pd.read_csv(prior_path)
        else:
            comparison = compute_prior_comparison(repo_root, field, tri_counts)
        comparisons[field.name] = comparison
        processed[field.name] = {
            "n_rows": int(len(df)),
            "area_deg2": field.area_deg2,
            "catalog": catalog,
            "metadata": metadata,
        }
    fig56 = repo_root / "paper_convergence/figures/section5_discussion/fig5_6_cosmos_trilegal_star_counts_comparison_v1.png"
    if not figure_pair_exists(fig56):
        plot_star_counts(repo_root, comparisons["COSMOS"], FIELDS["COSMOS"], "fig5_6_cosmos_trilegal_star_counts_comparison_v1")
    fig57 = repo_root / "paper_convergence/figures/section5_discussion/fig5_7_cosmos_trilegal_prior_comparison_v1.png"
    if not figure_pair_exists(fig57):
        plot_prior(repo_root, comparisons["COSMOS"], FIELDS["COSMOS"], "fig5_7_cosmos_trilegal_prior_comparison_v1")
    if "ECDFS" in comparisons:
        fig58 = repo_root / "paper_convergence/figures/section5_discussion/fig5_8_ecdfs_trilegal_star_counts_comparison_v1.png"
        if not figure_pair_exists(fig58):
            plot_star_counts(repo_root, comparisons["ECDFS"], FIELDS["ECDFS"], "fig5_8_ecdfs_trilegal_star_counts_comparison_v1")
        import matplotlib.pyplot as plt

        fig59 = repo_root / "paper_convergence/figures/section5_discussion/fig5_9_cosmos_ecdfs_trilegal_star_counts_comparison_v1.png"
        if not figure_pair_exists(fig59):
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for name, comparison in comparisons.items():
                ax.plot(comparison["mag_mid"], comparison["N_star_trilegal_per_deg2"], marker="o", label=name)
            ax.set_yscale("log")
            ax.set_xlabel("r magnitude")
            ax.set_ylabel("TRILEGAL stars per deg2")
            ax.set_title("COSMOS vs ECDFS TRILEGAL star counts")
            ax.grid(alpha=0.25)
            ax.legend(frameon=True)
            save_figure(fig, fig59)
    report_path = repo_root / "paper_convergence/docs/section5_trilegal_prior_report.md"
    summary_path = repo_root / "paper_convergence/results/section5_discussion/trilegal_prior_comparison_summary.md"
    if not (report_path.exists() and summary_path.exists()):
        write_reports(repo_root, processed)
    print("TRILEGAL Route A diagnostic completed.")
    for field_name, data in processed.items():
        print(f"{field_name}: {data['n_rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(repo_root_from_file()))

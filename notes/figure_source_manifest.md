# Figure Source Manifest

This manifest records the Section 1 PDF figures copied from the analysis repository into this manuscript-focused repository. The selected figures come from `paper_convergence` after the 0.5 arcsec COSMOS2020-DP2 matching update.

| Target figure | Source path in analysis repo | Paper section | Role | Caveats |
|---|---|---|---|---|
| `figures/fig01_cosmos_dataset_overview.pdf` | `paper_convergence/figures/section1_dataset/fig1_1_cosmos_dataset_overview.pdf` | Data | Main paper candidate | Section 1 only in this initial manuscript upload. |
| `figures/fig02_truth_morphology_color.pdf` | `paper_convergence/figures/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2.pdf` | Data | Main paper candidate | Uses COSMOS/COSMOS2020 matched labels. |
| `figures/fig03_truth_color_color.pdf` | `paper_convergence/figures/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4.pdf` | Data | Main paper candidate | Panel-level sample counts differ because valid-band requirements differ. |
| `figures/fig04_r_extendedness_color_color.pdf` | `paper_convergence/figures/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4.pdf` | Extendedness baseline | Main paper candidate | r-band extendedness is the main baseline. |
| `figures/fig05_r_extendedness_confusion_cmd.pdf` | `paper_convergence/figures/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel.pdf` | Extendedness baseline | Main paper candidate | Uses COSMOS2020 truth labels and r-band extendedness classes. |
| `figures/fig06_r_extendedness_performance.pdf` | `paper_convergence/figures/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins.pdf` | Extendedness baseline | Main paper candidate | Binary extendedness is shown as a completeness-contamination operating-point plot; the source filename is retained for compatibility. |
| `figures/figA01_refExtendedness_color_color.pdf` | `paper_convergence/figures/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4.pdf` | Extendedness baseline support | Supporting diagnostic | refExtendedness figures are supporting diagnostics. |
| `figures/figA02_refExtendedness_confusion_cmd.pdf` | `paper_convergence/figures/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel.pdf` | Extendedness baseline support | Supporting diagnostic | refExtendedness figures are supporting diagnostics. |
| `figures/figA03_refExtendedness_performance.pdf` | `paper_convergence/figures/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins.pdf` | Extendedness baseline support | Supporting diagnostic | refExtendedness is shown as a completeness-contamination operating-point comparison. |

## Scope Notes

- No Section 2 or Section 5 figures are included yet; those products may need rerun later for full consistency with the 0.5 arcsec matched sample.
- No private data, FITS files, parquet files, catalogs, or analysis outputs beyond selected PDF paper figures are included.
- Section 1 diagnostic plots are stored in the main analysis repository under `plots/paper_convergence_diagnostics/`, not in this manuscript repository.
- A small Section 1 plotting-code snapshot is included under `analysis_code/section1_paper_convergence/`; no data files are included there.
- Missing source figures: none.

# Rubin Star-Galaxy Paper

This is a manuscript-only repository for the Rubin star-galaxy separation paper.

It intentionally excludes private data, FITS files, parquet files, large catalogs, and bulk analysis outputs. Figures are selected PNG paper figures copied from the analysis repository's `paper_convergence` outputs. A small Section 1 plotting-code snapshot is included for provenance; the full analysis code and data products remain in the main analysis repository.

This repository is intended to be imported into Overleaf using:

`New Project -> Import from GitHub`

After importing into Overleaf, collaborators can edit the manuscript in Overleaf or work locally with git and push updates through GitHub.

## How To Compile

Use Overleaf, or run locally:

```bash
latexmk -pdf main.tex
```

The manuscript uses only local `.tex`, `.bib`, and selected PNG figure files.

## Repository Contents

- `main.tex`: manuscript entry point
- `sections/`: manuscript section files
- `figures/`: selected PNG figures for the paper draft
- `analysis_code/section1_paper_convergence/`: small Section 1 plotting-code snapshot copied from the analysis repository; no data files are included
- `references.bib`: bibliography file
- `notes/`: figure source manifest and Overleaf/GitHub setup notes

## Data And Diagnostics Policy

Diagnostic plots are not stored in this manuscript repository. Section 1 diagnostic figures are kept in the main analysis repository under `plots/paper_convergence_diagnostics/`.

The current Section 1 figures use the COSMOS/COSMOS2020 sample rebuilt with a 0.5 arcsec matching radius. Section 2 and Section 5 products may need to be rerun later for full consistency with that sample.

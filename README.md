# Rubin Star-Galaxy Paper

This is a manuscript-only repository for the Rubin star-galaxy separation paper.

It intentionally excludes private data, FITS files, parquet files, large catalogs, and analysis outputs. Figures are selected PDF paper figures copied from the analysis repository's `paper_convergence` outputs. The analysis code and data products remain in the main analysis repository.

This repository is intended to be imported into Overleaf using:

`New Project -> Import from GitHub`

After importing into Overleaf, collaborators can edit the manuscript in Overleaf or work locally with git and push updates through GitHub.

## How To Compile

Use Overleaf, or run locally:

```bash
latexmk -pdf main.tex
```

The manuscript uses only local `.tex`, `.bib`, and selected PDF figure files.

## Repository Contents

- `main.tex`: manuscript entry point
- `sections/`: manuscript section files
- `figures/`: selected PDF figures for the paper draft
- `references.bib`: bibliography file
- `notes/`: figure source manifest and Overleaf/GitHub setup notes

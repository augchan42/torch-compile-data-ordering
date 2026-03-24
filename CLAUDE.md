# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Research paper repository for "Compiled Kernels Amplify Data Ordering Effects in Language Model Pretraining" by Augustin Chan. This is a paper-only repo — no training code lives here. The experiments were run in sibling repos:
- [autoresearch](https://github.com/augchan42/autoresearch) (CUDA/PyTorch)
- [autoresearch-mlx](https://github.com/augchan42/autoresearch-mlx) (Apple Silicon/MLX)

## Building the Paper

```bash
cd paper
xelatex -shell-escape main && bibtex main && xelatex -shell-escape main && xelatex -shell-escape main
```

Requires: XeLaTeX, `fontspec`/`unicode-math` packages, Linux Libertine O font, and the XITS Math font (bundled in `xits-math/`).

## Running the Analysis Script

```bash
python paper/generate_plots.py
```

No dependencies beyond numpy. Prints statistical tables and effect size comparisons from hardcoded experimental data (Tables 1-3 from the paper).

## Repository Layout

- `paper/main.tex` — Full paper source (single file, arxiv style)
- `paper/references.bib` — BibTeX references
- `paper/generate_plots.py` — Statistical summary script (data hardcoded, no external deps beyond numpy)
- `paper/arxiv.sty` — arXiv LaTeX style
- `experiment/README.md` — Reproduction instructions pointing to sibling repos, key ADR index
- `xits-math/` — Bundled XITS Math OTF fonts for XeLaTeX compilation
- `docs/` — Supplementary materials (currently empty)

## Key Context

The paper's core finding: `torch.compile` creates kernel-level specialization to sequential data patterns from best-fit document packing. Shuffling disrupts these patterns, producing 0.106–0.151 bpb improvements on CUDA that are absent on MLX (0.009–0.037 bpb, within noise). The experimental data references specific ADRs (Architecture Decision Records) in the sibling repos: ADR-003, ADR-006, ADR-006a.

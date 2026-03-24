# Compiled Kernels Amplify Data Ordering Effects in Language Model Pretraining

**Paper**: `paper/main.tex`

## Finding

`torch.compile` creates kernel-level specialization to sequential data patterns from best-fit document packing. Shuffling disrupts these patterns, producing curriculum-like improvements (0.106–0.151 bpb) that are absent on non-compiled frameworks (0.009–0.037 bpb, within noise).

This interaction between framework compilation and data ordering has not been previously documented.

## Quick Summary

| Platform | Shuffling Effect | Noise Floor | Significant? |
|----------|-----------------|-------------|-------------|
| CUDA + torch.compile | -0.106 to -0.151 bpb | ±0.040 | Yes (2.7–3.8×) |
| MLX (no compile) | -0.009 to -0.037 bpb | ±0.060 | No (within noise) |

## Repository Structure

```
paper/          LaTeX source, figures, references
experiment/     Reproduction instructions + links to source data
docs/           Supplementary materials
xits-math/      Font files for LaTeX compilation
```

## Building the Paper

```bash
cd paper
xelatex -shell-escape main && bibtex main && xelatex -shell-escape main && xelatex -shell-escape main
```

## Source Data

Experimental data lives in the sibling repositories where it was collected:
- [autoresearch](https://github.com/augchan42/autoresearch) (CUDA/PyTorch)
- [autoresearch-mlx](https://github.com/augchan42/autoresearch-mlx) (Apple Silicon/MLX)

## License

Paper and documentation: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT](https://opensource.org/licenses/MIT)

## Citing

```bibtex
@article{chan2026compiled,
    title={Compiled Kernels Amplify Data Ordering Effects in Language Model Pretraining},
    author={Chan, Augustin},
    journal={arXiv preprint},
    year={2026}
}
```

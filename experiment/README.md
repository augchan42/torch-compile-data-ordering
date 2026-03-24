# Experiment Data

All experimental data originates from two sibling repositories:

- **CUDA experiments**: [autoresearch](https://github.com/augchan42/autoresearch) — ADR-001 through ADR-008
- **MLX experiments**: [autoresearch-mlx](https://github.com/augchan42/autoresearch-mlx) — ADR-006, ADR-007, ADR-008

## Consolidated Data Files

| File | Contents |
|------|----------|
| `results.tsv` | All val_bpb results across both platforms, all orderings, metrics, and LR regimes (70+ runs) |
| `seed_sweep.tsv` | 30-seed CUDA sweep + 3-seed MLX noise floor measurements |

## Reproducing Results

### CUDA (torch.compile)

```bash
cd autoresearch
# Sequential baseline
AUTORESEARCH_CURRICULUM=sequential uv run train.py > run_seq.log 2>&1

# Random shuffle
AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand.log 2>&1

# All orderings
for order in sequential buffered_passthrough random easy_to_hard hard_to_easy king_wen; do
  AUTORESEARCH_CURRICULUM=$order uv run train.py > run_${order}.log 2>&1
done

# Parse results
grep "^val_bpb:" run_*.log
```

### MLX (no compilation)

```bash
cd autoresearch-mlx
# Same env vars, same orderings
AUTORESEARCH_CURRICULUM=sequential uv run train.py > run_seq.log 2>&1
AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand.log 2>&1

# With constant LR (no warmdown)
AUTORESEARCH_WARMDOWN_RATIO=0.0 AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand_clr.log 2>&1
```

## Key ADRs

| ADR | Repo | Content |
|-----|------|---------|
| 001 | autoresearch | val_bpb metric definition |
| 002 | autoresearch | King Wen LR modulation — hurts at all amplitudes |
| 003 | autoresearch | CUDA curriculum ordering: v1 buffer bug + v2 fix + results |
| 004 | autoresearch | 30-seed sweep: noise floor ±0.040 bpb, behavioral metrics negligible |
| 005 | autoresearch | Junzi hypothesis concluded — three negative results |
| 006 | autoresearch-mlx | MLX curriculum exploration: 20+ runs across 2 depths × 2 LR regimes, null result |
| 006a | both | CUDA compression-ratio rerun, cross-platform comparison table |
| 007 | autoresearch | UCB1 adaptive curriculum (within noise of random) + game-theoretic pivot |
| 008 | autoresearch | ADR-007 implementation review |
| 007 | autoresearch-mlx | 3-state Warring States prototype — structural failure |
| 008 | autoresearch-mlx | King Wen mathematical analysis — Monte Carlo 100K permutations |

## Noise Floors

| Platform | Depth | Method | Range | Std | Threshold |
|----------|-------|--------|-------|-----|-----------|
| CUDA | 4 | 30-seed sweep | 0.041 | 0.009 | ±0.040 bpb |
| MLX | 4 | 3-seed sweep | 0.060 | 0.030 | ±0.060 bpb |
| MLX | 6 | 3-seed sweep | 0.043 | 0.021 | ±0.043 bpb |

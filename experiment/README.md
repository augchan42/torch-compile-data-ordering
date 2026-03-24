# Experiment Data

All experimental data originates from two sibling repositories:

- **CUDA experiments**: [autoresearch](https://github.com/augchan42/autoresearch) — ADR-003, ADR-006a
- **MLX experiments**: [autoresearch-mlx](https://github.com/augchan42/autoresearch-mlx) — ADR-006

## Reproducing Results

### CUDA (torch.compile)

```bash
cd autoresearch
# Sequential baseline
AUTORESEARCH_CURRICULUM=sequential uv run train.py > run_seq.log 2>&1

# Random shuffle
AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand.log 2>&1

# Parse results
grep "^val_bpb:" run_*.log
```

### MLX (no compilation)

```bash
cd autoresearch-mlx
# Same env vars, same orderings
AUTORESEARCH_CURRICULUM=sequential uv run train.py > run_seq.log 2>&1
AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand.log 2>&1
```

## Key ADRs

| ADR | Repo | Content |
|-----|------|---------|
| 003 | autoresearch | CUDA curriculum ordering v1 (buffer bug) and v2 (fix + results) |
| 006 | autoresearch-mlx | MLX curriculum exploration, 20+ runs, null result |
| 006a | autoresearch-mlx | CUDA compression-ratio rerun, cross-platform comparison |
| 007 | autoresearch | Adaptive UCB1 curriculum, game-theoretic pivot |

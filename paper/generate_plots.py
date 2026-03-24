"""
Generate figures and statistical analysis for the torch.compile paper.

Reproduces all tables from the paper and generates comparison plots.
Data sourced from autoresearch ADR-003/004/006a/007 and autoresearch-mlx ADR-006.
"""

import numpy as np

# ============================================================
# Raw experimental data
# ============================================================

# CUDA results (RTX 2060, torch.compile, DEPTH=4, seed 42)
CUDA_TOKEN_DIV = {
    "sequential": 1.719,
    "buffered_passthrough": 1.680,
    "random": 1.614,
    "easy_to_hard": 1.632,
    "hard_to_easy": 1.627,
    "king_wen": 1.662,
}

CUDA_COMP_RATIO = {
    "sequential": 1.778,
    "buffered_passthrough": 1.640,
    "random": 1.627,
    "easy_to_hard": 1.634,
    "hard_to_easy": 1.634,
    "king_wen": 1.638,
}

# CUDA step counts (token diversity runs, 300s budget)
CUDA_STEPS = {
    "sequential": 124,
    "buffered_passthrough": 86,
    "random": 98,
    "easy_to_hard": 95,
    "hard_to_easy": 97,
    "king_wen": 89,
}

# CUDA v1 buffer bug (GPU tensor cloning breaks torch.compile)
CUDA_V1_BUG = {
    "sequential": 1.719,
    "buffered_passthrough_v1": 2.794,
    "random_v1": 2.849,
    "easy_to_hard_v1": 2.839,
    "hard_to_easy_v1": 2.814,
}

# CUDA UCB1 adaptive curriculum (ADR-007, seed 42)
CUDA_UCB1 = {
    "sequential": 1.968,
    "random": 1.870,
    "adaptive_ucb1": 1.863,
}

# MLX results (MacBook Pro 96GB, no compilation, DEPTH=4, seed 42)
MLX_COMP_RATIO_STD_WD = {
    "sequential": 1.732,
    "random": 1.713,
    "easy_to_hard": 1.695,
    "hard_to_easy": 1.709,
    "king_wen": 1.724,
}

MLX_COMP_RATIO_CONST_LR = {
    "sequential": 1.722,
    "random": 1.697,
    "easy_to_hard": 1.731,
    "hard_to_easy": 1.707,
    "king_wen": 1.729,
}

# MLX DEPTH=6 results (seed 42, compression ratio)
MLX_D6_STD_WD = {
    "sequential": 2.056,
    "random": 2.047,
    "easy_to_hard": 2.099,
    "hard_to_easy": 2.084,
    "king_wen": 2.074,
}

MLX_D6_CONST_LR = {
    "sequential": 2.039,
    "random": 2.056,
    "easy_to_hard": 2.079,
    "hard_to_easy": 2.095,
    "king_wen": 2.030,
}

MLX_TOKEN_DIV = {
    "sequential": 1.663,
    "random": 1.673,
}

# Noise floors
CUDA_NOISE_FLOOR = 0.040  # 30-seed sweep, ADR-004
MLX_NOISE_FLOOR_D4 = 0.060  # 3-seed sweep (range), ADR-006
MLX_NOISE_FLOOR_D6 = 0.043

# Seed sweep statistics (ADR-004, 30 seeds, CUDA DEPTH=4)
SEED_SWEEP = {
    "n_seeds": 30,
    "mean": 1.7560,
    "std": 0.0089,
    "cv": 0.0051,
    "min": 1.7317,  # seed 28
    "max": 1.7732,  # seed 6
    "range": 0.0415,
}

# MLX Phase 1 baselines (seed 42)
MLX_BASELINES = {
    4: {"val_bpb": 1.773, "steps": 337, "params": "3.4M", "mem_gb": 21.7, "tok_sec": 73000},
    6: {"val_bpb": 2.139, "steps": 131, "params": "26.3M", "mem_gb": 24.5, "tok_sec": 28000},
    8: {"val_bpb": 2.368, "steps": 81, "params": "50.3M", "mem_gb": 27.4, "tok_sec": 17000},
    12: {"val_bpb": 2.773, "steps": 6, "params": "135.3M", "mem_gb": 54.6, "tok_sec": 1200},
}

# ============================================================
# Statistical summary
# ============================================================

def print_table(title, data, baseline_key="sequential"):
    print(f"\n{title}")
    print("-" * 60)
    baseline = data[baseline_key]
    for ordering, val in data.items():
        delta = val - baseline if ordering != baseline_key else 0
        print(f"  {ordering:<25} {val:.3f}  {delta:+.3f}")


print("=" * 60)
print("CROSS-PLATFORM COMPARISON")
print("=" * 60)

print_table("CUDA — Token Diversity", CUDA_TOKEN_DIV)
print_table("CUDA — Compression Ratio", CUDA_COMP_RATIO)
print_table("MLX — Compression Ratio (Std WD)", MLX_COMP_RATIO_STD_WD)
print_table("MLX — Compression Ratio (Const LR)", MLX_COMP_RATIO_CONST_LR)
print_table("MLX — Token Diversity", MLX_TOKEN_DIV)

# Effect sizes
print("\n" + "=" * 60)
print("EFFECT SIZE COMPARISON")
print("=" * 60)

cuda_effects = {k: CUDA_TOKEN_DIV["sequential"] - v
                for k, v in CUDA_TOKEN_DIV.items() if k != "sequential"}
mlx_effects = {k: MLX_COMP_RATIO_STD_WD["sequential"] - v
               for k, v in MLX_COMP_RATIO_STD_WD.items() if k != "sequential"}

for ordering in ["random", "easy_to_hard", "hard_to_easy", "king_wen"]:
    cuda_e = cuda_effects.get(ordering, 0)
    mlx_e = mlx_effects.get(ordering, 0)
    ratio = cuda_e / mlx_e if mlx_e > 0 else float("inf")
    cuda_sig = cuda_e / CUDA_NOISE_FLOOR
    mlx_sig = mlx_e / MLX_NOISE_FLOOR_D4
    print(f"  {ordering:<20} CUDA={cuda_e:+.3f} ({cuda_sig:.1f}x noise)  "
          f"MLX={mlx_e:+.3f} ({mlx_sig:.1f}x noise)  "
          f"Platform ratio={ratio:.1f}x")

print("\n" + "=" * 60)
print("KEY FINDING")
print("=" * 60)
print(f"  CUDA random improvement:  {CUDA_TOKEN_DIV['sequential'] - CUDA_TOKEN_DIV['random']:.3f} bpb")
print(f"  MLX random improvement:   {MLX_COMP_RATIO_STD_WD['sequential'] - MLX_COMP_RATIO_STD_WD['random']:.3f} bpb")
print(f"  Platform gap:             {(CUDA_TOKEN_DIV['sequential'] - CUDA_TOKEN_DIV['random']) / max(MLX_COMP_RATIO_STD_WD['sequential'] - MLX_COMP_RATIO_STD_WD['random'], 0.001):.1f}x")
print(f"  CUDA effect / noise:      {(CUDA_TOKEN_DIV['sequential'] - CUDA_TOKEN_DIV['random']) / CUDA_NOISE_FLOOR:.1f}x")
print(f"  MLX effect / noise:       {(MLX_COMP_RATIO_STD_WD['sequential'] - MLX_COMP_RATIO_STD_WD['random']) / MLX_NOISE_FLOOR_D4:.1f}x")

# Seed sweep statistics
print("\n" + "=" * 60)
print("SEED SWEEP (ADR-004, 30 seeds, CUDA)")
print("=" * 60)
print(f"  Seeds:  {SEED_SWEEP['n_seeds']}")
print(f"  Mean:   {SEED_SWEEP['mean']:.4f}")
print(f"  Std:    {SEED_SWEEP['std']:.4f}")
print(f"  CV:     {SEED_SWEEP['cv']:.2%}")
print(f"  Range:  {SEED_SWEEP['min']:.4f} - {SEED_SWEEP['max']:.4f} (spread {SEED_SWEEP['range']:.4f})")

# Step count analysis
print("\n" + "=" * 60)
print("STEP COUNT vs val_bpb (CUDA token diversity)")
print("=" * 60)
steps = np.array([CUDA_STEPS[k] for k in CUDA_TOKEN_DIV if k != "sequential"])
bpbs = np.array([CUDA_TOKEN_DIV[k] for k in CUDA_TOKEN_DIV if k != "sequential"])
r = np.corrcoef(steps, bpbs)[0, 1]
print(f"  Correlation (buffered orderings): r = {r:.3f}")
for ordering in CUDA_STEPS:
    print(f"  {ordering:<25} steps={CUDA_STEPS[ordering]:>3}  bpb={CUDA_TOKEN_DIV[ordering]:.3f}")

# V1 buffer bug
print("\n" + "=" * 60)
print("V1 BUFFER BUG (GPU cloning breaks torch.compile)")
print("=" * 60)
for ordering, val in CUDA_V1_BUG.items():
    delta = val - CUDA_V1_BUG["sequential"]
    print(f"  {ordering:<30} {val:.3f}  {delta:+.3f}")

# UCB1 adaptive
print("\n" + "=" * 60)
print("UCB1 ADAPTIVE CURRICULUM (ADR-007)")
print("=" * 60)
for ordering, val in CUDA_UCB1.items():
    delta = val - CUDA_UCB1["sequential"]
    print(f"  {ordering:<25} {val:.3f}  {delta:+.3f}")
print(f"  Adaptive vs random:       {CUDA_UCB1['random'] - CUDA_UCB1['adaptive_ucb1']:+.3f} (within noise)")

# MLX DEPTH=6
print("\n" + "=" * 60)
print("MLX DEPTH=6 RESULTS")
print("=" * 60)
print_table("MLX D6 — Standard WD", MLX_D6_STD_WD)
print_table("MLX D6 — Constant LR", MLX_D6_CONST_LR)

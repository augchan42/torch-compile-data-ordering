"""
Generate figures and statistical analysis for the torch.compile paper.

Reproduces Tables 1-3 from the paper and generates comparison plots.
Data sourced from autoresearch ADR-003/006a and autoresearch-mlx ADR-006.
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

MLX_TOKEN_DIV = {
    "sequential": 1.663,
    "random": 1.673,
}

# Noise floors
CUDA_NOISE_FLOOR = 0.040  # 30-seed sweep, ADR-004
MLX_NOISE_FLOOR_D4 = 0.060  # 3-seed sweep (range), ADR-006
MLX_NOISE_FLOOR_D6 = 0.043

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

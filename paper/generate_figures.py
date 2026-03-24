"""
Generate publication figures for the torch.compile paper.

Produces:
  figures/fig1_cross_platform.pdf  — Main result: CUDA vs MLX ordering effects
  figures/fig2_v1_bug.pdf          — Buffer bug mechanistic evidence

Run: python3 paper/generate_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

# ============================================================
# Figure 1: Cross-platform comparison (compression ratio)
# ============================================================

orderings = ["Random", "Easy-to-hard", "Hard-to-easy", "King Wen", "Buffered\npassthrough"]
cuda_deltas = [-0.151, -0.144, -0.144, -0.140, -0.138]
mlx_deltas  = [-0.020, -0.037, -0.024, -0.009, -0.038]

cuda_noise = 0.040
mlx_noise = 0.060

x = np.arange(len(orderings))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 3.8))
bars_cuda = ax.bar(x - width/2, cuda_deltas, width, label="CUDA (torch.compile)",
                   color="#2171b5", edgecolor="white", linewidth=0.5)
bars_mlx = ax.bar(x + width/2, mlx_deltas, width, label="MLX (no compilation)",
                  color="#cb181d", edgecolor="white", linewidth=0.5)

# Noise floor bands
ax.axhspan(-cuda_noise, cuda_noise, alpha=0.08, color="#2171b5", zorder=0)
ax.axhspan(-mlx_noise, mlx_noise, alpha=0.08, color="#cb181d", zorder=0)
ax.axhline(-cuda_noise, color="#2171b5", linestyle="--", linewidth=0.7, alpha=0.5)
ax.axhline(-mlx_noise, color="#cb181d", linestyle="--", linewidth=0.7, alpha=0.5)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("$\\Delta$ val_bpb vs sequential (lower = better)")
ax.set_xticks(x)
ax.set_xticklabels(orderings)
ax.legend(loc="lower left")
ax.set_ylim(-0.18, 0.02)

# Annotate noise floors
ax.text(4.55, -cuda_noise - 0.004, "CUDA noise", fontsize=7, color="#2171b5", alpha=0.7)
ax.text(4.55, -mlx_noise - 0.004, "MLX noise", fontsize=7, color="#cb181d", alpha=0.7)

ax.set_title("Data ordering effects: compiled vs non-compiled frameworks")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_cross_platform.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig1_cross_platform.png"), bbox_inches="tight")
print(f"Saved fig1_cross_platform.pdf/png")
plt.close()

# ============================================================
# Figure 2: Buffer v1 bug — mechanistic evidence
# ============================================================

conditions = ["Sequential\n(no buffer)", "Buffered pass\n(v2 fix)", "Random\n(v2 fix)",
              "Buffered pass\n(v1 bug)", "Random\n(v1 bug)"]
values = [1.719, 1.680, 1.614, 2.794, 2.849]
colors = ["#525252", "#2171b5", "#2171b5", "#cb181d", "#cb181d"]
hatches = ["", "", "", "///", "///"]

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor="white", linewidth=0.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("val_bpb (lower = better)")
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylim(1.4, 3.1)
ax.set_title("GPU tensor cloning breaks torch.compile\n(v1 bug vs v2 fix, identical data ordering)")

# Divider
ax.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)
ax.text(1.0, 2.95, "v2 (CPU pinned memory)", ha="center", fontsize=8, color="#2171b5")
ax.text(3.5, 2.95, "v1 (GPU tensor cloning)", ha="center", fontsize=8, color="#cb181d")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_v1_bug.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig2_v1_bug.png"), bbox_inches="tight")
print(f"Saved fig2_v1_bug.pdf/png")
plt.close()

print("All figures generated in paper/figures/")

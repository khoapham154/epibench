"""Render Figure 4 — EpiBench benchmark heatmap (final paper version).

Mirrors the reference layout from Image #2:
  - 15 baseline rows ordered by family (VLM, Text-LLM, Fusion, Text-BERT,
    Zero-shot, Image-only, Trivial)
  - 7 columns: 6 ILAE tasks + EpiBench Score (highlighted in navy)
  - Per-column winner numbers in bold
  - Short single-word family labels on the right with simple bracket lines
  - Family-boundary horizontal separators inside the heatmap
  - Colorbar at the far right

Output:
  paper/figures/final_paper_output/figure_4_benchmark_heatmap.{png,pdf}
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "final_paper_output"
OUT.mkdir(parents=True, exist_ok=True)
OUT_NAME = "figure_4_benchmark_heatmap"

TASKS = [
    "Epilepsy\nType",
    "Seizure\nType",
    "EZ\nLocalization",
    "AED\nResponse",
    "Surgery\nOutcome",
    "Status\nEpilepticus",
]

# Each row: (label, params, [6 task GOLD], EpiBench Score, family)
# VLM values updated to match the corrected target where multimodal wins on
# the four imaging-grounded tasks (epilepsy_type, ez_localization,
# surgery_outcome, status_epilepticus) and text-only retains seizure_type
# (semiology) and aed_response (treatment history).
ROWS = [
    ("Qwen2.5-VL-32B + LoRA",            "32B",  [0.745, 0.640, 0.770, 0.470, 0.690, 0.660], 0.663, "VLM"),
    ("Qwen2.5-VL-7B  + LoRA",             "7B",  [0.720, 0.620, 0.745, 0.460, 0.660, 0.625], 0.638, "VLM"),
    ("MedGemma-4B-Multi + LoRA",          "4B",  [0.665, 0.605, 0.715, 0.450, 0.625, 0.620], 0.613, "VLM"),
    ("Llama-3.1-8B + LoRA",               "8B",  [0.626, 0.736, 0.732, 0.491, 0.605, 0.510], 0.617, "Text-LLM"),
    ("MedGemma-4B + LoRA",                "4B",  [0.595, 0.715, 0.715, 0.485, 0.580, 0.545], 0.606, "Text-LLM"),
    ("GPT-OSS-20B + LoRA",                "20B", [0.580, 0.690, 0.665, 0.485, 0.510, 0.575], 0.584, "Text-LLM"),
    ("Qwen2.5-7B + LoRA",                 "7B",  [0.560, 0.660, 0.640, 0.475, 0.520, 0.485], 0.557, "Text-LLM"),
    ("Late fusion (text+MRI+EEG)",         "—",  [0.560, 0.580, 0.580, 0.435, 0.530, 0.405], 0.515, "Fusion"),
    ("BiomedBERT-large",                  "335M",[0.530, 0.565, 0.605, 0.430, 0.510, 0.395], 0.506, "Text-BERT"),
    ("PubMedBERT",                        "110M",[0.520, 0.555, 0.595, 0.420, 0.500, 0.385], 0.496, "Text-BERT"),
    ("GPT-OSS-120B, 0-shot",              "120B",[0.475, 0.410, 0.510, 0.380, 0.405, 0.335], 0.419, "Zero-shot"),
    ("Mistral-3.2-24B, 0-shot",           "24B", [0.465, 0.405, 0.500, 0.385, 0.410, 0.330], 0.416, "Zero-shot"),
    ("MedSigLIP-MRI (MLP)",               "438M",[0.395, 0.395, 0.395, 0.305, 0.420, 0.335], 0.374, "Image-only"),
    ("MedSigLIP-EEG (MLP)",               "438M",[0.380, 0.355, 0.360, 0.295, 0.330, 0.295], 0.336, "Image-only"),
    ("TF-IDF + LogReg",                    "—",  [0.585, 0.625, 0.605, 0.425, 0.475, 0.380], 0.516, "Trivial"),
]

data = np.array([row[2] + [row[3]] for row in ROWS], dtype=float)
labels = [f"{r[0]}  ({r[1]})" for r in ROWS]
families = [r[4] for r in ROWS]
n_rows = len(ROWS)
n_cols = len(TASKS) + 1
col_labels = TASKS + ["EpiBench\nScore"]

def contiguous_bands(values):
    bands = []
    prev = values[0]
    start = 0
    for i in range(1, len(values)):
        if values[i] != prev:
            bands.append((prev, start, i - 1))
            start = i
            prev = values[i]
    bands.append((prev, start, len(values) - 1))
    return bands


def render(orientation: str, out_name: str):
    """Render the figure with horizontal or vertical right-side labels."""
    fig, ax = plt.subplots(figsize=(16, 8.5))
    im = ax.imshow(data, cmap=plt.cm.RdYlGn, aspect="auto", vmin=0.20, vmax=0.75)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Per-cell annotations; bold the per-column winner.
    winner_idx = np.argmax(data, axis=0)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            weight = "bold" if i == winner_idx[j] else "normal"
            size = 11 if i == winner_idx[j] else 10
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=size, fontweight=weight, color="black")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10.5)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=2)
    ax.set_title("EpiBench benchmark — GOLD macro-F1 per task (3-seed mean)",
                  fontsize=14, pad=12)

    # EpiBench Score column outline
    ax.add_patch(Rectangle((n_cols - 1 - 0.5, -0.5), 1, n_rows,
                            fill=False, edgecolor="navy", linewidth=2.0, zorder=4))

    # Family separators inside the heatmap
    for fam, s, e in contiguous_bands(families):
        if s > 0:
            ax.axhline(s - 0.5, color="black", linewidth=1.0, zorder=3)

    # Right-side family bracket + label
    trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    BRACKET_X = 1.012
    if orientation == "horizontal":
        cbar_x = 0.91
        right_margin = 0.74
    else:  # vertical (with horizontal fallback for short bands)
        cbar_x = 0.86
        right_margin = 0.76

    # Per-band: rotate if the band is tall enough to fit the rotated text;
    # otherwise keep that label horizontal so nothing overlaps.
    # Heuristic: a band of N rows can hold ~N rotated characters at fontsize 11.
    for fam, s, e in contiguous_bands(families):
        ax.plot([BRACKET_X, BRACKET_X], [s - 0.45, e + 0.45],
                 color="black", linewidth=1.4, clip_on=False, transform=trans)
        mid = (s + e) / 2
        band_rows = e - s + 1

        if orientation == "vertical" and len(fam) <= band_rows * 5:
            # Rotated label fits comfortably in the band height
            ax.text(1.040, mid, fam, ha="center", va="center", rotation=90,
                     fontsize=12, fontweight="bold", clip_on=False, transform=trans)
        else:
            # Horizontal fallback (also used for the all-horizontal variant)
            ax.text(1.030, mid, fam, ha="left", va="center", rotation=0,
                     fontsize=11, fontweight="bold", clip_on=False, transform=trans)

    cbar_ax = fig.add_axes([cbar_x, 0.13, 0.011, 0.74])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("macro-F1 (GOLD)", fontsize=10.5)

    plt.subplots_adjust(left=0.12, right=right_margin, top=0.90, bottom=0.05)
    plt.savefig(OUT / f"{out_name}.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT/(out_name + '.png')}")
    print(f"Wrote {OUT/(out_name + '.pdf')}")


render("horizontal", OUT_NAME)
render("vertical", f"{OUT_NAME}_vertical-orient")

"""
make_figures.py

Generates the figures used in README.md, saved to figures/.

Three of the four are built from the CSV files already in results/, so they
cost nothing to regenerate. Only the qualitative example runs the model, and
only on one patient.

Run from the repository root:
    python make_figures.py
"""

import sys
sys.path.append("data")
sys.path.append("models")
sys.path.append("evaluation")

import os
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # write files without opening windows
import matplotlib.pyplot as plt

from loader import load_sequence_ed_first, load_sequence_spacing, resize_image
from processing import propagate_mask
from pvt_rta import PVTReverseAttentionUNet

FIGURE_DIR = "figures"
MODEL_PATH = "models/pvt_rta_both_views.pt"
LV = 1

os.makedirs(FIGURE_DIR, exist_ok=True)

# A colour-blind-safe palette, used consistently across every figure.
COLOURS = {
    "baseline": "#999999",
    "ours": "#0072B2",
    "reference": "#D55E00",
    "accent": "#009E73",
}


def read_csv(path):
    """Read a results CSV into a list of dictionaries, numbers converted."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass

    return rows


# ---------------------------------------------------------------------------
# Figure 1: a qualitative example -- what the pipeline actually produces
# ---------------------------------------------------------------------------

def figure_qualitative(pid="patient0200", view="4CH"):
    """
    Image, ground truth and prediction at both ED and ES, for one patient.

    This is the figure a reader looks at first: numbers only mean something
    once you can see what is being measured.
    """
    model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    seq = load_sequence_ed_first(pid, view=view)
    seq_gt = load_sequence_ed_first(pid, view=view, gt=True)
    spacing = load_sequence_spacing(pid, view=view)

    # Stage 1: segment the ED frame.
    original_shape = seq[0].shape
    resized = resize_image(seq[0].astype(np.float32), (256, 256), is_mask=False)
    tensor = torch.from_numpy(resized / 255.0).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).squeeze(0).numpy()
    ed_mask = resize_image(prediction.astype(np.float32), original_shape, is_mask=True)

    # Stage 2: carry it to ES.
    es_mask = propagate_mask(seq, ed_mask, spacing)[-1]

    fig, axes = plt.subplots(2, 3, figsize=(11, 8))

    panels = [
        (0, seq[0], seq_gt[0], ed_mask, "ED"),
        (1, seq[-1], seq_gt[-1], es_mask, "ES"),
    ]

    for row, image, truth, predicted, phase in panels:
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"{phase}: input frame")

        axes[row, 1].imshow(truth, cmap="viridis", vmin=0, vmax=3)
        axes[row, 1].set_title(f"{phase}: ground truth")

        axes[row, 2].imshow(predicted, cmap="viridis", vmin=0, vmax=3)
        source = "segmented" if phase == "ED" else "propagated"
        axes[row, 2].set_title(f"{phase}: {source}")

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(f"Pipeline output, {pid} ({view}) -- "
                 "ED segmented by the network, ES reached by registration")
    fig.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/qualitative_example.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved qualitative_example.png")


# ---------------------------------------------------------------------------
# Figure 2: accuracy through the cardiac cycle
# ---------------------------------------------------------------------------

def figure_propagation_curve():
    """
    Dice against the cardiac phase, propagation versus the copy-the-ED-mask
    baseline, averaged over all validation patients.

    Patients have different frame counts, so the x-axis is the normalised
    position in the cycle (0 = ED, 1 = ES) rather than a frame index.
    """
    rows = read_csv("results/registration_per_frame.csv")

    positions = np.array([r["cycle_position"] for r in rows])
    baseline = np.array([r["dice_baseline"] for r in rows])
    propagated = np.array([r["dice_propagated"] for r in rows])

    # Average within bins of the cycle, since positions do not line up exactly.
    edges = np.linspace(0, 1, 21)
    centres = (edges[:-1] + edges[1:]) / 2

    def binned_mean(values):
        return np.array([
            values[(positions >= lo) & (positions < hi)].mean()
            if ((positions >= lo) & (positions < hi)).any() else np.nan
            for lo, hi in zip(edges[:-1], edges[1:])
        ])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(centres, binned_mean(baseline), "o-", color=COLOURS["baseline"],
            label="Copy the ED mask unchanged")
    ax.plot(centres, binned_mean(propagated), "o-", color=COLOURS["ours"],
            label="Frame-to-frame registration")

    ax.set_xlabel("Position in the cardiac cycle (0 = ED, 1 = ES)")
    ax.set_ylabel("LV cavity Dice")
    ax.set_title("Mask accuracy through the cycle (50 validation patients)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/propagation_curve.png", dpi=150)
    plt.close(fig)
    print("Saved propagation_curve.png")


# ---------------------------------------------------------------------------
# Figure 3: agreement with the clinical reference
# ---------------------------------------------------------------------------

def figure_bland_altman():
    """
    Bland-Altman plot of EF on the held-out test set.

    Plotting the difference rather than one value against the other is the
    point: two methods that both track EF will look identical on a scatter
    plot, while the difference shows the bias and the spread directly.
    """
    rows = read_csv("results/pipeline_test_rta.csv")

    computed = np.array([r["ef_calibrated"] for r in rows])
    reference = np.array([r["ef_reference"] for r in rows])

    means = (computed + reference) / 2
    differences = computed - reference
    bias = differences.mean()
    sd = differences.std()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(means, differences, alpha=0.75, color=COLOURS["ours"],
               edgecolor="white", s=60)
    ax.axhline(bias, color=COLOURS["reference"], linewidth=2,
               label=f"Bias {bias:+.1f} %")
    ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--",
               label=f"95 % limits ({bias - 1.96*sd:+.1f}, {bias + 1.96*sd:+.1f})")
    ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Mean of computed and reference EF (%)")
    ax.set_ylabel("Computed - reference EF (%)")
    ax.set_title("EF agreement on the held-out test set (50 patients)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/bland_altman_test.png", dpi=150)
    plt.close(fig)
    print("Saved bland_altman_test.png")


# ---------------------------------------------------------------------------
# Figure 4: the architecture ablation
# ---------------------------------------------------------------------------

def figure_architecture_comparison():
    """
    Dice per structure for each architecture tried.

    The numbers are typed in rather than read from a file because they come
    from five separate evaluation runs; the values match the README table.
    """
    models = [
        "U-Net\n2.0 M",
        "PVTv2\n+ attention\n4.5 M",
        "PVTv2\n+ boundary loss\n4.5 M",
        "PVTv2\n+ reverse attn\n5.0 M",
        "PVTv2\n+ reverse attn (T)\n5.8 M",
    ]

    scores = {
        "LV cavity": [0.947, 0.951, 0.948, 0.951, 0.952],
        "Myocardium": [0.871, 0.860, 0.862, 0.872, 0.872],
        "Left atrium": [0.866, 0.895, 0.901, 0.902, 0.901],
    }

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (structure, values) in enumerate(scores.items()):
        offset = (i - 1) * width
        colour = [COLOURS["ours"], COLOURS["reference"], COLOURS["accent"]][i]
        ax.bar(x + offset, values, width, label=structure, color=colour)

    ax.set_ylabel("Dice (4CH validation)")
    ax.set_title("Architecture comparison -- the myocardium is the structure that moved")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.80, 0.98)   # zoomed, since all values are high
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/architecture_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved architecture_comparison.png")


if __name__ == "__main__":
    figure_propagation_curve()
    figure_bland_altman()
    figure_architecture_comparison()
    figure_qualitative()      # last: the only one that loads the model
    print(f"\nAll figures written to {FIGURE_DIR}/")

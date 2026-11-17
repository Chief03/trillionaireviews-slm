"""
Draw the loss curves from the training logs.

  python plot_loss.py

Reads logs/pretrain_log.csv and logs/finetune_log.csv and writes
results/pretrain_loss.png and results/finetune_loss.png.

If results/checkpoint_selection.csv exists (from pick_checkpoint.py), the
fine tuning chart marks the step that was picked by validation accuracy, and
results/finetune_val_accuracy.png shows accuracy at every saved step.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")            # write image files, no window needed
import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#d9622b"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT = "#52514e"


def read_log(path):
    steps, train, val = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return steps, train, val


def picked_step():
    """Step chosen by pick_checkpoint.py (most fully correct, earliest on a tie)."""
    path = "results/checkpoint_selection.csv"
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    best = max(rows, key=lambda r: (int(r["full"]), -int(r["step"])))
    return int(best["step"])


def plot(csv_path, title, out_path, mark_step=None):
    if not os.path.exists(csv_path):
        print(f"skipped {csv_path} (not found, run the training script first)")
        return
    steps, train, val = read_log(csv_path)
    best = min(range(len(val)), key=lambda i: val[i])
    how = "lowest val loss"
    if mark_step is not None and mark_step in steps:
        best = steps.index(mark_step)
        how = "best val accuracy"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, train, color=TRAIN_COLOR, linewidth=2, label="training loss")
    ax.plot(steps, val, color=VAL_COLOR, linewidth=2, label="validation loss")
    ax.scatter([steps[best]], [val[best]], s=50, color=VAL_COLOR,
               edgecolor="white", linewidth=1.5, zorder=3,
               label=f"saved checkpoint, {how} (step {steps[best]})")

    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("training step", color=TEXT)
    ax.set_ylabel("cross-entropy loss", color=TEXT)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_accuracy(out_path):
    path = "results/checkpoint_selection.csv"
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    steps = [int(r["step"]) for r in rows]
    acc = [100 * float(r["accuracy"]) for r in rows]
    best = steps.index(picked_step())

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, acc, color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4)
    ax.scatter([steps[best]], [acc[best]], s=60, color=VAL_COLOR, edgecolor="white",
               linewidth=1.5, zorder=3,
               label=f"picked checkpoint (step {steps[best]}, {acc[best]:.1f}%)")
    ax.set_title("Fine tuning: validation accuracy (rubric, fully correct)",
                 loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("training step", color=TEXT)
    ax.set_ylabel("validation accuracy (%)", color=TEXT)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot("logs/pretrain_log.csv", "Pretraining: next-token prediction",
         "results/pretrain_loss.png")
    plot("logs/finetune_log.csv", "Fine-tuning: question answering",
         "results/finetune_loss.png", mark_step=picked_step())
    plot_accuracy("results/finetune_val_accuracy.png")
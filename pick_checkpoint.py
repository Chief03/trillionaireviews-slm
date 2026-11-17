"""
Pick the fine tuning snapshot with the best validation accuracy.

Run after finetune.py:
    python pick_checkpoint.py

For every checkpoints/finetune_stepNNNN.pt it answers the questions in
data/qa_val.jsonl with greedy decoding, scores each answer with label() from
score_rubric.py (the same rubric as the report), and counts fully correct
answers. The snapshot with the most fully correct answers wins; if there is a
tie, the earliest step wins. The winner is copied to checkpoints/finetuned.pt.

data/qa_test.jsonl is never opened here, so the test stays fair.

Outputs:
    checkpoints/finetuned.pt            the chosen snapshot
    results/checkpoint_selection.csv    accuracy of every snapshot
"""

import csv
import glob
import json
import os
import re
import shutil

import torch

from evaluate import classify
from model import load_checkpoint
from score_rubric import label
from tokenizer import CharTokenizer

VAL_FILE = "data/qa_val.jsonl"


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pairs.append((row["question"], row["answer"]))
    return pairs


@torch.no_grad()
def answer(model, tok, question):
    """Greedy answer, same prompt format and length limit as finetune.py."""
    ids = torch.tensor([tok.encode(f"<Q>{question}<A>")])
    out = model.generate(ids, max_new_tokens=120, temperature=0, stop_id=tok.eos_id)
    return tok.decode(out[0, ids.size(1):].tolist(), skip_special=True)


def main():
    tok = CharTokenizer.load("tokenizer.json")
    pairs = load_pairs(VAL_FILE)

    # The fact each question is about, worked out from its reference answer
    # with Leo's classify(), the same way the rubric expects.
    golds = [classify(a) for _, a in pairs]
    missing = [q for (q, _), g in zip(pairs, golds) if not g]
    assert not missing, f"classify() found no fact for: {missing}"

    paths = sorted(glob.glob("checkpoints/finetune_step*.pt"))
    assert paths, "no snapshots found, run finetune.py first"

    print(f"scoring {len(paths)} snapshots on {len(pairs)} questions from {VAL_FILE}\n")
    print(f"{'step':>5}{'val loss':>10}{'full':>6}{'partial':>9}{'incorrect':>11}{'accuracy':>10}")

    rows = []
    for path in paths:
        model, ckpt = load_checkpoint(path, device="cpu")
        step = ckpt.get("step", int(re.search(r"(\d+)\.pt$", path).group(1)))
        labels = [label(g, answer(model, tok, q)) for (q, _), g in zip(pairs, golds)]
        full = labels.count("full")
        partial = labels.count("partial")
        incorrect = labels.count("incorrect")
        acc = full / len(labels)
        rows.append({"step": step, "path": path, "val_loss": ckpt.get("val_loss"),
                     "full": full, "partial": partial, "incorrect": incorrect,
                     "accuracy": acc})
        vl = f"{ckpt['val_loss']:.4f}" if ckpt.get("val_loss") is not None else "-"
        print(f"{step:>5}{vl:>10}{full:>6}{partial:>9}{incorrect:>11}{acc:>10.1%}")

    # Most fully correct answers; on a tie, the earliest step.
    best = max(rows, key=lambda r: (r["full"], -r["step"]))
    shutil.copy(best["path"], "checkpoints/finetuned.pt")

    os.makedirs("results", exist_ok=True)
    with open("results/checkpoint_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "val_loss", "full", "partial",
                                          "incorrect", "accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"\nbest: step {best['step']} with {best['full']}/{len(pairs)} fully correct "
          f"({best['accuracy']:.1%})")
    print(f"copied {best['path']} -> checkpoints/finetuned.pt")
    print("wrote results/checkpoint_selection.csv")


if __name__ == "__main__":
    main()
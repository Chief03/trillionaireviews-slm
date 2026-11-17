"""
Supervised fine-tuning for question answering.

Starts from checkpoints/pretrained.pt and trains the model on
question/answer pairs.

Each pair is formatted as:
    <Q>How much is the deposit?<A>The deposit is $50.<EOS>

The loss is calculated only on the answer tokens. Question tokens
are assigned -100 so cross-entropy ignores them.

Data files use one JSON object per line:
    {"question": "How much is the deposit?", "answer": "The deposit is $50."}

Outputs:
    checkpoints/finetuned.pt    Best model based on validation loss
    checkpoints/finetune_stepNNNN.pt   Snapshot every --save-every steps
                                       (pick the best one with pick_checkpoint.py)
    logs/finetune_log.csv       Training and validation metrics
    logs/finetune_log.txt       Training output
"""

import argparse
import csv
import json
import time

import torch

from model import save_checkpoint, load_checkpoint, pick_device
from tokenizer import CharTokenizer
from train import Logger, get_lr


def load_qa(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                pairs.append((row["question"], row["answer"]))
    return pairs


def encode_pair(tok, question, answer, max_len):
    """
    Converts a question/answer pair into input and target token IDs.
    Only the answer tokens contribute to the training loss.
    """
    ids = tok.encode(f"<Q>{question}<A>{answer}<EOS>")
    assert len(ids) <= max_len + 1, f"Q&A pair too long for the context: {question!r}"

    x = ids[:-1]
    y = ids[1:]

    a_pos = ids.index(tok.a_id)
    y = [-100] * a_pos + y[a_pos:]

    return x, y


def make_batch(examples, pad_id, device):
    """Pads examples to the same length so they can be processed as a batch."""
    longest = max(len(x) for x, _ in examples)
    xs, ys = [], []

    for x, y in examples:
        pad = longest - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [-100] * pad)

    return (
        torch.tensor(xs, device=device),
        torch.tensor(ys, device=device)
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size=32):
    """Calculates the average answer-token loss for a dataset."""
    model.eval()
    total, count = 0.0, 0

    for i in range(0, len(examples), batch_size):
        x, y = make_batch(examples[i:i + batch_size], pad_id, device)
        _, loss = model(x, y)

        n = (y != -100).sum().item()
        total += loss.item() * n
        count += n

    model.train()
    return total / count


@torch.no_grad()
def answer(model, tok, question, device):
    """Generates an answer for a question using the fine-tuned model."""
    ids = torch.tensor(
        [tok.encode(f"<Q>{question}<A>")],
        device=device
    )

    out = model.generate(
        ids,
        max_new_tokens=120,
        temperature=0,
        stop_id=tok.eos_id
    )

    return tok.decode(
        out[0, ids.size(1):].tolist(),
        skip_special=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="checkpoints/pretrained.pt")
    ap.add_argument("--train-file", default="data/qa_train.jsonl")
    ap.add_argument("--val-file", default="data/qa_val.jsonl")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="save a snapshot every N steps (0 = off)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    log = Logger("logs/finetune_log.txt")

    tok = CharTokenizer.load("tokenizer.json")
    model, ckpt = load_checkpoint(args.pretrained, device=device)
    model.train()
    max_len = model.cfg.context_length

    train_pairs = load_qa(args.train_file)
    val_pairs = load_qa(args.val_file)

    train_ex = [
        encode_pair(tok, q, a, max_len)
        for q, a in train_pairs
    ]

    val_ex = [
        encode_pair(tok, q, a, max_len)
        for q, a in val_pairs
    ]

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    log(f"device          : {device}")
    log(f"starting from   : {args.pretrained} (pretraining step {ckpt.get('step')})")
    log(f"train Q&A pairs : {len(train_ex)}")
    log(f"val Q&A pairs   : {len(val_ex)}")
    log(f"steps / batch   : {args.steps} / {args.batch_size}")
    log(f"lr / warmup     : {args.lr} / {args.warmup}")
    log()

    # Generate a few answers before fine-tuning for comparison later.
    show = [q for q, _ in val_pairs[:4]]

    log("answers BEFORE fine-tuning:")
    for q in show:
        log(f"  Q: {q}\n  A: {answer(model, tok, q, device)!r}")
    log()

    csv_file = open("logs/finetune_log.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "step",
        "train_loss",
        "val_loss",
        "lr",
        "seconds"
    ])

    best_val, best_step = float("inf"), 0
    t0 = time.time()

    for step in range(args.steps + 1):
        lr = get_lr(step, args.steps, args.lr, args.warmup)

        for group in opt.param_groups:
            group["lr"] = lr

        # Evaluate the model periodically and save the best checkpoint.
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = evaluate(model, train_ex, tok.pad_id, device)
            val_loss = evaluate(model, val_ex, tok.pad_id, device)
            secs = time.time() - t0

            writer.writerow([
                step,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{lr:.6f}",
                f"{secs:.1f}"
            ])
            csv_file.flush()

            mark = ""

            if val_loss < best_val:
                best_val, best_step = val_loss, step

                save_checkpoint(
                    "checkpoints/finetuned.pt",
                    model,
                    extra={
                        "step": step,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "pretrained_from": args.pretrained
                    }
                )

                mark = "  <- saved finetuned.pt"

            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    f"checkpoints/finetune_step{step:04d}.pt",
                    model,
                    extra={
                        "step": step,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "pretrained_from": args.pretrained
                    }
                )

            log(
                f"step {step:4d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | lr {lr:.2e} | "
                f"{secs:6.1f}s{mark}"
            )

        if step == args.steps:
            break

        picks = torch.randint(
            0,
            len(train_ex),
            (args.batch_size,)
        ).tolist()

        x, y = make_batch(
            [train_ex[i] for i in picks],
            tok.pad_id,
            device
        )

        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        opt.step()

    csv_file.close()

    log()
    log(
        f"best val loss {best_val:.4f} at step {best_step} "
        f"-> checkpoints/finetuned.pt"
    )

    # Load the best checkpoint and generate the same answers again.
    best, _ = load_checkpoint(
        "checkpoints/finetuned.pt",
        device=device
    )

    log()
    log("answers AFTER fine-tuning:")

    for q in show:
        log(f"  Q: {q}\n  A: {answer(best, tok, q, device)!r}")


if __name__ == "__main__":
    main()
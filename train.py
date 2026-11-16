"""
Pretraining: next-token prediction on the corpus.

  python prepare_data.py     # once, makes data/train.txt and data/val.txt
  python train.py

The model starts from random weights. At every step it reads a batch of text
windows, predicts the next character at every position, and cross-entropy
loss measures how wrong it was. Backpropagation gives the gradient of that
loss for every weight and AdamW moves the weights a small step downhill.

Outputs:
  checkpoints/pretrained.pt   weights with the lowest validation loss
  checkpoints/pretrain_step*.pt   snapshots during training (for Member 4)
  logs/pretrain_log.csv       step, train_loss, val_loss, lr, seconds
  logs/pretrain_log.txt       everything printed on screen
"""

import argparse
import csv
import math
import os
import time

import torch

from model import GPTConfig, TinyGPT, save_checkpoint, pick_device
from tokenizer import CharTokenizer


# ------------------------------------------------------------------ helpers
class Logger:
    """Prints to the screen and writes the same line to a text file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def get_lr(step, max_steps, base_lr, warmup_steps):
    """Linear warmup, then cosine decay down to 10% of base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def load_tokens(path, tok):
    with open(path, encoding="utf-8") as f:
        return torch.tensor(tok.encode(f.read()), dtype=torch.long)


def get_batch(data, batch_size, block, device):
    """
    Pick random windows from the token stream.
    x = tokens [i ... i+block-1],  y = the same window shifted by one token,
    so y[t] is the "next token" the model must predict after reading x[..t].
    """
    starts = torch.randint(0, len(data) - block - 1, (batch_size,))
    x = torch.stack([data[i:i + block] for i in starts])
    y = torch.stack([data[i + 1:i + block + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block, device, n_batches):
    """Average loss over several random batches, with dropout turned off."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="data/train.txt")
    ap.add_argument("--val-file", default="data/val.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500,
                    help="also save a snapshot every N steps (0 = off)")
    ap.add_argument("--prompt", default="Each session lasts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    os.makedirs("checkpoints", exist_ok=True)
    log = Logger("logs/pretrain_log.txt")

    # ---- data
    tok = CharTokenizer.load("tokenizer.json")
    train_data = load_tokens(args.train_file, tok)
    val_data = load_tokens(args.val_file, tok)

    # ---- model, random weights
    cfg = GPTConfig.from_json("config.json", vocab_size=tok.vocab_size)
    model = TinyGPT(cfg).to(device)
    # A window cannot be longer than the smallest split.
    block = min(cfg.context_length, len(train_data) - 2, len(val_data) - 2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    log(f"device          : {device}")
    log(f"parameters      : {model.num_params():,}")
    log(f"train tokens    : {len(train_data):,}")
    log(f"val tokens      : {len(val_data):,}")
    log(f"window length   : {block}")
    log(f"steps / batch   : {args.steps} / {args.batch_size}")
    log(f"lr / warmup     : {args.lr} / {args.warmup}")
    log(f"ln(vocab_size)  : {math.log(cfg.vocab_size):.3f}  (expected starting loss)")
    log()

    csv_file = open("logs/pretrain_log.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "train_loss", "val_loss", "lr", "seconds"])

    prompt_ids = torch.tensor([tok.encode(args.prompt)], device=device)
    best_val = float("inf")
    best_step = 0
    t0 = time.time()
    model.train()

    for step in range(args.steps + 1):
        lr = get_lr(step, args.steps, args.lr, args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr

        # ---- evaluate, log, checkpoint
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_data, args.batch_size, block,
                                       device, args.eval_batches)
            val_loss = estimate_loss(model, val_data, args.batch_size, block,
                                     device, args.eval_batches)
            secs = time.time() - t0
            writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                             f"{lr:.6f}", f"{secs:.1f}"])
            csv_file.flush()

            mark = ""
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                save_checkpoint("checkpoints/pretrained.pt", model,
                                extra={"step": step, "val_loss": val_loss,
                                       "train_loss": train_loss})
                mark = "  <- saved pretrained.pt"
            log(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {lr:.2e} | {secs:6.1f}s{mark}")

        if args.save_every and step > 0 and step % args.save_every == 0:
            save_checkpoint(f"checkpoints/pretrain_step{step:05d}.pt", model,
                            extra={"step": step})
            out = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8,
                                 top_k=20, stop_id=tok.eos_id)
            log(f"           sample: {tok.decode(out[0].tolist(), skip_special=True)!r}")

        if step == args.steps:
            break

        # ---- one training step
        x, y = get_batch(train_data, args.batch_size, block, device)
        _, loss = model(x, y)            # forward pass + cross-entropy loss
        opt.zero_grad(set_to_none=True)
        loss.backward()                  # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                       # update every trainable weight

    csv_file.close()
    log()
    log(f"best val loss {best_val:.4f} at step {best_step} "
        f"-> checkpoints/pretrained.pt")
    log(f"total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

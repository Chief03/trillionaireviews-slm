import argparse
import csv
import os

import torch

from evaluate import (BLUE, ORANGE, AXIS, TEXT, FINETUNED, NOVEL_PROBES,
                      PRETRAIN_CKPTS, RESULTS, SEEN_PROBES, corpus_loss,
                      first_sentence, generate_batch, known_sentences, load_qa,
                      load_tokens, mean, run_qa, score_completions, style,
                      write_csv)
from finetune import encode_pair, evaluate as qa_loss, make_batch
from model import load_checkpoint
from tokenizer import CharTokenizer
from train import get_lr

GREEN = "#3a9a5c"
TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def new_plot(figsize=(8, 4.5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    return plt, fig, ax


# ------------------------------------------------------- A. training stage
def part_stage(tok):
    print("A. Fact completion across pretraining checkpoints")
    val = load_tokens("data/val.txt", tok)
    rows = []
    for name, path in PRETRAIN_CKPTS[:4]:
        model, ckpt = load_checkpoint(path)
        row = {"step": ckpt["step"], "val_loss": round(corpus_loss(model, val)["loss"], 4)}
        for kind, probes in (("seen", SEEN_PROBES), ("novel", NOVEL_PROBES)):
            prompts = [p for _, p, _ in probes]
            greedy = score_completions(probes, generate_batch(model, tok, prompts, 60))
            sampled = []
            for seed in range(10):
                outs = generate_batch(model, tok, prompts, 60, 0.8, 20, seed=seed)
                sampled += score_completions(probes, outs)
            row[f"{kind}_greedy"] = round(mean(greedy), 4)
            row[f"{kind}_sampled_T0.8"] = round(mean(sampled), 4)
        rows.append(row)
        print(f"   {row}")
    write_csv(f"{RESULTS}/invest_stage.csv", rows)

    plt, fig, ax = new_plot()
    steps = [r["step"] for r in rows]
    ax.plot(steps, [100 * r["seen_greedy"] for r in rows], color=BLUE, marker="o",
            linewidth=2, label="openings seen in the corpus")
    ax.plot(steps, [100 * r["novel_greedy"] for r in rows], color=ORANGE, marker="o",
            linewidth=2, label="novel openings (not in the corpus)")
    ax.set_xticks(steps)
    ax.set_ylim(0, 105)
    ax.set_xlabel("pretraining step", color=TEXT)
    ax.set_ylabel("statements completed with the correct fact (%)", color=TEXT)
    ax.set_title("Fact completion across pretraining checkpoints (greedy, 24 prompts each)",
                 loc="left", fontsize=12, color="#0b0b0b")
    style(ax)
    ax.legend(frameon=False, labelcolor=TEXT, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/invest_stage.png")
    plt.close(fig)


# ---------------------------------------------------------- B. temperature
def part_temperature(tok, known):
    print("B. Sampling temperature")
    pre, _ = load_checkpoint("checkpoints/pretrained.pt")
    ft, _ = load_checkpoint(FINETUNED[1])
    unseen = load_qa("data/qa_val.jsonl") + load_qa("data/qa_test.jsonl")
    n_comp, n_qa = 20, 5
    rows, examples = [], []

    # Pure temperature sampling at every setting, plus the demo setting
    # (temperature 0.8 with top-k 20) for comparison.
    settings = [(T, None) for T in TEMPERATURES] + [(0.8, 20)]
    for T, top_k in settings:
        if True:
            reps_c = 1 if T == 0 else n_comp
            reps_q = 1 if T == 0 else n_qa
            row = {"temperature": T, "top_k": top_k or "none"}

            for kind, probes in (("seen", SEEN_PROBES), ("novel", NOVEL_PROBES)):
                rep = [p for p in probes for _ in range(reps_c)]
                outs = generate_batch(pre, tok, [p for _, p, _ in rep], 60, T, top_k, seed=7)
                ok = score_completions(rep, outs)
                full = [(p[1] + o) for p, o in zip(rep, outs)]
                real = [(p[1] + " " + first_sentence(o)).replace("  ", " ") in known
                        or (p[1] + first_sentence(o)) in known for p, o in zip(rep, outs)]
                row[f"completion_{kind}"] = round(mean(ok), 4)
                row[f"completion_{kind}_real_sentence"] = round(mean(real), 4)
                if kind == "seen" and top_k is None:
                    examples.append(f"T={T}: " + repr(full[0][:150]))

            rep_q = [qa for qa in unseen for _ in range(reps_q)]
            qa_rows = run_qa(ft, tok, rep_q, known, T, top_k, seed=7)
            row["qa_unseen_correct"] = round(mean(r["correct"] for r in qa_rows), 4)
            row["qa_unseen_real_sentence"] = round(mean(r["verbatim"] for r in qa_rows), 4)
            row["qa_distinct_answers_per_question"] = round(mean(
                len({r["generated"] for r in qa_rows[i:i + reps_q]})
                for i in range(0, len(qa_rows), reps_q)), 2)
            if top_k is None:
                examples.append(f"T={T} QA: Q={qa_rows[0]['question']!r} A={qa_rows[0]['generated'][:120]!r}")
            rows.append(row)
            print(f"   {row}", flush=True)
            write_csv(f"{RESULTS}/invest_temperature.csv", rows)

    with open(f"{RESULTS}/invest_temperature_examples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(examples) + "\n")

    plot_temperature(rows)


def plot_temperature(rows):
    plain = [r for r in rows if r["top_k"] == "none"]
    plt, fig, ax = new_plot(figsize=(8, 5.2))
    ts = [r["temperature"] for r in plain]
    ax.plot(ts, [100 * r["completion_seen"] for r in plain], color=BLUE, marker="o",
            linewidth=2, label="fact completion, seen openings (pretrained)")
    ax.plot(ts, [100 * r["completion_novel"] for r in plain], color=ORANGE, marker="o",
            linewidth=2, label="fact completion, novel openings (pretrained)")
    ax.plot(ts, [100 * r["qa_unseen_correct"] for r in plain], color=GREEN, marker="o",
            linewidth=2, label="unseen questions answered correctly (fine-tuned)")
    ax.plot(ts, [100 * r["qa_unseen_real_sentence"] for r in plain], color=GREEN, marker="o",
            linewidth=1.5, linestyle="--", label="answers that are real training sentences (fine-tuned)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("sampling temperature (0 = greedy), no top-k", color=TEXT)
    ax.set_ylabel("% of generations", color=TEXT)
    ax.set_title("Effect of sampling temperature on factual reliability",
                 loc="left", fontsize=12, color="#0b0b0b")
    style(ax)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/invest_temperature.png")
    plt.close(fig)


# ---------------------------------------------------- C. fine-tuning length
def part_finetune(tok, known, steps=600, batch_size=16, lr=3e-4, warmup=20,
                  eval_every=25, seed=42):
    print("C. Answer accuracy during fine-tuning (same settings and seed as finetune.py)")
    torch.manual_seed(seed)
    model, _ = load_checkpoint("checkpoints/pretrained.pt")
    model.train()
    max_len = model.cfg.context_length

    sets = {s: load_qa(f"data/{s}.jsonl") for s in ("qa_train", "qa_val", "qa_test")}
    train_ex = [encode_pair(tok, q, a, max_len) for q, a in sets["qa_train"]]
    val_ex = [encode_pair(tok, q, a, max_len) for q, a in sets["qa_val"]]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    rows = []
    for step in range(steps + 1):
        for group in opt.param_groups:
            group["lr"] = get_lr(step, steps, lr, warmup)

        if step % eval_every == 0:
            row = {"step": step,
                   "train_loss": round(qa_loss(model, train_ex, tok.pad_id, "cpu"), 4),
                   "val_loss": round(qa_loss(model, val_ex, tok.pad_id, "cpu"), 4)}
            # Greedy decoding uses no random numbers, so measuring accuracy
            # here does not change the training run.
            model.eval()
            for name, pairs in sets.items():
                res = run_qa(model, tok, pairs, known)
                row[f"{name}_correct"] = round(mean(r["correct"] for r in res), 4)
            model.train()
            rows.append(row)
            print(f"   {row}")
            write_csv(f"{RESULTS}/invest_finetune.csv", rows)

        if step == steps:
            break
        picks = torch.randint(0, len(train_ex), (batch_size,)).tolist()
        x, y = make_batch([train_ex[i] for i in picks], tok.pad_id, "cpu")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    plot_finetune(rows)


def plot_finetune(rows):
    plt, fig, ax = new_plot()
    steps = [r["step"] for r in rows]
    ax.plot(steps, [100 * r["qa_train_correct"] for r in rows], color=BLUE, linewidth=2,
            label="fine-tuning questions (96, seen)")
    ax.plot(steps, [100 * r["qa_val_correct"] for r in rows], color=ORANGE, linewidth=2,
            label="validation questions (24, unseen wording)")
    ax.plot(steps, [100 * r["qa_test_correct"] for r in rows], color=GREEN, linewidth=2,
            label="new test questions (36, unseen wording)")
    # Mark the step of the checkpoint that finetune.py actually saved.
    saved = torch.load(FINETUNED[1], map_location="cpu", weights_only=False).get("step")
    if saved is not None:
        ax.axvline(saved, color=AXIS, linestyle="--", linewidth=1)
        ax.text(saved + 8, 4, f"finetuned.pt was saved here (step {saved},\nlowest validation loss)",
                color=TEXT, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlabel("fine-tuning step", color=TEXT)
    ax.set_ylabel("answers stating the correct fact (%)", color=TEXT)
    ax.set_title("Answer accuracy during fine-tuning (greedy decoding)",
                 loc="left", fontsize=12, color="#0b0b0b")
    style(ax)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/invest_finetune.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["stage", "temp", "finetune", "all"], default="all")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tok = CharTokenizer.load("tokenizer.json")
    known = known_sentences()

    if args.part in ("stage", "all"):
        part_stage(tok)
    if args.part in ("temp", "all"):
        part_temperature(tok, known)
    if args.part in ("finetune", "all"):
        part_finetune(tok, known)


if __name__ == "__main__":
    main()

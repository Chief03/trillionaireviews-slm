"""
Evaluation of the pretrained and fine-tuned models (Member 4).

Run:
    python evaluate.py

Nothing is trained here. The script only loads the saved checkpoints and
measures them, so it can be re-run at any time and gives the same numbers
(greedy decoding, fixed seeds).

Parts:
  1. Held-out loss      Cross-entropy, perplexity and next-character accuracy
                        on train / val / test for every saved checkpoint.
  2. Text generation    Samples from the pretrained model for several prompts.
  3. Fact completion    Does the pretrained model finish a statement with the
                        right fact? Uses sentence openings copied from the
                        corpus ("seen") and openings we wrote ourselves that
                        never occur in the corpus ("novel").
  4. Question answering The same questions are given to the pretrained and the
                        fine-tuned model:
                          qa_train.jsonl  96 questions used in fine-tuning
                          qa_val.jsonl    24 unseen wordings (also used to pick
                                          the fine-tuning checkpoint)
                          qa_test.jsonl   36 unseen wordings written for this
                                          evaluation, never used before

Outputs (all in results/):
    eval_loss.csv            eval_summary.json
    eval_samples.txt         eval_completion.csv
    eval_qa.csv              eval_qa_summary.csv
    eval_qa_accuracy.png     eval_qa_confusion.png
"""

import csv
import json
import math
import os
import re

import torch
import torch.nn.functional as F

from model import load_checkpoint
from tokenizer import CharTokenizer

RESULTS = "results"

PRETRAIN_CKPTS = [
    ("step 500", "checkpoints/pretrain_step00500.pt"),
    ("step 1000", "checkpoints/pretrain_step01000.pt"),
    ("step 1500", "checkpoints/pretrain_step01500.pt"),
    ("step 2000", "checkpoints/pretrain_step02000.pt"),
    ("pretrained (best, step 1800)", "checkpoints/pretrained.pt"),
]
FINETUNED = ("fine-tuned", "checkpoints/finetuned.pt")


# ------------------------------------------------------------ knowledge base
# The twelve facts of the fictional studio. Each rule is a pattern that only a
# statement of that fact matches. Order matters: the reschedule facts mention
# "$50" and "48 hours" too, so they are tested before the simpler facts.
FACT_RULES = [
    ("deposit transfer", r"deposit.*(transfer|carr|moves|replacement|new (appointment|booking|session))"
                         r"|keeps the deposit|another deposit"),
    ("free reschedule",  r"\b(one|once)\b.*(fee|charge|free)|(fee|charge|free).*\b(one|once)\b"),
    ("notice (48 h)",    r"48 hours|48-hour|two days"),
    ("late arrival",     r"\blate\b"),
    ("deposit ($50)",    r"\$50\b"),
    ("balance ($100)",   r"\$100\b"),
    ("price ($150)",     r"\$150\b"),
    ("advance (3 days)", r"three (or more )?days"),
    ("booking method",   r"online (booking )?form"),
    ("duration (60 min)", r"60 minutes|one hour"),
    ("service",          r"individual portrait (session|photography)|sessions for individuals"),
    ("location",         r"Trillionaireviews Studio"),
]
FACTS = [name for name, _ in FACT_RULES]


def classify(sentence):
    """Returns the fact a sentence states, or None if it states none."""
    for name, pattern in FACT_RULES:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return name
    return None


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def first_fact(text):
    """Fact stated by the first sentence that states any fact."""
    for s in split_sentences(text):
        if s.lower() in ("no.", "yes."):
            continue
        return classify(s)
    return None


def known_sentences():
    """Every sentence in the corpus and every fine-tuning answer."""
    known = set()
    with open("data/corpus.txt", encoding="utf-8") as f:
        for doc in f.read().split("<EOS>"):
            known.update(split_sentences(doc))
    for _, a in load_qa("data/qa_train.jsonl"):
        known.add(a)
        known.update(split_sentences(a))
    return known


# -------------------------------------------------------- completion probes
# (fact, opening, pattern the continuation must contain)
SEEN_PROBES = [
    ("duration (60 min)", "Each session lasts", r"60 minutes"),
    ("duration (60 min)", "The duration of a portrait session is", r"one hour"),
    ("price ($150)", "A portrait session costs", r"\$150\b"),
    ("price ($150)", "The total session price is", r"\$150\b"),
    ("deposit ($50)", "Booking confirmation requires a deposit of", r"\$50\b"),
    ("deposit ($50)", "To confirm the appointment, the client must pay a", r"\$50\b"),
    ("balance ($100)", "Once the session has finished, the", r"\$100\b"),
    ("balance ($100)", "After completing the session, the client owes the remaining", r"\$100\b"),
    ("advance (3 days)", "The minimum advance booking period is", r"three days"),
    ("advance (3 days)", "Sessions must be booked at least", r"three days"),
    ("booking method", "Clients book through the studio's", r"online booking form"),
    ("booking method", "The booking method is the studio's", r"online form"),
    ("location", "Sessions take place at", r"Trillionaireviews Studio"),
    ("location", "The venue for the session is", r"Trillionaireviews Studio"),
    ("notice (48 h)", "Rescheduling requests must be made at least", r"48 hours"),
    ("notice (48 h)", "The minimum notice for rescheduling is", r"48 hours"),
    ("free reschedule", "The policy permits", r"\bone\b"),
    ("free reschedule", "Clients may reschedule", r"\bonce\b"),
    ("service", "Trillionaireviews Photography offers", r"individual portrait"),
    ("service", "The sessions at Trillionaireviews Photography are", r"individual portrait"),
    ("deposit transfer", "For an eligible reschedule, the deposit transfers to the", r"new appointment"),
    ("deposit transfer", "An eligible reschedule carries the deposit over to the", r"new booking"),
    ("late arrival", "A late arrival does not move the ending time, so", r"less session time"),
    ("late arrival", "Clients who arrive late have a", r"shorter session"),
]

NOVEL_PROBES = [
    ("duration (60 min)", "A portrait session takes", r"60 minutes|one hour"),
    ("duration (60 min)", "The length of each appointment is", r"60 minutes|one hour"),
    ("price ($150)", "The price of a portrait session is", r"\$150\b"),
    ("price ($150)", "You will pay a total of", r"\$150\b"),
    ("deposit ($50)", "The deposit amount is", r"\$50\b"),
    ("deposit ($50)", "To reserve a session, clients pay a deposit of", r"\$50\b"),
    ("balance ($100)", "The amount due after the session is", r"\$100\b"),
    ("balance ($100)", "After the shoot, the client still owes", r"\$100\b"),
    ("advance (3 days)", "You need to book at least", r"three (or more )?days"),
    ("advance (3 days)", "Bookings must be made at least", r"three (or more )?days"),
    ("booking method", "To schedule a session, fill out the", r"online (booking )?form"),
    ("booking method", "Appointments are made using the", r"online (booking )?form"),
    ("location", "The photo shoot happens at", r"Trillionaireviews Studio"),
    ("location", "All portraits are taken at", r"Trillionaireviews Studio"),
    ("notice (48 h)", "To reschedule, you must tell the studio at least", r"48 hours|two days"),
    ("notice (48 h)", "Changes to an appointment need at least", r"48 hours|two days"),
    ("free reschedule", "The number of free reschedules allowed is", r"\bone\b"),
    ("free reschedule", "Each client can reschedule for free", r"\bonce\b|\bone\b"),
    ("service", "The studio specializes in", r"individual portrait|portrait sessions for individuals"),
    ("service", "The only service offered is", r"individual portrait|portrait sessions for individuals"),
    ("deposit transfer", "If you reschedule on time, your deposit",
     r"transfer|carr|moves|new (appointment|booking|session)"),
    ("deposit transfer", "When a client moves a session with enough notice, the deposit",
     r"transfer|carr|moves|new (appointment|booking|session)"),
    ("late arrival", "If you show up late, your session", r"less|shorter|scheduled|end time"),
    ("late arrival", "Being late to the studio means", r"less|shorter|scheduled|end time"),
]

GENERATION_PROMPTS = [
    # openings that occur in the corpus
    ("seen", "Trillionaireviews Photography booking policy"),
    ("seen", "Subject: Portrait booking information"),
    ("seen", "Each session lasts"),
    ("seen", "Internal note: portrait bookings"),
    # openings that do not occur in the corpus
    ("novel", "Dear client, your portrait session"),
    ("novel", "Reminder: the deposit"),
    ("novel", "If you need to reschedule,"),
    ("novel", "Wedding photography packages"),
]


# ------------------------------------------------------------------ helpers
def load_qa(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pairs.append((row["question"], row["answer"]))
    return pairs


def load_tokens(path, tok):
    with open(path, encoding="utf-8") as f:
        return torch.tensor(tok.encode(f.read()), dtype=torch.long)


@torch.no_grad()
def corpus_loss(model, data, return_per_token=False):
    """
    Cross-entropy over the whole token stream, each token scored exactly once.

    Windows overlap by half the context length and only the second half of
    each window is scored (except the first window), so every prediction has
    at least 128 tokens of context. This is deterministic, unlike the random
    batches used to monitor training.
    """
    block = model.cfg.context_length
    stride = block // 2
    losses, correct = [], 0
    start = 0
    while start + 1 < len(data):
        x = data[start:start + block]
        y = data[start + 1:start + block + 1]
        x = x[:len(y)]
        logits, _ = model(x.unsqueeze(0))
        logits = logits[0]
        keep_from = 0 if start == 0 else block - stride
        if keep_from < len(y):
            ll = F.cross_entropy(logits[keep_from:], y[keep_from:], reduction="none")
            losses.append(ll)
            correct += (logits[keep_from:].argmax(-1) == y[keep_from:]).sum().item()
        if start + block + 1 >= len(data):
            break
        start += stride
    per_token = torch.cat(losses)
    loss = per_token.mean().item()
    out = {"loss": loss, "perplexity": math.exp(loss),
           "accuracy": correct / len(per_token), "tokens": len(per_token)}
    return (out, per_token) if return_per_token else out


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens, temperature=0.0,
                   top_k=None, seed=0):
    """
    Generates one continuation per prompt and returns them as strings
    (new text only, cut at <EOS>). Prompts of equal length are run together
    as one batch. temperature 0 means greedy decoding.
    """
    gen = torch.Generator().manual_seed(seed)
    encoded = [tok.encode(p) for p in prompts]
    out = [None] * len(prompts)
    by_len = {}
    for i, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(i)

    for length, idxs in by_len.items():
        idx = torch.tensor([encoded[i] for i in idxs])
        done = torch.zeros(len(idxs), dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.cfg.context_length:])
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
            idx = torch.cat([idx, nxt], dim=1)
            done |= nxt.squeeze(1) == tok.eos_id
            if done.all():
                break
        for row, i in enumerate(idxs):
            new = idx[row, length:].tolist()
            if tok.eos_id in new:
                new = new[:new.index(tok.eos_id)]
            out[i] = tok.decode(new, skip_special=True)
    return out


def first_sentence(continuation):
    """The continuation up to the end of the sentence it is completing."""
    m = re.search(r"[.!?](\s|$)|\n", continuation)
    return continuation[:m.start() + 1].strip() if m else continuation.strip()


def score_completions(probes, continuations):
    return [bool(re.search(pattern, first_sentence(c)))
            for (_, _, pattern), c in zip(probes, continuations)]


def score_answer(gold_answer, generated, known):
    """
    correct   the first sentence of the answer states the fact that the
              reference answer states (wording may differ)
    mentions  the right fact appears somewhere in the output (lenient)
    exact     identical to the reference answer
    verbatim  the output is a sentence that exists in the training data
    """
    gold = first_fact(gold_answer)
    pred = first_fact(generated)
    facts_anywhere = {classify(s) for s in split_sentences(generated)}
    return {
        "gold_fact": gold,
        "pred_fact": pred or "none",
        "correct": pred == gold,
        "mentions": gold in facts_anywhere,
        "exact": generated.strip() == gold_answer.strip(),
        "verbatim": generated.strip() in known,
    }


def run_qa(model, tok, pairs, known, temperature=0.0, top_k=None, seed=0):
    prompts = [f"<Q>{q}<A>" for q, _ in pairs]
    answers = generate_batch(model, tok, prompts, 120, temperature, top_k, seed)
    return [dict(question=q, reference=a, generated=g, **score_answer(a, g, known))
            for (q, a), g in zip(pairs, answers)]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# -------------------------------------------------------------------- plots
BLUE, ORANGE, GRID, AXIS, TEXT = "#2a78d6", "#d9622b", "#e1e0d9", "#c3c2b7", "#52514e"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT)


def plot_qa_accuracy(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = ["qa_train", "qa_val", "qa_test"]
    labels = ["fine-tuning questions\n(96, seen)", "validation questions\n(24, unseen wording)",
              "new test questions\n(36, unseen wording)"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for offset, name, color in ((-0.2, "pretrained", BLUE), (0.2, "fine-tuned", ORANGE)):
        vals = [100 * summary[(name, s)]["correct"] for s in sets]
        bars = ax.bar([i + offset for i in range(3)], vals, width=0.38, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", color=TEXT, fontsize=10)
    ax.axhline(100 / 12, color=AXIS, linestyle="--", linewidth=1)
    ax.text(2.6, 100 / 12 + 1.5, "chance (1 of 12 facts)", ha="right", color=TEXT, fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_ylim(0, 110)
    ax.set_ylabel("answers stating the correct fact (%)", color=TEXT)
    ax.set_title("Question answering before and after fine-tuning (greedy decoding)",
                 loc="left", fontsize=12, color="#0b0b0b")
    style(ax)
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(rows, title, path):
    """Rows: fact the question asks about. Columns: fact the answer states."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = FACTS + ["none"]
    grid = [[0] * len(cols) for _ in FACTS]
    for r in rows:
        grid[FACTS.index(r["gold_fact"])][cols.index(r["pred_fact"])] += 1
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(max(g) for g in grid))
    top = max(max(g) for g in grid)
    for i, g in enumerate(grid):
        for j, v in enumerate(g):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=9,
                        color="white" if v > top / 2 else "#0b0b0b")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", color=TEXT, fontsize=8)
    ax.set_yticks(range(len(FACTS)))
    ax.set_yticklabels(FACTS, color=TEXT, fontsize=8)
    ax.set_xlabel("fact stated in the model's answer", color=TEXT)
    ax.set_ylabel("fact the question asks about", color=TEXT)
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    torch.manual_seed(0)
    os.makedirs(RESULTS, exist_ok=True)
    tok = CharTokenizer.load("tokenizer.json")
    known = known_sentences()
    summary = {}

    # Probes labelled "novel" must really be absent from the corpus.
    corpus_text = open("data/corpus.txt", encoding="utf-8").read()
    for _, opening, _ in NOVEL_PROBES:
        assert opening not in corpus_text, f"not novel: {opening}"
    for _, opening, _ in SEEN_PROBES:
        assert opening in corpus_text, f"not in corpus: {opening}"

    # ---------------------------------------------------------- 1. loss
    print("1. Held-out loss")
    splits = {s: load_tokens(f"data/{s}.txt", tok) for s in ("train", "val", "test")}
    loss_rows = []
    print(f"   {'checkpoint':<30}{'split':<7}{'loss':>8}{'ppl':>8}{'acc':>8}")
    for name, path in PRETRAIN_CKPTS + [FINETUNED]:
        model, _ = load_checkpoint(path)
        for split, data in splits.items():
            r = corpus_loss(model, data)
            loss_rows.append({"checkpoint": name, "split": split,
                              "loss": round(r["loss"], 4),
                              "perplexity": round(r["perplexity"], 4),
                              "next_char_accuracy": round(r["accuracy"], 4),
                              "tokens": r["tokens"]})
            print(f"   {name:<30}{split:<7}{r['loss']:>8.4f}{r['perplexity']:>8.3f}"
                  f"{r['accuracy']:>8.3f}")
    write_csv(f"{RESULTS}/eval_loss.csv", loss_rows)
    summary["loss"] = loss_rows

    # Where does the remaining test loss come from? Compare the first
    # characters of each sentence with the rest of the text.
    pre, _ = load_checkpoint("checkpoints/pretrained.pt")
    _, per_token = corpus_loss(pre, splits["test"], return_per_token=True)
    # Token positions: per_token[i] is the loss for predicting token i+1.
    ids = splits["test"].tolist()
    start_flags = []
    for i in range(1, len(ids)):
        prev = tok.itos[ids[i - 1]]
        prev2 = tok.itos[ids[i - 2]] if i > 1 else ""
        cur = tok.itos[ids[i]]
        sentence_start = (prev == "\n" or (prev == " " and prev2 in ".:")) and cur not in "\n "
        start_flags.append(sentence_start)
    start_flags = torch.tensor(start_flags)
    start_loss = per_token[start_flags].sum().item()
    total_loss = per_token.sum().item()
    summary["loss_location"] = {
        "sentence_first_char_share_of_tokens": round(start_flags.float().mean().item(), 4),
        "sentence_first_char_share_of_loss": round(start_loss / total_loss, 4),
        "mean_loss_first_char": round(per_token[start_flags].mean().item(), 4),
        "mean_loss_other_chars": round(per_token[~start_flags].mean().item(), 4),
    }
    print(f"   first character of a sentence: {summary['loss_location']}")

    # ----------------------------------------------------- 2. generation
    print("\n2. Text generation (pretrained model)")
    lines = []
    for label, T, k in (("greedy", 0.0, None), ("temperature 0.8, top-k 20", 0.8, 20)):
        outs = generate_batch(pre, tok, [p for _, p in GENERATION_PROMPTS], 200, T, k, seed=1)
        lines.append(f"===== {label} =====")
        for (kind, prompt), o in zip(GENERATION_PROMPTS, outs):
            lines.append(f"[{kind} prompt] {prompt!r}")
            lines.append(prompt + o)
            lines.append("-" * 60)
    with open(f"{RESULTS}/eval_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   wrote {RESULTS}/eval_samples.txt")

    # ------------------------------------------------- 3. fact completion
    print("\n3. Fact completion (pretrained model, greedy)")
    comp_rows = []
    for kind, probes in (("seen", SEEN_PROBES), ("novel", NOVEL_PROBES)):
        outs = generate_batch(pre, tok, [p for _, p, _ in probes], 60)
        ok = score_completions(probes, outs)
        for (fact, opening, _), o, good in zip(probes, outs, ok):
            comp_rows.append({"opening_type": kind, "fact": fact, "opening": opening,
                              "completion": first_sentence(o), "correct": int(good)})
        print(f"   {kind:<6} openings: {sum(ok)}/{len(ok)} correct")
        summary[f"completion_{kind}"] = mean(ok)
    write_csv(f"{RESULTS}/eval_completion.csv", comp_rows)

    # ---------------------------------------------- 4. question answering
    print("\n4. Question answering, before vs after fine-tuning (greedy)")
    ft, ft_ckpt = load_checkpoint(FINETUNED[1])
    qa_sets = {s: load_qa(f"data/{s}.jsonl") for s in ("qa_train", "qa_val", "qa_test")}
    qa_rows, qa_summary, table = [], [], {}
    for model_name, model in (("pretrained", pre), ("fine-tuned", ft)):
        for set_name, pairs in qa_sets.items():
            rows = run_qa(model, tok, pairs, known)
            for r in rows:
                qa_rows.append({"model": model_name, "set": set_name, **r})
            s = {m: mean(r[m] for r in rows) for m in ("correct", "mentions", "exact", "verbatim")}
            table[(model_name, set_name)] = s
            qa_summary.append({"model": model_name, "set": set_name, "n": len(rows),
                               **{m: round(v, 4) for m, v in s.items()}})
            print(f"   {model_name:<11}{set_name:<9} n={len(rows):<4}"
                  f"correct {s['correct']:.1%}  mentions {s['mentions']:.1%}  "
                  f"exact {s['exact']:.1%}  verbatim {s['verbatim']:.1%}")
    write_csv(f"{RESULTS}/eval_qa.csv", qa_rows)
    write_csv(f"{RESULTS}/eval_qa_summary.csv", qa_summary)
    summary["qa"] = qa_summary
    summary["finetuned_checkpoint_step"] = ft_ckpt.get("step")

    # Accuracy per fact on the unseen questions (val + test), fine-tuned model.
    unseen = [r for r in qa_rows if r["model"] == "fine-tuned" and r["set"] != "qa_train"]
    per_fact = {f: mean(r["correct"] for r in unseen if r["gold_fact"] == f) for f in FACTS}
    summary["finetuned_unseen_accuracy_per_fact"] = {f: round(v, 3) for f, v in per_fact.items()}
    print("   fine-tuned, unseen questions, per fact:")
    for f, v in sorted(per_fact.items(), key=lambda t: -t[1]):
        print(f"      {f:<20}{v:.0%}")

    plot_qa_accuracy(table, f"{RESULTS}/eval_qa_accuracy.png")
    plot_confusion(unseen, "Fine-tuned model on 60 unseen questions: asked fact vs answered fact",
                   f"{RESULTS}/eval_qa_confusion.png")

    with open(f"{RESULTS}/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote results to {RESULTS}/")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

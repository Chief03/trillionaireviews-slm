"""
Scores the saved answers with the report's three-category rubric (Section 7).

Run after evaluate.py:
    python score_rubric.py

Reads results/eval_qa.csv (verbatim answers, nothing is regenerated) and
writes results/eval_rubric.csv with one label per held-out answer. Also\nprints the answer-only Q&A loss of both checkpoints on the held-out questions.

Rules
  The answer is the first sentence of the output that states any fact.
  Fully correct  that sentence states the fact the question asks about,
                 including any required condition, and is a complete
                 sentence.
  Partially      it states the right fact but is missing a required
  correct        condition, or the output starts in the middle of a word or
                 sentence (a fragment). These are marked needs_review = yes
                 for the two-reviewer check.
  Incorrect      it states a different fact, no fact, or nothing.

Required conditions (from the report's rubric):
  balance          $100 and that it is due after the session
  free reschedule  one reschedule and the 48-hour notice
  deposit transfer the transfer and that the reschedule must be eligible
"""

import csv
import re
from collections import Counter

from evaluate import classify, split_sentences

REQUIRED = {
    "service": (r"individual portrait|portrait sessions for individuals", None),
    "duration (60 min)": (r"60 minutes|one hour|60-minute", None),
    "price ($150)": (r"\$150\b", None),
    "deposit ($50)": (r"\$50\b", None),
    "balance ($100)": (r"\$100\b", r"after|ends|finished|over|complet"),
    "advance (3 days)": (r"three (or more )?days", None),
    "booking method": (r"online (booking )?form", None),
    "location": (r"Trillionaireviews Studio", None),
    "notice (48 h)": (r"48 hours|48-hour|two days", None),
    "free reschedule": (r"\b(one|once)\b", r"48 hours|48-hour|two days"),
    "deposit transfer": (r"transfer|carr|moves|new (appointment|booking|session)",
                         r"eligible|48 hours|48-hour|qualif|enough notice|on time"),
    "late arrival": (r"less|shorter|reduc|scheduled|finish|end", None),
}


def label(gold_fact, generated):
    sentences = [s for s in split_sentences(generated) if s.lower() not in ("yes.", "no.")]
    answer = next((s for s in sentences if classify(s)), None)
    if answer is None or classify(answer) != gold_fact:
        return "incorrect"
    core, condition = REQUIRED[gold_fact]
    if not re.search(core, answer, re.IGNORECASE):
        return "incorrect"
    if condition and not re.search(condition, answer, re.IGNORECASE):
        return "partial"
    text = generated.strip()
    if not (text[0].isupper() or text[0] in "$0123456789"):
        return "partial"                      # fragment
    return "full"


def main():
    rows = [r for r in csv.DictReader(open("results/eval_qa.csv", encoding="utf-8"))
            if r["set"] != "qa_train"]
    out = []
    for r in rows:
        lab = label(r["gold_fact"], r["generated"])
        out.append({"model": r["model"], "set": r["set"], "question": r["question"],
                    "reference": r["reference"], "generated": r["generated"],
                    "rubric_label": lab, "needs_review": "yes" if lab == "partial" else ""})
    with open("results/eval_rubric.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{'model':<12}{'set':<9}{'n':>4}{'full':>6}{'partial':>9}{'incorrect':>11}{'accuracy':>10}")
    for model in ("pretrained", "fine-tuned"):
        for s in ("qa_val", "qa_test"):
            c = Counter(r["rubric_label"] for r in out if r["model"] == model and r["set"] == s)
            n = sum(c.values())
            print(f"{model:<12}{s:<9}{n:>4}{c['full']:>6}{c['partial']:>9}{c['incorrect']:>11}"
                  f"{c['full'] / n:>10.1%}")


def qa_losses():
    """Answer-only loss on the held-out questions, same mask as finetune.py."""
    from evaluate import load_qa
    from finetune import encode_pair, evaluate as answer_loss
    from model import load_checkpoint
    from tokenizer import CharTokenizer

    tok = CharTokenizer.load("tokenizer.json")
    print("\nHeld-out Q&A loss (answer tokens and final <EOS> only)")
    for name, path in (("pretrained", "checkpoints/pretrained.pt"),
                       ("fine-tuned", "checkpoints/finetuned.pt")):
        model, _ = load_checkpoint(path)
        for s in ("qa_val", "qa_test"):
            ex = [encode_pair(tok, q, a, model.cfg.context_length) for q, a in load_qa(f"data/{s}.jsonl")]
            print(f"   {name:<12}{s:<9}{answer_loss(model, ex, tok.pad_id, 'cpu'):.4f}")


if __name__ == "__main__":
    main()
    qa_losses()

"""
Split the corpus into train / validation / test files.

  python prepare_data.py

The corpus is a list of documents separated by <EOS>. We split by WHOLE
documents, not by characters, so a validation document is never partly seen
during training.

  data/train.txt  the model learns from this
  data/val.txt    used during training to watch for overfitting
  data/test.txt   never touched by training, kept for the final evaluation

Run this again every time data/corpus.txt changes.
"""

import argparse
import random

from tokenizer import CharTokenizer

SEP = "<EOS>"


def read_documents(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    docs = [d.strip() for d in text.split(SEP)]
    return [d for d in docs if d]


def write_documents(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n" + SEP + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    docs = read_documents(args.corpus)
    random.Random(args.seed).shuffle(docs)   # same seed -> same split every time

    n = len(docs)
    n_val = max(1, round(n * args.val_frac))
    n_test = max(1, round(n * args.test_frac))
    assert n - n_val - n_test >= 1, "corpus has too few documents to split"

    val = docs[:n_val]
    test = docs[n_val:n_val + n_test]
    train = docs[n_val + n_test:]

    write_documents("data/train.txt", train)
    write_documents("data/val.txt", val)
    write_documents("data/test.txt", test)

    tok = CharTokenizer.load("tokenizer.json")
    print(f"{'split':<8}{'documents':>10}{'tokens':>10}")
    total = 0
    for name, part in (("train", train), ("val", val), ("test", test)):
        n_tok = sum(len(tok.encode(d, add_eos=True)) for d in part)
        total += n_tok
        print(f"{name:<8}{len(part):>10}{n_tok:>10,}")
    print(f"{'total':<8}{n:>10}{total:>10,}")

    if n < 50:
        print(f"\nNOTE: only {n} documents. Validation loss on so little data is "
              "very noisy.\nAsk Member 1 for a larger corpus, then run this again.")


if __name__ == "__main__":
    main()

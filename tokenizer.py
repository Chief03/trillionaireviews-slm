"""
Character-level tokenizer for the Trillionaireviews Photography SLM.

Why character-level?
  * The corpus is small and highly repetitive, so a learned subword vocabulary
    (BPE) buys little and adds complexity.
  * No out-of-vocabulary words: any new prompt typed during the demo can be
    encoded, because every printable ASCII character is always in the vocab.
  * The vocabulary is tiny (~100 tokens), which keeps the embedding matrix
    and LM head small.

Vocabulary stability:
  The vocab = special tokens + ALL printable ASCII characters (+ any extra
  characters found in the corpus). Because the base set is fixed, the token
  IDs do NOT change when Member 1 grows the corpus, so checkpoints trained on
  an older corpus stay compatible with the tokenizer file.

Special tokens (each is ONE token, never split into characters):
  <PAD>  padding for batching Q&A examples of different lengths
  <UNK>  any character not in the vocabulary
  <BOS>  beginning of a sequence
  <EOS>  end of a document / end of an answer (used as a stop token)
  <Q>    marks the start of a question  (supervised fine-tuning)
  <A>    marks the start of an answer   (supervised fine-tuning)

Usage:
  python tokenizer.py --corpus data/corpus.txt --out tokenizer.json
"""

import argparse
import json
import re
import string

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<Q>", "<A>"]

# Map typographic characters produced by AI text generators to plain ASCII,
# so "studio’s" and "studio's" become the same token sequence.
NORMALIZE_MAP = {
    "’": "'", "‘": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "—": "-", "–": "-",   # em / en dash
    "→": "->",                 # arrow
    "…": "...",                # ellipsis
    " ": " ",                  # non-breaking space
    "\r": "",
    "\t": " ",
}

# Printable ASCII except vertical whitespace other than "\n".
BASE_CHARS = [c for c in string.printable if c not in "\r\x0b\x0c\t"]


def normalize(text: str) -> str:
    for src, dst in NORMALIZE_MAP.items():
        text = text.replace(src, dst)
    return text


class CharTokenizer:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)                         # id -> token string
        self.stoi = {t: i for i, t in enumerate(itos)}  # token string -> id
        # Regex that matches any special token, so encode() keeps them atomic.
        self._special_re = re.compile(
            "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
        )

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, corpus_text: str) -> "CharTokenizer":
        text = cls._strip_special(normalize(corpus_text))
        extra = sorted(set(text) - set(BASE_CHARS))
        itos = SPECIAL_TOKENS + BASE_CHARS + extra
        return cls(itos)

    @staticmethod
    def _strip_special(text: str) -> str:
        for t in SPECIAL_TOKENS:
            text = text.replace(t, "")
        return text

    # ---------------------------------------------------------- encode/decode
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        text = normalize(text)
        ids = [self.bos_id] if add_bos else []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in SPECIAL_TOKENS:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self.stoi.get(ch, self.unk_id) for ch in piece)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special: bool = False) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    # ------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    pad_id = property(lambda self: self.stoi["<PAD>"])
    unk_id = property(lambda self: self.stoi["<UNK>"])
    bos_id = property(lambda self: self.stoi["<BOS>"])
    eos_id = property(lambda self: self.stoi["<EOS>"])
    q_id = property(lambda self: self.stoi["<Q>"])
    a_id = property(lambda self: self.stoi["<A>"])

    # ------------------------------------------------------------ save/load
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "special_tokens": SPECIAL_TOKENS,
                       "itos": self.itos}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["itos"])


def main():
    ap = argparse.ArgumentParser(description="Build the character tokenizer.")
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer.build(text)
    tok.save(args.out)

    n_tokens = len(tok.encode(text))
    n_unk = tok.encode(text).count(tok.unk_id)
    print(f"vocab_size      : {tok.vocab_size}")
    print(f"corpus chars    : {len(text):,}")
    print(f"corpus tokens   : {n_tokens:,}")
    print(f"unknown tokens  : {n_unk}")
    print(f"saved tokenizer : {args.out}")


if __name__ == "__main__":
    main()

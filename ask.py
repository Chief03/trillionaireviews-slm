"""
Ask the trained model questions (inference).

    python ask.py "Where is the session held?"
    python ask.py "How much is the deposit?" "Can I reschedule twice?"
    python ask.py                          # interactive: type questions, empty line to quit
    python ask.py --compare "How early do I have to book?"   # pretrained vs fine-tuned
    python ask.py --complete "Each session lasts"            # pretrained text completion

Options:
    --checkpoint PATH   model to load (default checkpoints/finetuned.pt)
    --temperature T     0 = greedy, the setting used for every result in the report
    --top-k K           sample only from the K most likely tokens (with T > 0)
    --max-new N         maximum characters to generate (default 120)

Questions are formatted exactly as in fine-tuning and evaluation:
<Q>question<A>, and generation stops at <EOS>. No training happens here;
the script only loads a saved checkpoint and tokenizer.json.
"""

import argparse

import torch

from model import load_checkpoint
from tokenizer import CharTokenizer


@torch.no_grad()
def generate(model, tok, prompt, max_new, temperature, top_k):
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, stop_id=tok.eos_id)
    return tok.decode(out[0, ids.size(1):].tolist(), skip_special=True).strip()


def main():
    ap = argparse.ArgumentParser(description="Ask the Trillionaireviews model a question.")
    ap.add_argument("questions", nargs="*", help="one or more questions (none = interactive)")
    ap.add_argument("--checkpoint", default="checkpoints/finetuned.pt")
    ap.add_argument("--compare", action="store_true",
                    help="also answer with checkpoints/pretrained.pt")
    ap.add_argument("--complete", action="store_true",
                    help="treat the input as text to continue (no <Q>/<A> format)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = CharTokenizer.load("tokenizer.json")

    paths = [args.checkpoint]
    if args.complete and args.checkpoint == "checkpoints/finetuned.pt":
        paths = ["checkpoints/pretrained.pt"]
    if args.compare:
        paths = ["checkpoints/pretrained.pt", args.checkpoint]

    models = []
    for p in paths:
        m, ckpt = load_checkpoint(p, device="cpu")
        m.eval()
        models.append((f"{p} (step {ckpt.get('step', '?')})", m))

    def answer(text):
        prompt = text if args.complete else f"<Q>{text}<A>"
        label = "Text" if args.complete else "Q"
        print(f"{label}: {text}")
        for name, m in models:
            reply = generate(m, tok, prompt, args.max_new, args.temperature, args.top_k)
            tag = f"A [{name}]" if len(models) > 1 else "A"
            print(f"{tag}: {reply}")
        print()

    if args.questions:
        for q in args.questions:
            answer(q)
        return

    print(f"Loaded {', '.join(n for n, _ in models)}. Empty line to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        answer(q)


if __name__ == "__main__":
    main()

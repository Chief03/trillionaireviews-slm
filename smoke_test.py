"""
Sanity checks for the tokenizer and Transformer (Member 2's part).

  python smoke_test.py

Checks:
  1. Tokenizer encode -> decode round-trips the corpus; special tokens stay atomic.
  2. Shapes: embedding (V, d_model), LM head (V, d_model), logits (B, T, V).
  3. Initial loss is close to ln(V) (uniform prediction at random init).
  4. Causal mask: changing a future token does not change earlier logits.
  5. Gradients reach every trainable parameter.
  6. A short overfit run on the sample corpus drives the loss down.
  7. generate() runs, and save/load of a checkpoint reproduces identical logits.
"""

import math
import os
import tempfile

import torch

from model import GPTConfig, TinyGPT, save_checkpoint, load_checkpoint, pick_device
from tokenizer import CharTokenizer, normalize

torch.manual_seed(0)

# ---------------------------------------------------------------- tokenizer
text = open("data/corpus.txt", encoding="utf-8").read()
tok = CharTokenizer.build(text)
tok.save("tokenizer.json")
tok = CharTokenizer.load("tokenizer.json")

ids = tok.encode(text)
assert tok.decode(ids) == normalize(text), "round-trip failed"
assert tok.unk_id not in ids, "corpus produced <UNK> tokens"
qa = tok.encode("<Q>How much is the deposit?<A>$50<EOS>")
assert qa[0] == tok.q_id and tok.a_id in qa and qa[-1] == tok.eos_id
print(f"[1] tokenizer ok   vocab_size={tok.vocab_size}  corpus_tokens={len(ids):,}")

# -------------------------------------------------------------------- model
cfg = GPTConfig.from_json("config.json", vocab_size=tok.vocab_size)
model = TinyGPT(cfg)
V, D = cfg.vocab_size, cfg.d_model
assert model.tok_emb.weight.shape == (V, D)
assert model.lm_head.weight.shape == (V, D)
print(f"[2] shapes ok      E_tok={tuple(model.tok_emb.weight.shape)}  "
      f"LM head={tuple(model.lm_head.weight.shape)}  params={model.num_params():,}")

data = torch.tensor(ids, dtype=torch.long)
T = min(cfg.context_length, len(data) - 1)
x = torch.stack([data[i:i + T] for i in (0, 100, 200, 300)])
y = torch.stack([data[i + 1:i + T + 1] for i in (0, 100, 200, 300)])

model.eval()
logits, loss = model(x, y)
assert logits.shape == (4, T, V)
print(f"[3] init loss      {loss.item():.3f}  (ln V = {math.log(V):.3f})")
assert abs(loss.item() - math.log(V)) < 0.5

# Causal mask: perturb the last token; logits at all earlier positions must not move.
x2 = x.clone()
x2[:, -1] = (x2[:, -1] + 1) % V
l1, _ = model(x)
l2, _ = model(x2)
assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6), "future token leaked!"
assert not torch.allclose(l1[:, -1], l2[:, -1]), "last position should change"
print("[4] causal mask ok (future tokens cannot influence the past)")

model.train()
_, loss = model(x, y)
loss.backward()
missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
assert not missing, f"no gradient for {missing}"
print("[5] gradients reach all parameters")

# ------------------------------------------------------------ overfit check
device = pick_device()
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
start = None
for step in range(150):
    i = torch.randint(0, len(data) - T - 1, (16,))
    xb = torch.stack([data[j:j + T] for j in i]).to(device)
    yb = torch.stack([data[j + 1:j + T + 1] for j in i]).to(device)
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    start = start or loss.item()
print(f"[6] overfit ok     loss {start:.3f} -> {loss.item():.3f}  (device={device})")
assert loss.item() < start * 0.6

# ------------------------------------------------------ generate + save/load
prompt = torch.tensor([tok.encode("Each session lasts")], device=device)
out = model.generate(prompt, max_new_tokens=60, temperature=0.8, top_k=20)
print(f"[7] sample: {tok.decode(out[0].tolist())!r}")

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "ckpt.pt")
    model.eval()
    save_checkpoint(path, model, extra={"note": "smoke test"})
    m2, _ = load_checkpoint(path, device=device)
    a, _ = model(xb)
    b, _ = m2(xb)
    assert torch.allclose(a, b, atol=1e-5)
print("    save/load ok\n\nAll checks passed.")

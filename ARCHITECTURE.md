# Tokenizer and Transformer Architecture (Member 2)

## Files
| File | Purpose |
|---|---|
| `tokenizer.py` | Character-level tokenizer: vocabulary, token IDs, `encode` / `decode`, save/load |
| `tokenizer.json` | The built vocabulary (generated, ship this with both checkpoints) |
| `model.py` | Decoder-only Transformer (`TinyGPT`), generation, checkpoint save/load |
| `config.json` | Model hyperparameters (`vocab_size` is filled in from the tokenizer) |
| `smoke_test.py` | Sanity checks for everything above |

```bash
python tokenizer.py --corpus data/corpus.txt --out tokenizer.json   # rebuild vocab
python smoke_test.py                                                # verify
```

## 1. Tokenization

**Strategy: character-level.** Every character is one token. We chose it over word-level or BPE because:
- The corpus is small and repetitive, so subword merges add complexity without much benefit.
- The model never sees an unknown word. Any prompt typed during the demo or by the graders can be encoded.
- The vocabulary stays tiny, so the embedding matrix and the LM head stay small.

**Vocabulary (102 tokens)** = 6 special tokens + 96 printable ASCII characters (+ any extra characters found in the corpus).

| ID | Token | Use |
|---|---|---|
| 0 | `<PAD>` | padding Q&A batches |
| 1 | `<UNK>` | unseen character |
| 2 | `<BOS>` | start of sequence |
| 3 | `<EOS>` | end of document / end of answer (stop token for generation) |
| 4 | `<Q>` | start of a question (fine-tuning) |
| 5 | `<A>` | start of an answer (fine-tuning) |

Special tokens are matched before character splitting, so `<Q>` is one ID, not three characters. Typographic characters from AI-generated text (curly quotes, em dashes) are normalized to ASCII first.

The base character set is fixed, so **token IDs do not change when the corpus grows**. The pretrained and fine-tuned checkpoints therefore share one `tokenizer.json`.

### How vocabulary size sets the embedding and LM-head dimensions
With V = 102 and d_model = 128:
- **Token embedding matrix** `E_tok ∈ R^(V × d_model)` = 102 × 128. Token ID *i* selects row *i*.
- **LM head** `W_head ∈ R^(d_model → V)`, stored as a 102 × 128 weight. Each position's 128-dim hidden vector becomes **102 logits, one per vocabulary token**. Softmax over those gives the next-token distribution.

Doubling V would double both matrices. With BPE (V ≈ 5,000), these two layers alone would hold about 1.3M parameters, more than our whole model.

## 2. Model

```
token ids (B,T)
  → token embedding E_tok[ids]          (B,T,128)
  + positional embedding E_pos[0..T-1]  (B,T,128)   learned, one row per position
  → 4 × DecoderBlock
        x = x + MultiHeadCausalAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
  → final LayerNorm
  → LM head                             (B,T,102)   logits
```

### Causal multi-head self-attention
For each of the 4 heads (d_k = 128 / 4 = 32):

    Q = X·W_Q,   K = X·W_K,   V = X·W_V
    scores = Q·Kᵀ / √d_k                         (T × T)
    scores[i, j] = −∞  for j > i                 ← causal mask
    head = softmax(scores) · V

The heads are concatenated and projected by **W_O**. W_Q, W_K, W_V and W_O are explicit `nn.Linear(128,128)` layers. The mask is a lower-triangular matrix, so position *i* only attends to positions ≤ *i*. Without it, the model could read the very token it is supposed to predict. `smoke_test.py` checks this: changing the last token leaves every earlier logit unchanged.

### Other design choices
- **Feed-forward:** 128 → 512 → 128 with GELU, applied independently at each position.
- **Residual connections** around both sub-layers keep gradients flowing through depth.
- **Pre-LayerNorm** (normalize before each sub-layer) trains more stably than post-norm for small models without warmup.
- **Dropout 0.1** on attention weights and residual outputs. This matters because the corpus is small and easy to memorize.
- **Initialization:** N(0, 0.02). Residual output projections are scaled by 1/√(2·n_layers), as in GPT-2.

## 3. Hyperparameters and size

| Setting | Value | Reason |
|---|---|---|
| `n_layers` | 4 | enough depth for a single narrow domain |
| `n_heads` | 4 | d_k = 32 per head |
| `d_model` | 128 | trains in minutes on a laptop CPU or Apple MPS |
| `d_ff` | 512 | standard 4 × d_model |
| `context_length` | 256 characters | fits a full Q&A pair or a paragraph of a document |
| **Total parameters** | **850,176** | |

Parameter breakdown:

| Component | Parameters |
|---|---|
| Token embedding (102 × 128) | 13,056 |
| Positional embedding (256 × 128) | 32,768 |
| Per block: attention 4·128² = 65,536; FFN 131,712; 2 LayerNorms 512 | 197,760 |
| 4 blocks | 791,040 |
| Final LayerNorm | 256 |
| LM head (128 × 102) | 13,056 |

## 4. Interface for training (Member 3) and evaluation (Member 4)

```python
from tokenizer import CharTokenizer
from model import GPTConfig, TinyGPT, save_checkpoint, load_checkpoint, pick_device

tok   = CharTokenizer.load("tokenizer.json")
cfg   = GPTConfig.from_json("config.json", vocab_size=tok.vocab_size)
model = TinyGPT(cfg).to(pick_device())

# Pretraining: targets = inputs shifted by one token
logits, loss = model(x, y)

# Fine-tuning: format "<Q>question<A>answer<EOS>"; set targets to -100 on the
# question tokens so the loss is computed on answer tokens only.

save_checkpoint("checkpoints/pretrained.pt", model, extra={"step": step})
model, ckpt = load_checkpoint("checkpoints/pretrained.pt", device="cpu")

ids = torch.tensor([tok.encode("<Q>How much is the deposit?<A>")])
out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=20, stop_id=tok.eos_id)
print(tok.decode(out[0].tolist(), skip_special=True))
```

Checkpoints store the config with the weights, so `load_checkpoint` rebuilds the model without needing `config.json`.

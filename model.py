"""
Decoder-only Transformer language model, written from scratch in PyTorch.

Only basic building blocks from torch are used (nn.Linear, nn.Embedding,
nn.LayerNorm, softmax, autograd). Attention, masking, blocks and the LM head
are implemented explicitly so each piece maps to a concept from class.

Data flow for an input of B sequences, each T tokens long:

  token ids        (B, T)
  token embedding  (B, T, d_model)   lookup in E_tok  of shape (vocab_size, d_model)
  + pos embedding  (B, T, d_model)   lookup in E_pos  of shape (context_length, d_model)
  N x DecoderBlock (B, T, d_model)
  final LayerNorm  (B, T, d_model)
  LM head          (B, T, vocab_size) = one logit per vocabulary token

For other team members:
  model = TinyGPT(GPTConfig(...))
  logits, loss = model(x, targets)          # targets may use -100 to ignore
  out_ids = model.generate(prompt_ids, ...)
  save_checkpoint(path, model, extra={...}) / load_checkpoint(path)
"""

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config
@dataclass
class GPTConfig:
    vocab_size: int = 106        # set from the tokenizer, not by hand
    context_length: int = 256    # max tokens the model can attend over
    n_layers: int = 4            # number of decoder blocks
    n_heads: int = 4             # attention heads per block
    d_model: int = 128           # embedding / residual stream width
    d_ff: int = 512              # feed-forward hidden width (4 * d_model)
    dropout: float = 0.1
    tie_weights: bool = False    # share E_tok with the LM head

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @classmethod
    def from_json(cls, path: str, vocab_size: int | None = None) -> "GPTConfig":
        with open(path) as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        if vocab_size is not None:
            cfg["vocab_size"] = vocab_size
        return cls(**cfg)


# ------------------------------------------------------- causal attention
class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask.

    For each head h (head_dim d_k = d_model / n_heads):
        Q = X W_Q,  K = X W_K,  V = X W_V
        scores = Q K^T / sqrt(d_k)                 (T x T)
        scores[i, j] = -inf  for j > i             (causal mask: no peeking ahead)
        A = softmax(scores) V
    Heads are concatenated and mixed by the output projection W_O.

    W_Q, W_K, W_V each hold all heads at once (d_model -> d_model); the
    reshape below splits them into n_heads slices of size d_k.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads

        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular matrix: mask[i, j] = 1 if token i may see token j.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape

        # Project, then split into heads: (B, T, C) -> (B, n_heads, T, d_k)
        q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask: future positions get -inf so softmax gives them weight 0.
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                   # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        out = self.resid_dropout(self.W_O(out))
        return (out, attn) if return_attn else out


# ------------------------------------------------------------ feed-forward
class FeedForward(nn.Module):
    """Position-wise MLP: d_model -> d_ff -> d_model with GELU."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------- decoder block
class DecoderBlock(nn.Module):
    """
    Pre-norm decoder block:
        x = x + Attention(LayerNorm(x))     residual connection 1
        x = x + FeedForward(LayerNorm(x))   residual connection 2
    Pre-norm (normalize before each sub-layer) trains more stably than the
    original post-norm layout, especially for small models without warmup.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# -------------------------------------------------------------- full model
class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding matrix: one d_model-dim row per vocabulary token.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # (V, d_model)
        # Learned positional embeddings: one row per position 0..context_length-1.
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # LM head: maps each d_model vector to vocab_size logits.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # weight (V, d_model)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style: scale residual output projections by 1/sqrt(2 * n_layers).
        for name, p in self.named_parameters():
            if name.endswith("W_O.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_pos: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.weight.numel() if exclude_pos else n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) token ids, T <= context_length
        targets: (B, T) next-token ids; positions set to -100 are ignored
                 (use this in fine-tuning to train only on answer tokens).
        Returns (logits of shape (B, T, vocab_size), loss or None).
        """
        B, T = idx.shape
        assert T <= self.cfg.context_length, \
            f"sequence length {T} exceeds context_length {self.cfg.context_length}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)        # (B, T, d_model)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 1.0, top_k: int | None = None,
                 stop_id: int | None = None) -> torch.Tensor:
        """
        Autoregressive sampling. idx is (B, T) prompt ids; returns prompt + new ids.
        temperature: <1 sharper / more deterministic, >1 more random; 0 = greedy.
        top_k: sample only from the k most likely tokens.
        stop_id: stop when this token (e.g. <EOS>) is produced (B == 1).
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                     # last position only

            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if stop_id is not None and idx.size(0) == 1 and next_id.item() == stop_id:
                break
        self.train(was_training)
        return idx


# -------------------------------------------------------- checkpoint helpers
def save_checkpoint(path: str, model: TinyGPT, extra: dict | None = None) -> None:
    """Saves weights + config so the model can be rebuilt without code changes."""
    torch.save({"config": asdict(model.cfg),
                "model_state": model.state_dict(),
                **(extra or {})}, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[TinyGPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

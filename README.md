# Trillionaireviews SLM

A small decoder-only Transformer language model, written from scratch in PyTorch and trained on a synthetic corpus about a fictional photography studio, **Trillionaireviews Photography**. The domain is limited to individual portrait-session booking and rescheduling (12 facts).

Course project for CS 5370, Fall 2026.
Authors: Blessing Omotade, Eworitse Mabuyaku, Onyinyechukwu Ifeanyi-Ukaegbu, Leonel Kachie Takoudjou.

The model is trained in two stages:

1. **Pretraining** on 240 synthetic documents (next-character prediction).
2. **Supervised fine-tuning** on 240 question–answer pairs.

Both checkpoints are included, so you can run and compare them without retraining.

---

## 1. Requirements

| Item | Requirement |
|---|---|
| Python | 3.10 or newer (developed on 3.14) |
| Packages | `torch>=2.2`, `numpy<2`, `matplotlib` (see `requirements.txt`) |
| Hardware | **CPU only is sufficient.** No GPU needed. |
| Memory / disk | Under 1 GB RAM for inference; each checkpoint is about 3.4 MB |
| Tested on | MacBook Pro (CPU) |

Approximate run times on CPU: inference takes about a second per answer; pretraining took about 59 minutes (3,552 s) and fine-tuning about 3.5 minutes (203 s).

### Install

```bash
git clone https://github.com/Chief03/trillionaireviews-slm.git
cd trillionaireviews-slm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run every command below from the repository root. The scripts use relative paths such as `checkpoints/finetuned.pt` and `tokenizer.json`.

---

## 2. Files needed to run the model (no retraining)

| File | Purpose |
|---|---|
| `checkpoints/finetuned.pt` | Fine-tuned model (step 450, chosen by validation accuracy) |
| `checkpoints/pretrained.pt` | Pretrained model (step 1,800, lowest validation loss) |
| `tokenizer.json` | Character-level vocabulary (102 tokens). It must be the same file used in training. |
| `config.json` | Model configuration. Each checkpoint also stores its own copy, so weights load without extra arguments. |
| `model.py`, `tokenizer.py` | Model and tokenizer code |
| `ask.py` | Inference script |

---

## 3. Quick start: ask a question

```bash
python ask.py "Where is the session held?"
```

`ask.py` loads `checkpoints/finetuned.pt`, formats the input as `<Q>question<A>`, and decodes greedily until the model emits `<EOS>`.

More ways to run it:

```bash
# Interactive mode: type questions, press Enter on an empty line to quit
python ask.py

# Side-by-side comparison: pretrained vs. fine-tuned on the same question
python ask.py --compare "How early do I have to book?"

# Text completion with the pretrained model
python ask.py --complete "Each session lasts"

# Turn on sampling (default is greedy, which is what every reported result uses)
python ask.py --temperature 0.8 --top-k 20 "How much is the deposit?"
```
Sampling is random and makes answers less reliable (see report Section 7). All reported results use greedy decoding.

### Example inputs

| Input | Typical fine-tuned output |
|---|---|
| `How early do I have to book?` | `Sessions must be booked at least three days in advance.` |
| `Where is the session held?` | `Sessions take place at Trillionaireviews Studio.` |
| `How long is a portrait session?` | `The duration of a portrait session is one hour.` |
| `--complete "Each session lasts"` | `60 minutes. The scheduled finish time remains unchanged if the client is late. This leaves less time for the session. T` |
Completions stop at a fixed length, so they can end mid-word.

Outputs can differ slightly from this table. Greedy decoding is deterministic, but small numeric differences across PyTorch versions can change a borderline character.

Try questions that are not in the training files. For example, `Can I move my appointment to next week?` was never used in training. Expect the model to answer with a sentence from the training answers, and to get the fact wrong more than half the time on unseen wordings (44% fully correct on our 36 test questions).

### Using the model from Python

```python
from model import load_checkpoint
from tokenizer import CharTokenizer
import torch

tok = CharTokenizer.load("tokenizer.json")
model, _ = load_checkpoint("checkpoints/finetuned.pt")

prompt = "<Q>Where is the session held?<A>"
ids = torch.tensor([tok.encode(prompt)])
out = model.generate(ids, max_new_tokens=200, temperature=0, stop_id=tok.eos_id)
print(tok.decode(out[0, ids.shape[1]:].tolist(), skip_special=True))
```

Note that `<Q>`, `<A>` and `<EOS>` are single special tokens, and `encode` recognizes them.

---

## 4. Sanity check

```bash
python smoke_test.py
```

Runs 7 checks (tokenizer round-trip, causal masking, shapes, loss at initialization, overfit on a tiny batch, checkpoint save/load, generation). All should pass.

---

## 5. Reproduce the full pipeline (optional)

Seeds are fixed (corpus 5370, split 42, training 42), so the data splits are reproducible. Training loss and checkpoints may differ slightly across hardware and PyTorch versions.

```bash
python tokenizer.py --corpus data/corpus.txt --out tokenizer.json   # build vocabulary
python prepare_data.py                                              # split into train/val/test.txt
python train.py                                                     # pretraining (about 1 h on CPU)
python finetune.py --train-file data/new_qa_train.jsonl             # supervised fine-tuning
python pick_checkpoint.py                                           # pick finetuned.pt by validation accuracy
python plot_loss.py                                                 # loss curves
python evaluate.py                                                  # QA + completion evaluation
python score_rubric.py                                              # rubric labels (full / partial / incorrect)
python investigate.py                                               # checkpoint, fine-tuning and temperature experiments
```

Re-running `train.py` and `finetune.py` overwrites the files in `checkpoints/`. Copy them first if you want to keep the submitted versions.

Only run this if you are retraining from scratch. The submitted checkpoints need the included tokenizer.json.
---

## 6. Repository layout

```
data/
  corpus.txt              240 generated documents (<EOS>-separated)
  train.txt val.txt test.txt   document-level split: 192 / 24 / 24
  new_qa_train.jsonl      240 fine-tuning Q&A pairs (used)
  qa_train.jsonl          96 Q&A pairs (first fine-tuning run)
  qa_val.jsonl            24 validation questions
  qa_test.jsonl           36 held-out test questions (never used for training or checkpoint selection)
checkpoints/              pretrained.pt, finetuned.pt, pretrain_step*.pt snapshots
logs/                     pretrain_log.{csv,txt}, finetune_log.{csv,txt}
results/                  loss curves, eval tables, rubric labels, generated samples
model.py                  TinyGPT: causal multi-head attention (W_Q, W_K, W_V, W_O), blocks, LM head
tokenizer.py              character-level tokenizer
config.json               model hyperparameters
```

---

## 7. Model summary

| Setting | Value |
|---|---|
| Architecture | Decoder-only Transformer, pre-layer-norm, learned positional embeddings |
| Decoder blocks / heads | 4 / 4 (32 dimensions per head) |
| Embedding dimension | 128 |
| Feed-forward | 128 → 512 → 128, GELU |
| Context length | 256 tokens |
| Dropout | 0.1 |
| Vocabulary | 102 characters and special tokens |
| Parameters | 850,176 |
| Tokenizer | Character-level (96 characters + `<PAD> <UNK> <BOS> <EOS> <Q> <A>`) |

---

## 8. Known limitations

- The model knows 12 facts about one fictional service. It is not a general assistant.
- The corpus reuses one pool of sentences, so validation and test text overlaps heavily with training text. The low pretraining validation loss reflects this overlap.
- On 36 held-out questions with greedy decoding, the fine-tuned model was fully correct on 16 (44%), against 0 for the pretrained model. Most errors are fluent but answer a different question than the one asked.
- `results/eval_summary.json` and `results/eval_qa_summary.csv` use an earlier, looser keyword scorer. The report's headline numbers come from the three-label rubric in `results/eval_rubric.csv` (produced by `score_rubric.py`).

---

## 9. AI use

ChatGPT assisted with dataset wording, the dataset-generation script, automated data checks and the report draft. Claude (Anthropic) helped review the data-split, training, fine-tuning, checkpoint-selection and plotting scripts, and helped write the evaluation and investigation code and the 36 test questions. The team reviewed and verified the code and results.

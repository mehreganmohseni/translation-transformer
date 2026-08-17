# Small Translation Transformer (WMT17 German → English)

An encoder-decoder Transformer, following ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762),
with the attention mechanism implemented from scratch and unit-tested, and a complete CLI-driven
pipeline for training it on a subset of WMT17 German→English.

This covers both parts of the task:

1. **Attention / multi-head attention from scratch** (`modelling/attention.py`), built only from
   `nn.Linear`, `nn.Softmax`, and tensor ops — no `nn.MultiheadAttention` or
   `F.scaled_dot_product_attention`. Verified against the provided pytest suite
   (`test_attention.py`, `test_mha.py`).
2. **A full translation-transformer training project** (`modelling/transformer.py`, `data/`,
   `train.py`, `translate.py`) built on top of those two classes, trained on real WMT17 de-en data,
   with results from three different model-size/data-subset configurations reported below.

## Project layout

```
modelling/
  attention.py     Attention, MultiHeadAttention (from scratch, pytest-verified)
  transformer.py   Seq2SeqTransformer, encoder/decoder stacks, greedy_decode
data/
  prepare.py       Streams a WMT17 de-en subset, trains a BPE tokenizer, caches tokenized tensors
  dataset.py       TranslationDataset + batch collation (padding, BOS/EOS shifting)
train.py            Training CLI
translate.py        Inference CLI (translate one sentence with a trained checkpoint)
test_attention.py, test_mha.py   Subtask 1 pytest suite
results/            training_log.json for each reported run (tiny / small / large)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.0+. Runs on CPU, Apple Silicon (MPS), or CUDA — `--device auto`
(the default) picks the best available automatically.

## Part 1 — Verify the attention implementation

```bash
python3 -m pytest test_attention.py test_mha.py -v
```

All 6 cases (self-attention, causal-masked self-attention, cross-attention — for both `Attention`
and `MultiHeadAttention`) pass, checked against the fixed expected-output tensors provided.

## Part 2 — Train a translation model

### 1. Prepare data

Streams WMT17 de-en from Hugging Face (only the requested number of examples are pulled, not the
full ~5.9M-pair corpus), filters degenerate/overlong sentence pairs, trains a shared byte-level BPE
tokenizer, and caches everything.

```bash
python3 data/prepare.py \
  --num-train-examples 100000 \
  --num-val-examples 3000 \
  --num-test-examples 3000 \
  --vocab-size 8000 \
  --output-dir data/processed
```

Validation and test use WMT17's real `validation`/`test` splits — held out from training, so
reported numbers are genuine, not self-selected.

### 2. Train

```bash
python3 train.py \
  --data-dir data/processed \
  --output-dir runs/small \
  --d-model 256 --num-heads 4 --num-layers 3 --d-ff 512 \
  --batch-size 64 --epochs 10 --warmup-steps 400
```

Key flags:
- `--device {auto,cpu,mps,cuda}` — defaults to `auto`.
- `--bleu-samples N` — number of test sentences to greedy-decode and score with `sacrebleu` after
  training (0 disables).
- `--eval-max-batches N` — caps per-epoch validation to N batches, to keep epoch time predictable.

Outputs in `--output-dir`: `best.pt` / `last.pt` checkpoints, and `training_log.json` (per-epoch
loss/perplexity, timing, final BLEU, sample translations).

### 3. Translate a sentence

```bash
python3 translate.py --checkpoint runs/small/best.pt --text "Guten Morgen, wie geht es Ihnen?"
```

## Hardware requirements & reproducing the results

- **Tiny** needs no GPU — it trains in under a minute even on a laptop CPU. This is the easiest
  config to just run locally to confirm everything works end-to-end.
- **Small** and **Large** are much faster with a GPU (minutes, vs. potentially hours on CPU alone).
  If you don't have a local GPU, Google Colab's free tier gives you a T4, which is enough — the
  results reported below were run on **Google Colab Pro with an NVIDIA A100** specifically, but
  Small/Large will still complete on a free-tier T4, just slower.

### Running locally (any config)

```bash
pip install -r requirements.txt
python3 data/prepare.py --num-train-examples 20000 --vocab-size 4000 --output-dir data/processed_tiny
python3 train.py --data-dir data/processed_tiny --output-dir runs/tiny \
  --d-model 128 --num-heads 4 --num-layers 2 --d-ff 256 --max-len 64 \
  --epochs 8 --warmup-steps 200
```
`--device auto` (the default) picks CUDA/MPS/CPU automatically, so the same commands work
unchanged on a GPU machine — just swap in the Small/Large flags from the Results table below.

### Running on Google Colab (for Small/Large without a local GPU)

1. New notebook at `colab.research.google.com` → `Runtime → Change runtime type → GPU`.
2. Clone the repo and install the pinned dependency versions (matches what's actually verified
   working — a newer/older `datasets`/`huggingface_hub` pairing can break dataset loading):
   ```python
   !git clone https://github.com/mehreganmohseni/translation-transformer.git
   %cd translation-transformer
   !pip install -q "datasets==4.5.0" "huggingface_hub==1.8.0" "tokenizers==0.22.2" sacrebleu
   ```
3. `Runtime → Restart session`, then `%cd /content/translation-transformer` again (imports/cwd
   don't survive a restart, but the pinned installs do).
4. Run data prep + training with `--device cuda`, e.g. for Small:
   ```python
   !python3 data/prepare.py --num-train-examples 100000 --output-dir data/processed
   !python3 train.py --data-dir data/processed --output-dir runs/small --device cuda
   ```
5. (Optional) mount Google Drive first and point `--output-dir` at it if you want results to
   survive a runtime disconnect — not needed just to verify a single run finishes successfully.

## Design notes

- **From scratch**: both encoder self-attention and decoder self/cross-attention route through the
  same `MultiHeadAttention` class from Part 1 — the training pipeline isn't a separate reimplementation.
- **Pre-LN** (`x + Sublayer(LayerNorm(x))`) is used instead of the paper's post-LN — trains more
  stably without the paper's more delicate warmup tuning, which matters for a short run on modest
  hardware. This is the one deliberate deviation from the paper; everything else (Adam with
  `betas=(0.9,0.98), eps=1e-9`, the Noam learning-rate schedule, label smoothing = 0.1, sinusoidal
  positional encoding, weight tying) matches it directly.
- **Shared tokenizer**: one byte-level BPE vocabulary across German and English, with the
  encoder/decoder embedding and output projection weight-tied.
- **MPS note**: `nn.CrossEntropyLoss` (with or without `label_smoothing`) crashes with a SIGBUS on
  the PyTorch MPS backend in `torch==2.0.1`. `train.py` uses a manual `LabelSmoothingLoss`
  (log_softmax + gather) instead — numerically equivalent, and safe on CPU/MPS/CUDA.
- **Scale**: model size and data subset are deliberately small (see Results) — the goal is a
  correct, debuggable, end-to-end pipeline, not competitive translation quality. The original
  paper's smallest ("base") model trained for 12 hours on 8 P100 GPUs over the full ~4.5M-pair
  WMT14 corpus; the configs below train in seconds to tens of minutes on a single GPU.

## Results

### Training environment

All three runs below were trained end-to-end (data prep → training → held-out evaluation) on
**Google Colab Pro, on an NVIDIA A100 GPU** (`--device cuda`), following this workflow:

1. Clone this repository into the Colab runtime and install the extra dependencies not already
   bundled with Colab's PyTorch image (`datasets`, `tokenizers`, `sacrebleu`).
2. Mount Google Drive, so checkpoints/logs survive a runtime disconnect (Colab's local disk is
   wiped when a session ends).
3. Run `data/prepare.py` for the chosen config directly on Colab (data streams straight from
   Hugging Face — nothing needs to be uploaded manually).
4. Run `train.py` with `--device cuda` and `--output-dir` pointed at the mounted Drive folder.
5. Copy the resulting `training_log.json` into the cloned repo and `git push` it back to GitHub
   directly from the Colab session — this is how the three files under `results/` got here.

Repeating this for Tiny/Small/Large just means re-running steps 3–5 with each config's flags (see
the table below).

Three configurations were trained this way, from smallest to largest:

| | Tiny | Small | Large |
|---|---|---|---|
| `d_model` / heads / layers / `d_ff` | 128 / 4 / 2 / 256 | 256 / 4 / 3 / 512 | 512 / 8 / 4 / 1024 |
| Vocab size | 4,000 | 8,000 | 16,000 |
| Training pairs | 20,000 | 100,000 | 300,000 |
| Epochs | 8 | 10 | 6 |
| Parameters | 1,171,968 | 5,993,472 | 29,198,336 |
| Training time | 58.3s | 10.8 min | 43.1 min |
| Best val. loss | 5.909 | 5.946 | 6.061 |
| Best val. perplexity | 368.5 | 382.3 | 428.6 |
| **Test BLEU** (sacrebleu, 200 sentences) | 0.14 | 0.20 | **1.13** |

Full per-epoch histories, args, and 5 sample translations per run are in `results/*_training_log.json`.

**Training is correct and stable**: validation loss decreases every single epoch in all three runs
(e.g. Small: 7.06 → 5.95 over 10 epochs), with no divergence or instability, on all three model
sizes.

**A note on comparing loss across configs**: raw validation loss/perplexity isn't directly
comparable *between* the three runs, because each trained its own tokenizer with a different
vocabulary size (4k/8k/16k). The label-smoothing loss averages log-probabilities uniformly across
the *entire* vocabulary as its smoothing term, so a larger vocabulary mechanically raises the loss
floor regardless of translation quality — this is why Large shows the *highest* loss despite being
the best-performing model. **BLEU is the metric that's actually comparable across configs** (it
scores decoded text, not internal loss), and it improves monotonically with scale — Large's BLEU is
~8x Tiny's — which is the expected, correct direction.

**Translation quality is intentionally poor**, consistent with the task's framing ("the focus is on
demonstrating your programming skills rather than achieving high translation quality"). Sample
outputs are often repetitive or incoherent, e.g. (Tiny run):

| Reference | Hypothesis |
|---|---|
| "28-Year-Old Chef Found Dead at San Francisco Mall" | "The Can, in the Can, in the Can, in the Can, ..." |

This is the expected failure mode for a model this small trained this briefly on this little data —
greedy decoding with no beam search, a few thousand optimizer steps, and tens of thousands of
sentence pairs are far below what's needed for fluent output. The meaningful signals are the
monotonic loss decrease (the pipeline learns correctly) and BLEU improving with scale (more
capacity/data genuinely helps), not the absolute translation quality.

## Known limitations / possible extensions

- Greedy decoding only (no beam search) — leaves some BLEU on the table versus the paper's
  beam-4 setup.
- No checkpoint averaging (the paper averages its last several checkpoints; this keeps only the
  single best-validation-loss checkpoint).
- Fixed batch size by sentence count, not token count (the paper batches by ~25k tokens with
  length bucketing).

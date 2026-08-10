# Small Translation Transformer (WMT17 German → English)

An encoder-decoder Transformer, following ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762),
implemented from scratch (attention, multi-head attention, encoder/decoder stacks) and trained on a subset
of WMT17 German→English.

The attention mechanism (`modelling/attention.py`) is unit-tested against fixed expected outputs — see
`test_attention.py` / `test_mha.py`. The rest of the model (`modelling/transformer.py`) is built on top of
those two classes.

## Project layout

```
modelling/
  attention.py     Attention, MultiHeadAttention (from scratch)
  transformer.py    Seq2SeqTransformer, encoder/decoder stacks, greedy_decode
data/
  prepare.py         Streams a WMT17 de-en subset, trains a BPE tokenizer, caches tokenized tensors
  dataset.py         TranslationDataset + batch collation (padding, BOS/EOS shifting)
train.py              Training CLI
translate.py          Inference CLI (translate one sentence with a trained checkpoint)
test_attention.py, test_mha.py   Subtask 1 pytest suite
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.0+. Tested on macOS/Apple Silicon (MPS) and CPU.

## 1. Verify the attention implementation

```bash
python3 -m pytest test_attention.py test_mha.py -v
```

## 2. Prepare data

Streams WMT17 de-en from Hugging Face (no full-dataset download — only the requested number of
examples are pulled), filters degenerate/overlong sentence pairs, trains a shared byte-level BPE
tokenizer, and caches everything to `data/processed/`.

```bash
python3 data/prepare.py \
  --num-train-examples 100000 \
  --num-val-examples 3000 \
  --num-test-examples 3000 \
  --vocab-size 8000 \
  --output-dir data/processed
```

Validation uses WMT17's real `validation` split, test uses the real `test` split — both held out from
training, so reported numbers are genuine, not self-selected.

## 3. Train

```bash
python3 train.py \
  --data-dir data/processed \
  --output-dir runs/exp1 \
  --d-model 256 --num-heads 4 --num-layers 3 --d-ff 512 \
  --batch-size 64 --epochs 10 --warmup-steps 400
```

Key flags:
- `--device {auto,cpu,mps,cuda}` — defaults to `auto` (prefers CUDA, then Apple MPS, then CPU).
- `--bleu-samples N` — number of test sentences to greedy-decode and score with `sacrebleu` after
  training (0 disables). Kept small by default since decoding is the slow part.
- `--eval-max-batches N` — caps per-epoch validation to N batches, to keep epoch time predictable.

Outputs in `--output-dir`:
- `best.pt` / `last.pt` — checkpoints (model weights + the args/config needed to rebuild the model).
- `training_log.json` — per-epoch train/val loss & perplexity, timing, final BLEU, and sample
  translations, used to fill in the Results section below.

## 4. Translate a sentence

```bash
python3 translate.py --checkpoint runs/exp1/best.pt --text "Guten Morgen, wie geht es Ihnen?"
```

## Design notes

- **From scratch**: `Attention` and `MultiHeadAttention` are built only from `nn.Linear`, `nn.Softmax`,
  and tensor ops — no `nn.MultiheadAttention` / `F.scaled_dot_product_attention`. `Seq2SeqTransformer`
  is built on top of those two classes (both encoder self-attention and decoder self/cross-attention
  route through `MultiHeadAttention`).
- **Pre-LN** (`x + Sublayer(LayerNorm(x))`) is used instead of the paper's post-LN — it trains more
  stably without needing the paper's more delicate warmup tuning, which matters for a short run on
  modest hardware. This is a deliberate, minor deviation from the paper.
- **Shared tokenizer**: one byte-level BPE vocabulary across German and English, with the encoder/decoder
  embedding and the output projection weight-tied — keeps parameter count and vocabulary handling simple.
- **MPS note**: `nn.CrossEntropyLoss` (with or without `label_smoothing`) crashes with a SIGBUS on the
  PyTorch MPS backend in `torch==2.0.1`. `train.py` uses a manual `LabelSmoothingLoss` (log_softmax +
  gather) instead, which is numerically equivalent and MPS-safe.
- **Scale**: model size and data subset are deliberately small (see Results below) — the goal is a
  correct, debuggable, end-to-end pipeline, not competitive translation quality. For reference, the
  original paper's smallest ("base") model trained for 12 hours on 8 P100 GPUs over the full ~4.5M-pair
  WMT14 corpus.

## Results

_To be filled in after the real training run (see `runs/exp1/training_log.json`):_

- Model size / data subset used:
- Number of parameters:
- Hardware / device:
- Total training time:
- Final train loss / validation loss / validation perplexity:
- Test BLEU (sacrebleu) on N held-out sentences:
- Sample translations (source / reference / model output):

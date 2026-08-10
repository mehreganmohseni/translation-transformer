import argparse
import json
import os
import time

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from data.dataset import TranslationDataset, make_collate_fn
from modelling.transformer import Seq2SeqTransformer, greedy_decode


class LabelSmoothingLoss(nn.Module):
    """Label-smoothed NLL loss built from log_softmax/gather.

    nn.CrossEntropyLoss (with or without label_smoothing) crashes with a
    SIGBUS on the MPS backend in torch==2.0.1, so this avoids that kernel
    entirely while giving the same result on CPU/CUDA.
    """

    def __init__(self, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.smoothing = smoothing

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        mask = (labels != self.pad_id).float()
        return (loss * mask).sum() / mask.sum().clamp(min=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small translation transformer on WMT17 de-en.")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="runs/exp1")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=128)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-max-batches", type=int, default=50, help="Cap per-epoch validation to this many batches.")
    parser.add_argument("--bleu-samples", type=int, default=200, help="Number of test sentences to decode for final BLEU. 0 disables.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-examples-log", type=int, default=5)
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def noam_lr(step, d_model, warmup_steps):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_validation(model, loader, criterion, device, max_batches):
    model.eval()
    total_loss, total_tokens, n_batches = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["src_ids"], batch["decoder_input"], batch["src_mask"], batch["tgt_mask"])
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            n_tok = (batch["labels"] != model.pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
            n_batches += 1
            if n_batches >= max_batches:
                break
    return total_loss / max(total_tokens, 1)


def evaluate_bleu(model, tokenizer, test_src, test_tgt_text, device, bos_id, eos_id, pad_id, num_samples, max_len, num_examples_log):
    if num_samples <= 0 or len(test_src) == 0:
        return None, []

    num_samples = min(num_samples, len(test_src))
    model.eval()
    hypotheses, references, examples = [], [], []

    batch_size = 32
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            chunk_src = [ids[:max_len] for ids in test_src[start:start + batch_size]]
            src_len = max(len(s) for s in chunk_src)
            src_ids = torch.full((len(chunk_src), src_len), pad_id, dtype=torch.long)
            src_mask = torch.zeros((len(chunk_src), src_len), dtype=torch.long)
            for i, ids in enumerate(chunk_src):
                src_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                src_mask[i, :len(ids)] = 1
            src_ids, src_mask = src_ids.to(device), src_mask.to(device)

            ys = greedy_decode(model, src_ids, src_mask, bos_id, eos_id, max_len=max_len)
            for row in ys.tolist():
                if eos_id in row:
                    row = row[:row.index(eos_id)]
                row = [t for t in row if t != bos_id]
                hypotheses.append(tokenizer.decode(row))

    references = test_tgt_text[:num_samples]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    for i in range(min(num_examples_log, num_samples)):
        examples.append({"reference": references[i], "hypothesis": hypotheses[i]})

    return bleu.score, examples


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    data = torch.load(os.path.join(args.data_dir, "data.pt"))
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    pad_id, bos_id, eos_id = data["pad_id"], data["bos_id"], data["eos_id"]

    collate_fn = make_collate_fn(pad_id, bos_id, eos_id, max_len=args.max_len)
    train_ds = TranslationDataset(data["train"]["src"], data["train"]["tgt"])
    val_ds = TranslationDataset(data["val"]["src"], data["val"]["tgt"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        vocab_size=data["vocab_size"],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=pad_id,
    ).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    criterion = LabelSmoothingLoss(pad_id, smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: noam_lr(step + 1, args.d_model, args.warmup_steps)
    )

    history = []
    best_val_loss = float("inf")
    global_step = 0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, running_tokens = 0.0, 0

        for batch in train_loader:
            batch = move_batch(batch, device)
            logits = model(batch["src_ids"], batch["decoder_input"], batch["src_mask"], batch["tgt_mask"])
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            n_tok = (batch["labels"] != pad_id).sum().item()
            running_loss += loss.item() * n_tok
            running_tokens += n_tok
            global_step += 1

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step:6d} | loss {loss.item():.4f} | lr {lr:.6f}")

        train_loss = running_loss / max(running_tokens, 1)
        val_loss = run_validation(model, val_loader, criterion, device, args.eval_max_batches)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_ppl {torch.exp(torch.tensor(val_loss)):.2f} | "
            f"time {epoch_time:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": float(torch.exp(torch.tensor(val_loss))),
            "epoch_time_seconds": epoch_time,
        })

        checkpoint = {"model_state": model.state_dict(), "args": vars(args), "data_config": {
            "pad_id": pad_id, "bos_id": bos_id, "eos_id": eos_id, "vocab_size": data["vocab_size"],
        }}
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))

    total_time = time.time() - train_start
    print(f"Training finished in {total_time / 60:.1f} min")

    test_tgt_text = [tokenizer.decode(ids) for ids in data["test"]["tgt"]]
    bleu_score, examples = evaluate_bleu(
        model, tokenizer, data["test"]["src"], test_tgt_text, device,
        bos_id, eos_id, pad_id, args.bleu_samples, args.max_len, args.num_examples_log,
    )
    if bleu_score is not None:
        print(f"Test BLEU (on {min(args.bleu_samples, len(data['test']['src']))} sentences): {bleu_score:.2f}")

    summary = {
        "device": str(device),
        "num_parameters": model.num_parameters(),
        "total_training_time_seconds": total_time,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": float(torch.exp(torch.tensor(best_val_loss))),
        "test_bleu": bleu_score,
        "sample_translations": examples,
        "history": history,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training log to {os.path.join(args.output_dir, 'training_log.json')}")


if __name__ == "__main__":
    main()

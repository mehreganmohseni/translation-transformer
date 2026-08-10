import argparse
import itertools
import json
import os
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a WMT17 de-en subset for training.")
    parser.add_argument("--num-train-examples", type=int, default=100_000)
    parser.add_argument("--num-val-examples", type=int, default=3000)
    parser.add_argument("--num-test-examples", type=int, default=3000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-words", type=int, default=100, help="Filter out sentences longer than this (whitespace-split words).")
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_valid_pair(de, en, min_words, max_words):
    de_len, en_len = len(de.split()), len(en.split())
    if not de.strip() or not en.strip():
        return False
    return min_words <= de_len <= max_words and min_words <= en_len <= max_words


def collect_pairs(split, limit, min_words, max_words):
    ds = load_dataset("wmt17", "de-en", split=split, streaming=True)
    pairs = []
    for example in ds:
        de = example["translation"]["de"]
        en = example["translation"]["en"]
        if is_valid_pair(de, en, min_words, max_words):
            pairs.append((de, en))
            if len(pairs) >= limit:
                break
    return pairs


def build_tokenizer(sentences, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    return tokenizer


def encode_pairs(tokenizer, pairs):
    src_ids = [tokenizer.encode(de).ids for de, _ in pairs]
    tgt_ids = [tokenizer.encode(en).ids for _, en in pairs]
    return src_ids, tgt_ids


def avg_len(seqs):
    return sum(len(s) for s in seqs) / max(len(seqs), 1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    print(f"Streaming train split (target {args.num_train_examples} pairs)...")
    train_pairs = collect_pairs("train", args.num_train_examples, args.min_words, args.max_words)
    print(f"  collected {len(train_pairs)} train pairs")

    print(f"Streaming validation split (target {args.num_val_examples} pairs)...")
    val_pairs = collect_pairs("validation", args.num_val_examples, args.min_words, args.max_words)
    print(f"  collected {len(val_pairs)} validation pairs")

    print(f"Streaming test split (target {args.num_test_examples} pairs)...")
    test_pairs = collect_pairs("test", args.num_test_examples, args.min_words, args.max_words)
    print(f"  collected {len(test_pairs)} test pairs")

    print(f"Training shared BPE tokenizer (vocab_size={args.vocab_size})...")
    train_sentences = itertools.chain(
        (de for de, _ in train_pairs), (en for _, en in train_pairs)
    )
    tokenizer = build_tokenizer(train_sentences, args.vocab_size)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    unk_id = tokenizer.token_to_id("<unk>")

    print("Encoding splits...")
    train_src, train_tgt = encode_pairs(tokenizer, train_pairs)
    val_src, val_tgt = encode_pairs(tokenizer, val_pairs)
    test_src, test_tgt = encode_pairs(tokenizer, test_pairs)

    data = {
        "train": {"src": train_src, "tgt": train_tgt},
        "val": {"src": val_src, "tgt": val_tgt},
        "test": {"src": test_src, "tgt": test_tgt},
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unk_id": unk_id,
        "vocab_size": tokenizer.get_vocab_size(),
    }
    data_path = os.path.join(args.output_dir, "data.pt")
    torch.save(data, data_path)

    stats = {
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
        "vocab_size": tokenizer.get_vocab_size(),
        "avg_src_len_tokens": avg_len(train_src),
        "avg_tgt_len_tokens": avg_len(train_tgt),
        "prep_time_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output_dir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Done.")
    print(json.dumps(stats, indent=2))
    print(f"Saved tokenized data to {data_path}")
    print(f"Saved tokenizer to {tokenizer_path}")


if __name__ == "__main__":
    main()

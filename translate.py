import argparse

import torch
from tokenizers import Tokenizer

from modelling.transformer import Seq2SeqTransformer, greedy_decode


def parse_args():
    parser = argparse.ArgumentParser(description="Translate German text with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    parser.add_argument("--text", type=str, required=True, help="German source sentence to translate.")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint["args"]
    data_config = checkpoint["data_config"]

    tokenizer = Tokenizer.from_file(args.tokenizer)

    model = Seq2SeqTransformer(
        vocab_size=data_config["vocab_size"],
        d_model=train_args["d_model"],
        num_heads=train_args["num_heads"],
        num_layers=train_args["num_layers"],
        d_ff=train_args["d_ff"],
        dropout=train_args["dropout"],
        max_len=train_args["max_len"],
        pad_id=data_config["pad_id"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    src_ids = torch.tensor([tokenizer.encode(args.text).ids], dtype=torch.long, device=device)
    src_mask = torch.ones_like(src_ids)

    max_len = min(args.max_len, train_args["max_len"])  
    ys = greedy_decode(
        model, src_ids, src_mask,
        data_config["bos_id"], data_config["eos_id"], max_len=max_len,
    )
    tokens = ys[0].tolist()
    if data_config["eos_id"] in tokens:
        tokens = tokens[:tokens.index(data_config["eos_id"])]
    tokens = [t for t in tokens if t != data_config["bos_id"]]

    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()

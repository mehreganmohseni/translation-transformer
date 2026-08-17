import math

import torch
import torch.nn as nn

from modelling.attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):

    """Pre-LN encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, mask_future=False)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, src_mask))
        normed = self.norm2(x)
        x = x + self.dropout(self.ff(normed))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, mask_future=True)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, mask_future=False)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, memory, memory, src_mask))
        normed = self.norm3(x)
        x = x + self.dropout(self.ff(normed))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff, dropout)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  

    def _embed(self, ids):
        x = self.embedding(ids) * math.sqrt(self.d_model)
        return self.dropout(self.pos_encoding(x))

    def encode(self, src_ids, src_mask):
        return self.encoder(self._embed(src_ids), src_mask)

    def decode(self, tgt_ids, memory, tgt_mask, src_mask):
        return self.decoder(self._embed(tgt_ids), memory, tgt_mask, src_mask)

    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask):
        memory = self.encode(src_ids, src_mask)
        dec_out = self.decode(tgt_ids, memory, tgt_mask, src_mask)
        return self.output_proj(dec_out)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def greedy_decode(model, src_ids, src_mask, bos_id, eos_id, max_len=64):
    model.eval()
    device = src_ids.device
    batch_size = src_ids.size(0)

    memory = model.encode(src_ids, src_mask)
    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = torch.ones_like(ys)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        next_token = model.output_proj(out[:, -1]).argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if bool(finished.all()):
            break

    return ys

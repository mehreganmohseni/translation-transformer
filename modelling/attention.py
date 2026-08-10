import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Scaled dot-product attention, built from nn.Linear/nn.Softmax primitives."""

    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attention_mask=None):
        d_k = query.size(-1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_padding_mask, float("-inf"))

        if self.mask_future:
            q_len, k_len = query.size(-2), key.size(-2)
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = self.softmax(scores)
        return weights @ value


class MultiHeadAttention(nn.Module):
    """Multi-head attention built on top of the single-head Attention module."""

    def __init__(self, hidden_dim: int, num_heads: int, mask_future: bool = False):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attention = Attention(mask_future=mask_future)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, num_heads, seq, head_dim)

    def _merge_heads(self, x):
        batch, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, num_heads * head_dim)

    def forward(self, query, key, value, attention_mask=None):
        batch = query.size(0)

        q = self._split_heads(self.query_transform(query))
        k = self._split_heads(self.key_transform(key))
        v = self._split_heads(self.value_transform(value))

        q = q.reshape(batch * self.num_heads, q.size(2), self.head_dim)
        k = k.reshape(batch * self.num_heads, k.size(2), self.head_dim)
        v = v.reshape(batch * self.num_heads, v.size(2), self.head_dim)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(batch, self.num_heads, -1)
            mask = mask.reshape(batch * self.num_heads, -1)
        else:
            mask = None

        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.view(batch, self.num_heads, -1, self.head_dim)
        merged = self._merge_heads(attn_out)

        return self.output_transform(merged)

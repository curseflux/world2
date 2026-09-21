"""A small-vocabulary GPT-2-style decoder, implemented independently in PyTorch."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import DIRECTIONS


class Tokenizer:
    """Atomic node IDs, configured compass actions, EOS and PAD."""

    pad_id = 0
    eos_id = 1

    def __init__(self, num_nodes: int, directions=None):
        self.num_nodes = num_nodes
        self.directions = tuple(DIRECTIONS if directions is None else directions)
        if not self.directions or len(set(self.directions)) != len(self.directions) or any(d not in DIRECTIONS for d in self.directions):
            raise ValueError("Tokenizer directions must be a nonempty unique subset of compass directions")
        self.direction_ids = {d: i + 2 for i, d in enumerate(self.directions)}
        self.id_directions = {i: d for d, i in self.direction_ids.items()}
        self.node_offset = len(self.directions) + 2
        self.vocab_size = self.node_offset + num_nodes

    def node(self, node: int) -> int:
        if not 1 <= node <= self.num_nodes:
            raise ValueError(f"Node {node} is outside 1..{self.num_nodes}")
        return self.node_offset + node - 1

    def encode(self, route: dict, eos: bool = True) -> list[int]:
        result = [self.node(route['origin']), self.node(route['destination'])]
        result.extend(self.direction_ids[d] for d in route['directions'])
        return result + [self.eos_id] if eos else result

    def decode_token(self, token: int) -> str:
        if token == self.pad_id:
            return 'PAD'
        if token == self.eos_id:
            return 'EOS'
        return self.id_directions.get(token, f"NODE:{token - self.node_offset + 1}")


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, past=None, use_cache=False):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v)]
        offset = 0 if past is None else past[0].size(-2)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=-2), torch.cat((past[1], v), dim=-2)
        mask = None
        if offset and length > 1:
            mask = (torch.arange(k.size(-2), device=x.device)[None, :]
                    <= offset + torch.arange(length, device=x.device)[:, None])
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                          dropout_p=self.dropout if self.training else 0.0,
                                          is_causal=offset == 0)
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.residual_dropout(self.out(y)), (k, v) if use_cache else None


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['dim']
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, cfg['heads'], cfg['dropout'])
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * cfg.get('mlp_ratio', 4))),
                                 nn.GELU(approximate='tanh'),
                                 nn.Linear(int(dim * cfg.get('mlp_ratio', 4)), dim),
                                 nn.Dropout(cfg['dropout']))

    def forward(self, x, past=None, use_cache=False):
        delta, cache = self.attn(self.ln1(x), past, use_cache)
        x = x + delta
        return x + self.mlp(self.ln2(x)), cache


@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    cache: list | None


class GridTransformer(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict, context_length: int):
        super().__init__()
        self.config = dict(cfg)
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, cfg['dim'])
        self.position_embedding = nn.Embedding(context_length, cfg['dim'])
        self.dropout = nn.Dropout(cfg['dropout'])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['layers'])])
        self.final_norm = nn.LayerNorm(cfg['dim'], eps=1e-5)
        self.gradient_checkpointing = False
        self.apply(self._initialize)
        # GPT-2's residual projections are scaled by the number of residual branches.
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg['layers']))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * cfg['layers']))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, tokens, past=None, use_cache=False, probe_layer=-1):
        if probe_layer < -1 or probe_layer >= len(self.blocks):
            raise ValueError('probe.layer must be -1 (final normalized) or a zero-based block index')
        offset = 0 if past is None else past[0][0].size(-2)
        if offset + tokens.size(1) > self.context_length:
            raise ValueError('Token sequence exceeds model.context_length')
        positions = torch.arange(offset, offset + tokens.size(1), device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        caches, selected = [], None
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training and not use_cache:
                x = checkpoint(lambda t, b=block: b(t)[0], x, use_reentrant=False)
                cache = None
            else:
                x, cache = block(x, None if past is None else past[i], use_cache)
            if i == probe_layer:
                selected = x
            if use_cache:
                caches.append(cache)
        x = self.final_norm(x)
        # Tied embedding / output matrix, as in GPT-2.
        return ModelOutput(F.linear(x, self.token_embedding.weight),
                           x if probe_layer == -1 else selected,
                           caches if use_cache else None)


def select_cache(cache, indices):
    return [(k.index_select(0, indices), v.index_select(0, indices)) for k, v in cache]

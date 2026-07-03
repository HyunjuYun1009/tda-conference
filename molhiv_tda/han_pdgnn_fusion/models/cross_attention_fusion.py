"""Node fusion and graph-wise cross-attention over fused latent vectors."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_bond_pair_scale(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
    max_nodes: int,
    alpha: torch.Tensor,
    num_bond_types: int = 5,
) -> torch.Tensor:
    """
    Bond-type scale for each node pair within a graph.

    Returns:
        scale [batch_size, max_nodes, max_nodes] where
        scale[g, i, j] = alpha[bond_type] if nodes i,j share a bond in graph g, else 0.
        Undirected: both (i, j) and (j, i) are set.
    """
    device = edge_attr.device
    num_nodes = batch.size(0)
    bond_type = edge_attr[:, 0].long().clamp(0, num_bond_types - 1)
    src, dst = edge_index

    scale = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    global_to_local = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    for g in range(batch_size):
        node_idx = (batch == g).nonzero(as_tuple=True)[0]
        n = node_idx.size(0)
        if n == 0:
            continue
        global_to_local[node_idx] = torch.arange(n, device=device)

        edge_mask = (batch[src] == g) & (batch[dst] == g)
        if not edge_mask.any():
            continue
        es, ed = src[edge_mask], dst[edge_mask]
        bt = bond_type[edge_mask]
        ls, ld = global_to_local[es], global_to_local[ed]
        w = alpha[bt]
        scale[g, ls, ld] = w
        scale[g, ld, ls] = w

    return scale


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, model_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h], dim=-1))


class GraphwiseCrossAttention(nn.Module):
    """
    Cross-attention where fused L attends to HAN (Z) and PDGNN (H) streams.

    Bond-modulated pairwise logits (edge-level, not per-node):
      logit_ij = (Q_i · K_j / sqrt(d)) * alpha_k   when bond type k exists between i,j
      logit_ij = -inf                               when no bond (attention weight 0)

    alpha_k is a learnable scalar per bond type (alpha ∈ R^5).
    K/V are built from cat(Z', H') without bond scaling on features.

    Nodes from different graphs cannot attend to each other (padding mask).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        mode: str = "cross",
        num_bond_types: int = 5,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mode = mode
        self.num_bond_types = num_bond_types
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim

        self.q_proj = nn.Linear(model_dim, model_dim)
        if mode == "cross":
            self.k_proj = nn.Linear(2 * model_dim, model_dim)
            self.v_proj = nn.Linear(2 * model_dim, model_dim)
            self.alpha = nn.Parameter(torch.ones(num_bond_types))
        else:
            self.k_proj = nn.Linear(model_dim, model_dim)
            self.v_proj = nn.Linear(model_dim, model_dim)
            self.alpha = None

        self.z_proj = nn.Linear(model_dim, model_dim)
        self.h_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.size()
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, hd = x.size()
        return x.transpose(1, 2).contiguous().view(b, n, h * hd)

    def forward(
        self,
        L: torch.Tensor,
        Z: torch.Tensor,
        H: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = L.device
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        max_nodes = 0
        for g in range(batch_size):
            max_nodes = max(max_nodes, int((batch == g).sum().item()))
        if max_nodes == 0:
            return L

        L_pad = torch.zeros(batch_size, max_nodes, self.model_dim, device=device)
        Z_pad = torch.zeros(batch_size, max_nodes, self.model_dim, device=device)
        H_pad = torch.zeros(batch_size, max_nodes, self.model_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for g in range(batch_size):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            n = idx.size(0)
            if n == 0:
                continue
            L_pad[g, :n] = L[idx]
            Z_pad[g, :n] = Z[idx]
            H_pad[g, :n] = H[idx]
            mask[g, :n] = True

        Q = self._split_heads(self.q_proj(L_pad))
        if self.mode == "cross":
            kv_src = torch.cat([self.z_proj(Z_pad), self.h_proj(H_pad)], dim=-1)
            K = self._split_heads(self.k_proj(kv_src))
            V = self._split_heads(self.v_proj(kv_src))
        else:
            K = self._split_heads(self.k_proj(L_pad))
            V = self._split_heads(self.v_proj(L_pad))

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        if edge_index is not None and edge_attr is not None and self.alpha is not None:
            bond_scale = build_bond_pair_scale(
                edge_index,
                edge_attr,
                batch,
                batch_size,
                max_nodes,
                self.alpha,
                self.num_bond_types,
            )
            attn_logits = attn_logits * bond_scale.unsqueeze(1)
            bonded = pair_mask & (bond_scale > 0)
            attn_logits = attn_logits.masked_fill(~bonded.unsqueeze(1), -1e9)
        else:
            attn_logits = attn_logits.masked_fill(~pair_mask.unsqueeze(1), -1e9)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        context = self._merge_heads(torch.matmul(attn, V))
        out_pad = self.out_proj(context)

        out = torch.zeros_like(L)
        for g in range(batch_size):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            n = idx.size(0)
            if n == 0:
                continue
            out[idx] = out_pad[g, :n] + L[idx]
        return out


def graphwise_cross_attention(
    L: torch.Tensor,
    Z: torch.Tensor,
    H: torch.Tensor,
    batch: torch.Tensor,
    module: GraphwiseCrossAttention,
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    return module(L, Z, H, batch, edge_index, edge_attr)

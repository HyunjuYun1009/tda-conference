"""Fine-tuning model: frozen PDGNN(+TDA) backbone + a lightweight bond-relation
HAN branch fused at the node level (before add-pooling).

Design summary (see project brief for the approved decisions):
- The PDGNN backbone (AtomEncoder, PDConv stack, edge_phys_proj) is loaded from a
  pretrained checkpoint and fully frozen. It acts purely as a feature extractor.
- A HAN branch produces per-node embeddings h_han[N, H] using bond types as
  heterogeneous relations, operating directly on the homogeneous edge_index /
  edge_attr (no heavy HeteroData conversion).
- Fusion happens at the PDGNN encode() point right before global_add_pool: the
  backbone's node features x[N, 600] are concatenated with a gated HAN output
  (gate * h_han) -> [N, 600 + H], then pooled together. Because pooling is additive,
  concat-then-pool == pool-then-concat, so this is equivalent to pooling each
  branch and concatenating the graph vectors.
- A learnable scalar gate (init 0) multiplies the HAN output. At init the HAN
  contribution is exactly 0, so the well-tuned PDGNN behaviour is preserved; the
  gate (and then the HAN branch) is learned in gradually.
- A new head takes [graph_dim(600) + H + extra(BondTDA 50 + 3D TDA 75)] and is
  trained from scratch (the head input dimensionality changed vs. the backbone).

Trainable = HAN branch + gate + new head. Everything under `backbone` is frozen.

HAN output dim H defaults to 128: this keeps the number of added parameters modest
relative to the 600-dim PDGNN graph vector (so it cannot overwhelm the frozen
backbone), while still giving the two-level relation/semantic attention enough
capacity. It also matches the hidden size used by the existing HAN references in
this repo (models/han_molecule.py, han_pdgnn_fusion/models/han_encoder.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax, scatter

from models.pdgnn_tda import PDGNNTDA

# OGB bond-type relations (edge_attr[:, 0]): single, double, triple, aromatic, misc.
NUM_BOND_RELATIONS = 5


class RelationGATConv(nn.Module):
    """Multi-head GAT-style attention aggregation for a single bond relation.

    Operates on the shared (homogeneous) node set; only the edges belonging to
    one relation are passed in. Nodes with no incoming edges of this relation get
    a zero output, which the semantic attention then simply down-weights.
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x.new_zeros(num_nodes, self.hidden_dim)
        H, C = self.heads, self.head_dim
        xh = self.lin(x).view(-1, H, C)  # [N, H, C]
        src, dst = edge_index
        alpha = (xh[src] * self.att_src).sum(-1) + (xh[dst] * self.att_dst).sum(-1)  # [E, H]
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, dst, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = xh[src] * alpha.unsqueeze(-1)  # [E, H, C]
        out = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")
        return out.reshape(num_nodes, H * C)


class HANLayer(nn.Module):
    """One HAN layer: per-relation node attention + semantic-level attention."""

    def __init__(self, hidden_dim: int, num_relations: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.rel_convs = nn.ModuleList(
            [RelationGATConv(hidden_dim, heads, dropout) for _ in range(num_relations)]
        )
        self.semantic_query = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        rel_outs = []
        for r, conv in enumerate(self.rel_convs):
            mask = edge_type == r
            rel_outs.append(conv(x, edge_index[:, mask], num_nodes))
        stacked = torch.stack(rel_outs, dim=0)  # [R, N, H]

        # Semantic-level attention: score each relation output per node.
        query = torch.tanh(self.semantic_query(x)).unsqueeze(0)  # [1, N, H]
        scores = (stacked * query).sum(-1) / (self.hidden_dim ** 0.5)  # [R, N]
        beta = torch.softmax(scores, dim=0).unsqueeze(-1)  # [R, N, 1]
        fused = (beta * stacked).sum(dim=0)  # [N, H]
        return F.relu(x + fused)


class BondRelationHAN(nn.Module):
    """Lightweight HAN over bond-type relations, producing per-node embeddings.

    Reuses the incoming homogeneous edge_index/edge_attr; the relation for each
    edge is edge_attr[:, 0] (the OGB bond type). Output: h_han[N, hidden_dim].
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        num_relations: int = NUM_BOND_RELATIONS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            [HANLayer(hidden_dim, num_relations, heads, dropout) for _ in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, batch) -> torch.Tensor:
        num_nodes = batch.x.size(0)
        x = self.atom_encoder(batch.x)
        edge_type = batch.edge_attr[:, 0].long().clamp(0, self.num_relations - 1)
        for layer in self.layers:
            x = layer(x, batch.edge_index, edge_type, num_nodes)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PDGNNHANFinetune(nn.Module):
    """Frozen PDGNN(+TDA) backbone + gated HAN node branch + new head."""

    def __init__(
        self,
        num_tasks: int = 1,
        num_layers: int = 4,
        emb_dim: int = 300,
        dropout: float = 0.3,
        use_bond_tda: bool = True,
        bond_tda_dim: int = 50,
        use_mw: bool = False,
        use_tda_3d: bool = True,
        tda_3d_dim: int = 75,
        use_edge_electro: bool = True,
        edge_phys_dim: int = 2,
        han_hidden: int = 128,
        han_layers: int = 2,
        han_heads: int = 4,
        han_dropout: float = 0.2,
        backbone_ckpt: Optional[str | Path] = None,
    ):
        super().__init__()
        self.use_bond_tda = use_bond_tda
        self.use_mw = use_mw
        self.use_tda_3d = use_tda_3d

        # Backbone with the exact same config as the pretrained checkpoint so the
        # state_dict loads cleanly. Its own head is kept but unused (and frozen).
        self.backbone = PDGNNTDA(
            num_tasks=num_tasks,
            num_layers=num_layers,
            emb_dim=emb_dim,
            dropout=dropout,
            use_bond_tda=use_bond_tda,
            bond_tda_dim=bond_tda_dim,
            use_mw=use_mw,
            use_tda_3d=use_tda_3d,
            tda_3d_dim=tda_3d_dim,
            use_edge_electro=use_edge_electro,
            edge_phys_dim=edge_phys_dim,
        )
        if backbone_ckpt is not None:
            state = torch.load(backbone_ckpt, map_location="cpu")
            self.backbone.load_state_dict(state)
        self.freeze_backbone()

        # HAN branch + gate (init 0 -> no disruption at start).
        self.han = BondRelationHAN(
            hidden_dim=han_hidden,
            num_layers=han_layers,
            heads=han_heads,
            dropout=han_dropout,
        )
        self.gate = nn.Parameter(torch.zeros(1))

        graph_dim = 2 * emb_dim
        extra_dim = 0
        if use_bond_tda:
            extra_dim += bond_tda_dim
        if use_mw:
            extra_dim += 1
        if use_tda_3d:
            extra_dim += tda_3d_dim

        head_in = graph_dim + han_hidden + extra_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 2 * emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * emb_dim, num_tasks),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode (deterministic features)."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        batch,
        bond_tda: torch.Tensor | None = None,
        mw: torch.Tensor | None = None,
        tda_3d: torch.Tensor | None = None,
    ):
        # Frozen PDGNN node features (pre-pool), no grad.
        with torch.no_grad():
            node_x = self.backbone.backbone.encode_nodes(batch)  # [N, 600]

        h_han = self.han(batch)  # [N, H]
        node_cat = torch.cat([node_x, self.gate * h_han], dim=1)  # [N, 600 + H]
        graph_emb = global_add_pool(node_cat, batch.batch)  # [B, 600 + H]

        extras = [graph_emb]
        if self.use_bond_tda:
            if bond_tda is None:
                raise ValueError("bond_tda features required but not provided")
            extras.append(bond_tda)
        if self.use_mw:
            if mw is None:
                raise ValueError("molecular weight features required but not provided")
            extras.append(mw)
        if self.use_tda_3d:
            if tda_3d is None:
                raise ValueError("3D TDA features required but not provided")
            extras.append(tda_3d)

        return self.head(torch.cat(extras, dim=1))

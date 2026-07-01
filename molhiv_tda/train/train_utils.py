"""Shared training and evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from torch_geometric.data import Batch


def gather_graph_features(
    batch,
    feature_tensor: torch.Tensor,
) -> torch.Tensor:
    """Select precomputed graph-level features for a batch."""
    idx = batch.graph_idx.view(-1)
    return feature_tensor[idx]


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    evaluator: Evaluator,
    device,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        kwargs = {}
        if bond_tda is not None:
            kwargs["bond_tda"] = gather_graph_features(batch, bond_tda.to(device))
        if mw is not None:
            kwargs["mw"] = gather_graph_features(batch, mw.to(device))
        if tda_3d is not None:
            kwargs["tda_3d"] = gather_graph_features(batch, tda_3d.to(device))

        if kwargs:
            pred = model(batch, **kwargs)
        else:
            pred = model(batch)

        y_true.append(batch.y.view(batch.y.size(0), -1).detach().cpu())
        y_pred.append(pred.view(pred.size(0), -1).detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return {"rocauc": evaluator.eval(input_dict)["rocauc"]}


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        kwargs = {}
        if bond_tda is not None:
            kwargs["bond_tda"] = gather_graph_features(batch, bond_tda.to(device))
        if mw is not None:
            kwargs["mw"] = gather_graph_features(batch, mw.to(device))
        if tda_3d is not None:
            kwargs["tda_3d"] = gather_graph_features(batch, tda_3d.to(device))

        if kwargs:
            pred = model(batch, **kwargs)
        else:
            pred = model(batch)

        is_labeled = batch.y == batch.y
        loss = F.binary_cross_entropy_with_logits(
            pred.to(torch.float32)[is_labeled],
            batch.y.to(torch.float32)[is_labeled],
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


def run_training(
    model,
    loaders,
    evaluator,
    device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    bond_tda: Optional[torch.Tensor] = None,
    mw: Optional[torch.Tensor] = None,
    tda_3d: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid = -1.0
    best_test = -1.0
    best_state = None
    stale = 0

    for _ in range(epochs):
        train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
        )
        valid_score = evaluate_model(
            model,
            loaders["valid"],
            evaluator,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
        )["rocauc"]
        test_score = evaluate_model(
            model,
            loaders["test"],
            evaluator,
            device,
            bond_tda=bond_tda,
            mw=mw,
            tda_3d=tda_3d,
        )["rocauc"]

        if valid_score > best_valid:
            best_valid = valid_score
            best_test = test_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"valid_rocauc": best_valid, "test_rocauc": best_test}


def save_result(result: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


#!/usr/bin/env python3
"""Train PDGNN with TDA / molecular weight ablations on OGBG-MolHIV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    BATCH_SIZE,
    BOND_TDA_DIM,
    BOND_TDA_CACHE,
    DEFAULT_DEVICE,
    DROPOUT,
    EMB_DIM,
    EPOCHS,
    MW_CACHE,
    NUM_BACKBONE_LAYERS,
    NUM_WORKERS,
    PATIENCE,
    RESULTS_ROOT,
    TDA_3D_CACHE,
    TDA_3D_DIM,
    LR,
    WEIGHT_DECAY,
)
from data.load_molhiv import load_feature_tensor, load_molhiv, make_loaders
from models.pdgnn_tda import PDGNNTDA
from train.train_utils import run_training, save_result
from utils.device import device_label, resolve_device

CONFIGS = {
    "pdgnn_mw": dict(use_bond_tda=False, use_mw=True, use_tda_3d=False, label="PDGNN + MW"),
    "pdgnn_bond_tda": dict(use_bond_tda=True, use_mw=False, use_tda_3d=False, label="PDGNN + BondTDA"),
    "pdgnn_bond_tda_mw": dict(use_bond_tda=True, use_mw=True, use_tda_3d=False, label="PDGNN + BondTDA + MW"),
    "pdgnn_3d_tda": dict(use_bond_tda=False, use_mw=False, use_tda_3d=True, label="PDGNN + 3DTDA"),
    "pdgnn_bond_tda_3d_mw": dict(
        use_bond_tda=True, use_mw=True, use_tda_3d=True, label="PDGNN + BondTDA + 3DTDA + MW"
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device_label(device)}")
    cfg = CONFIGS[args.config]

    dataset, split_idx, evaluator, _ = load_molhiv(args.dataset_root)
    loaders = make_loaders(
        dataset, split_idx,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers if device.type == "cuda" else 0,
    )

    bond_tda = load_feature_tensor(BOND_TDA_CACHE, len(dataset), BOND_TDA_DIM) if cfg["use_bond_tda"] else None
    mw = load_feature_tensor(MW_CACHE, len(dataset), 1) if cfg["use_mw"] else None
    tda_3d = load_feature_tensor(TDA_3D_CACHE, len(dataset), TDA_3D_DIM) if cfg["use_tda_3d"] else None

    if cfg["use_bond_tda"] and not BOND_TDA_CACHE.exists():
        raise FileNotFoundError(f"Missing {BOND_TDA_CACHE}. Run scripts/preprocess_bond_tda.py first.")
    if cfg["use_mw"] and not MW_CACHE.exists():
        raise FileNotFoundError(f"Missing {MW_CACHE}. Run scripts/preprocess_molecular_weight.py first.")
    if cfg["use_tda_3d"] and not TDA_3D_CACHE.exists():
        raise FileNotFoundError(f"Missing {TDA_3D_CACHE}. Run scripts/preprocess_3d_tda.py first.")

    model = PDGNNTDA(
        num_tasks=dataset.num_tasks,
        num_layers=NUM_BACKBONE_LAYERS,
        emb_dim=EMB_DIM,
        dropout=DROPOUT,
        use_bond_tda=cfg["use_bond_tda"],
        bond_tda_dim=BOND_TDA_DIM,
        use_mw=cfg["use_mw"],
        use_tda_3d=cfg["use_tda_3d"],
        tda_3d_dim=TDA_3D_DIM,
    ).to(device)

    metrics = run_training(
        model,
        loaders,
        evaluator,
        device,
        epochs=args.epochs,
        patience=PATIENCE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        bond_tda=bond_tda,
        mw=mw,
        tda_3d=tda_3d,
    )

    result = {
        "model": cfg["label"],
        "backbone": "PDGNN",
        "bond_type_tda": cfg["use_bond_tda"],
        "molecular_weight": cfg["use_mw"],
        "tda_3d": cfg["use_tda_3d"],
        **metrics,
    }
    out = RESULTS_ROOT / f"{args.config}.json"
    save_result(result, out)
    print(result)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

import os

import torch
import yaml

from .data import get_data, loader
from .train import train_net


def load_cfg(path="configs/default.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("quick_test"):
        cfg["widths"] = [16, 64, 256]
        cfg["two_layer_widths"] = [64, 128]
        cfg["k_seeds"] = 2
        cfg["epochs"] = 3
        cfg["eval_n"] = 2000
        cfg["snapshot_epochs"] = [1, 2, 3]

    for dirname in ("ckpt_dir", "res_dir", "fig_dir"):
        os.makedirs(cfg[dirname], exist_ok=True)

    return cfg


def setup(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)

    Xtr, ytr, Xev, yev = get_data(cfg["dataset"], cfg["eval_n"], device)
    train_loader = loader(Xtr, ytr, cfg["batch"])

    return device, train_loader, Xev, yev


def trained_pair(hidden, pair_id, cfg, train_loader, device, snapshots=None):
    params_a, snapshots_a = train_net(hidden, 100 + 2*pair_id, cfg, train_loader, device, snapshots)
    params_b, snapshots_b = train_net(hidden, 101 + 2*pair_id, cfg, train_loader, device, snapshots)

    return (params_a, snapshots_a), (params_b, snapshots_b)
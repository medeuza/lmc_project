import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .models import MLP, get_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_net(hidden, seed, cfg, train_loader, device, snapshots=None, step_snapshots=None):
    hidden_tag = "x".join(map(str, hidden))
    tag = f"{cfg['dataset']}_h{hidden_tag}_s{seed}_e{cfg['epochs']}.pt"

    path = os.path.join(cfg["ckpt_dir"], tag)
    snapshot_path = os.path.join(cfg["ckpt_dir"], "snap_" + tag)

    has_snapshots = snapshots is not None or step_snapshots is not None

    if os.path.exists(path) and (not has_snapshots or os.path.exists(snapshot_path)):
        final_params = torch.load(path, map_location="cpu")
        saved_snapshots = torch.load(snapshot_path, map_location="cpu") if has_snapshots else {}
        return final_params, saved_snapshots

    set_seed(seed)

    model = MLP(hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    saved_snapshots = {}
    step = 0
    step_set = set(step_snapshots or [])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()

            step += 1

            if step in step_set:
                saved_snapshots[f"step{step}"] = get_params(model)

        if snapshots and epoch in snapshots:
            saved_snapshots[f"epoch{epoch}"] = get_params(model)

    final_params = get_params(model)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    torch.save(final_params, path)

    if has_snapshots:
        torch.save(saved_snapshots, snapshot_path)

    return final_params, saved_snapshots
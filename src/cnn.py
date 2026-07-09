import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class SmallCNN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.conv1 = nn.Conv2d(1, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.fc = nn.Linear(c, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def get_params(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_cnn(params, c, device):
    model = SmallCNN(c).to(device)
    model.load_state_dict(params)
    model.eval()
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_channel_perms(params, perm1, perm2):
    out = {key: value.clone() for key, value in params.items()}

    out["conv1.weight"] = out["conv1.weight"][perm1]
    out["conv1.bias"] = out["conv1.bias"][perm1]
    out["conv2.weight"] = out["conv2.weight"][:, perm1]

    out["conv2.weight"] = out["conv2.weight"][perm2]
    out["conv2.bias"] = out["conv2.bias"][perm2]
    out["fc.weight"] = out["fc.weight"][:, perm2]

    return out


def _lap(cost):
    _, col = linear_sum_assignment(-cost.numpy())
    return torch.as_tensor(col, dtype=torch.long)


def weight_matching_cnn(params_a, params_b, iters=20):
    c = params_a["conv1.weight"].shape[0]

    perm1 = torch.arange(c)
    perm2 = torch.arange(c)

    for i in range(iters):
        params_b_cur = apply_channel_perms(params_b, perm1, perm2)

        a1 = params_a["conv1.weight"].reshape(c, -1)
        b1 = params_b_cur["conv1.weight"].reshape(c, -1)

        a_out = params_a["conv2.weight"].permute(1, 0, 2, 3).reshape(c, -1)
        b_out = params_b_cur["conv2.weight"].permute(1, 0, 2, 3).reshape(c, -1)

        cost1 = a1 @ b1.T + a_out @ b_out.T
        new_perm1 = perm1[_lap(cost1)]

        params_b_cur = apply_channel_perms(params_b, new_perm1, perm2)

        a2 = params_a["conv2.weight"].reshape(c, -1)
        b2 = params_b_cur["conv2.weight"].reshape(c, -1)

        cost2 = a2 @ b2.T + params_a["fc.weight"].T @ params_b_cur["fc.weight"]
        new_perm2 = perm2[_lap(cost2)]

        if torch.equal(new_perm1, perm1) and torch.equal(new_perm2, perm2):
            perm1, perm2 = new_perm1, new_perm2
            break

        perm1, perm2 = new_perm1, new_perm2

    return (perm1, perm2), apply_channel_perms(params_b, perm1, perm2)


def alignability_cnn(params_a, params_b_aligned):
    c = params_a["conv1.weight"].shape[0]

    a1 = params_a["conv1.weight"].reshape(c, -1)
    b1 = params_b_aligned["conv1.weight"].reshape(c, -1)

    a2 = params_a["conv2.weight"].reshape(c, -1)
    b2 = params_b_aligned["conv2.weight"].reshape(c, -1)

    sim1 = F.cosine_similarity(a1, b1, dim=1)
    sim2 = F.cosine_similarity(a2, b2, dim=1)

    return float(torch.cat([sim1, sim2]).mean().item())


def get_img_data(name, eval_n, device, root="./data"):
    from torchvision import datasets, transforms

    dataset = getattr(datasets, name)
    transform = transforms.ToTensor()

    train_data = dataset(root=root, train=True, download=True, transform=transform)
    test_data = dataset(root=root, train=False, download=True, transform=transform)

    Xtr = train_data.data.float().unsqueeze(1)/255.0
    ytr = train_data.targets.long()

    Xte = test_data.data.float().unsqueeze(1)/255.0
    yte = test_data.targets.long()

    idx = torch.randperm(len(Xte))[:eval_n]

    return Xtr, ytr, Xte[idx].to(device), yte[idx].to(device)


def train_cnn(c, seed, cfg, Xtr, ytr, device):
    tag = f"cnn_{cfg['dataset']}_c{c}_s{seed}_e{cfg['epochs']}.pt"
    path = os.path.join(cfg["ckpt_dir"], tag)

    if os.path.exists(path):
        return torch.load(path, map_location="cpu")

    set_seed(seed)

    model = SmallCNN(c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    n = len(Xtr)
    batch_size = cfg["batch"]

    for j in range(cfg["epochs"]):
        perm = torch.randperm(n)

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)

            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()

    params = get_params(model)
    torch.save(params, path)

    return params


@torch.no_grad()
def eval_cnn(params, c, X, y, device):
    output = make_cnn(params, c, device)(X)
    loss = F.cross_entropy(output, y).item()
    err = (output.argmax(1) != y).float().mean().item()
    return loss, err


def barrier_cnn(params_a, params_b, c, X, y, device, n=25):
    alphas = np.linspace(0, 1, n)

    loss_a, _ = eval_cnn(params_a, c, X, y, device)
    loss_b, _ = eval_cnn(params_b, c, X, y, device)

    gaps = []

    for alpha in alphas:
        params_t = {
            key: (1 - alpha) * params_a[key] + alpha * params_b[key]
            for key in params_a
        }

        loss, _ = eval_cnn(params_t, c, X, y, device)
        gaps.append(loss - ((1 - alpha) * loss_a + alpha * loss_b))

    return float(max(gaps))


@torch.no_grad()
def _preact_stats_cnn(params, X):
    z1 = F.conv2d(X, params["conv1.weight"], params["conv1.bias"], padding=1)
    a1 = torch.relu(z1)
    z2 = F.conv2d(a1, params["conv2.weight"], params["conv2.bias"], padding=1)

    mean1 = z1.mean(dim=(0, 2, 3)).cpu()
    std1 = z1.std(dim=(0, 2, 3)).cpu()

    mean2 = z2.mean(dim=(0, 2, 3)).cpu()
    std2 = z2.std(dim=(0, 2, 3)).cpu()

    return (mean1, std1), (mean2, std2)


@torch.no_grad()
def repair_interpolate_cnn(params_a, params_b, alpha, X, stats_a=None, stats_b=None):
    if stats_a is None:
        stats_a = _preact_stats_cnn(params_a, X)

    if stats_b is None:
        stats_b = _preact_stats_cnn(params_b, X)

    (mean_a1, std_a1), (mean_a2, std_a2) = stats_a
    (mean_b1, std_b1), (mean_b2, std_b2) = stats_b

    target_mean1 = (1 - alpha)*mean_a1 + alpha*mean_b1
    target_std1 = (1 - alpha)*std_a1 + alpha*std_b1

    target_mean2 = (1 - alpha)*mean_a2 + alpha*mean_b2
    target_std2 = (1 - alpha)*std_a2 + alpha *std_b2

    params = {
        key: (1 - alpha)*params_a[key] + alpha*params_b[key]
        for key in params_a
    }

    (cur_mean1, cur_std1), _ = _preact_stats_cnn(params, X)

    scale1 = target_std1/(cur_std1 + 1e-6)
    shift1 = target_mean1 - scale1*cur_mean1

    params["conv1.weight"] = params["conv1.weight"]*scale1[:, None, None, None]
    params["conv1.bias"] = params["conv1.bias"]*scale1 + shift1

    _, (cur_mean2, cur_std2) = _preact_stats_cnn(params, X)

    scale2 = target_std2/(cur_std2 + 1e-6)
    shift2 = target_mean2 - scale2*cur_mean2

    params["conv2.weight"] = params["conv2.weight"]*scale2[:, None, None, None]
    params["conv2.bias"] = params["conv2.bias"]*scale2 + shift2

    return params


def barrier_cnn_full(params_a, params_b, c, X, y, device, n=25, repair=False, stats_X=None):
    alphas = np.linspace(0, 1, n)

    loss_a, err_a = eval_cnn(params_a, c, X, y, device)
    loss_b, err_b = eval_cnn(params_b, c, X, y, device)

    X_stats = X if stats_X is None else stats_X

    stats_a = _preact_stats_cnn(params_a, X_stats) if repair else None
    stats_b = _preact_stats_cnn(params_b, X_stats) if repair else None

    loss_gaps = []
    err_gaps = []

    for alpha in alphas:
        if repair:
            params_t = repair_interpolate_cnn(params_a, params_b, alpha, X_stats, stats_a, stats_b)
        else:
            params_t = {
                key: (1 - alpha)*params_a[key] + alpha*params_b[key]
                for key in params_a
            }

        loss, err = eval_cnn(params_t, c, X, y, device)
        loss_gaps.append(loss - ((1 - alpha)*loss_a + alpha*loss_b))
        err_gaps.append(err - ((1 - alpha)*err_a + alpha*err_b))

    return {"loss": float(max(loss_gaps)),"err": float(max(err_gaps)),}
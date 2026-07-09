import numpy as np
import torch
import torch.nn.functional as F

from .barrier import eval_params
from .models import make_net


def alignability_blocks(pA, pBp, n_layers):
    align_in = []
    align_out = []
    for h in range(1, n_layers):
        align_in.append(F.cosine_similarity(
                pA[f"layers.{h - 1}.weight"],
                pBp[f"layers.{h - 1}.weight"],
                dim=1))
        align_out.append(
            F.cosine_similarity(
                pA[f"layers.{h}.weight"].T,
                pBp[f"layers.{h}.weight"].T,
                dim=1))

    return float(torch.cat(align_in).mean().item()), float(torch.cat(align_out).mean().item())


@torch.no_grad()
def disagreement(pA, pB, hidden, X, device):
    pred_a = make_net(pA, hidden, device)(X).argmax(1)
    pred_b = make_net(pB, hidden, device)(X).argmax(1)
    return float((pred_a != pred_b).float().mean().item())


@torch.no_grad()
def linear_cka(pA, pB, hidden, X, device):
    _, acts_a = make_net(pA, hidden, device)(X, return_acts=True)
    _, acts_b = make_net(pB, hidden, device)(X, return_acts=True)

    values = []

    for acts1, acts2 in zip(acts_a, acts_b):
        acts1 = acts1 - acts1.mean(0, keepdim=True)
        acts2 = acts2 - acts2.mean(0, keepdim=True)

        hsic = (acts1.T @ acts2).norm() ** 2
        norm = (acts1.T @ acts1).norm()*(acts2.T @ acts2).norm() + 1e-12

        values.append(float(hsic/norm))

    return float(np.mean(values))


@torch.no_grad()
def flatness_proxy(params, hidden, X, y, device, sigma=0.05, n_draws=5, seed=0):
    base_loss, _ = eval_params(params, hidden, X, y, device)
    generator = torch.Generator().manual_seed(seed)

    increases = []

    for _ in range(n_draws):
        noisy_params = {
            key: value + torch.randn(value.shape, generator=generator)*(sigma*value.std())
            for key, value in params.items()
        }

        loss, _ = eval_params(noisy_params, hidden, X, y, device)
        increases.append(loss - base_loss)

    return float(np.mean(increases))
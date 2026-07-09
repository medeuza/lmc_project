import numpy as np
import torch
import torch.nn.functional as F

from .models import make_net


def interpolate(pA, pB, alpha):
    return {key: (1 - alpha) * pA[key] + alpha*pB[key] for key in pA}


@torch.no_grad()
def eval_params(params, hidden, X, y, device):
    model = make_net(params, hidden, device)
    output = model(X)
    loss = F.cross_entropy(output, y).item()
    error = (output.argmax(1) != y).float().mean().item()
    return loss, error

def barrier(pA, pB, hidden, X, y, device, n=25):
    alphas = np.linspace(0, 1, n)

    loss_a, err_a = eval_params(pA, hidden, X, y, device)
    loss_b, err_b = eval_params(pB, hidden, X, y, device)

    loss_curve = []
    err_curve = []

    for alpha in alphas:
        loss, err = eval_params(interpolate(pA, pB, alpha), hidden, X, y, device)
        loss_curve.append(loss)
        err_curve.append(err)

    loss_barrier = [
        loss_curve[i] - ((1 - alpha) * loss_a + alpha * loss_b)
        for i, alpha in enumerate(alphas)
    ]

    err_barrier = [
        err_curve[i] - ((1 - alpha) * err_a + alpha * err_b)
        for i, alpha in enumerate(alphas)
    ]

    return {"loss": float(max(loss_barrier)),
        "err": float(max(err_barrier)),
        "alphas": alphas.tolist(),
        "curve_loss": loss_curve,
        "curve_err": err_curve}


@torch.no_grad()
def preact_stats_layer0(params, hidden, X, device):
    model = make_net(params, hidden, device)
    activations = model.layers[0](X)
    return activations.mean(0).cpu(), activations.std(0).cpu()


@torch.no_grad()
def repair_interpolate(pA, pB, alpha, hidden, X, device):
    mean_a, std_a = preact_stats_layer0(pA, hidden, X, device)
    mean_b, std_b = preact_stats_layer0(pB, hidden, X, device)

    target_mean = (1 - alpha)*mean_a + alpha*mean_b
    target_std = (1 - alpha)*std_a + alpha*std_b

    params = interpolate(pA, pB, alpha)
    current_mean, current_std = preact_stats_layer0(params, hidden, X, device)

    scale = target_std/(current_std + 1e-6)
    shift = target_mean - scale*current_mean

    params["layers.0.weight"] *= scale[:, None]
    params["layers.0.bias"] = params["layers.0.bias"]*scale + shift

    return params


def barrier_repair(pA, pB, hidden, X, y, device, n=25):
    alphas = np.linspace(0, 1, n)

    loss_a, _ = eval_params(pA, hidden, X, y, device)
    loss_b, _ = eval_params(pB, hidden, X, y, device)

    barrier_values = []

    for alpha in alphas:
        loss, _ = eval_params(repair_interpolate(pA, pB, alpha, hidden, X, device), hidden, X, y, device,)
        barrier_values.append(loss - ((1 - alpha) * loss_a + alpha*loss_b))

    return float(max(barrier_values))
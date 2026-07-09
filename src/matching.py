import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _weight_keys(n_layers):
    weights = [f"layers.{i}.weight" for i in range(n_layers)]
    biases = [f"layers.{i}.bias" for i in range(n_layers)]
    return weights, biases


def apply_perms(params, perms, n_layers):
    weights, biases = _weight_keys(n_layers)
    out = {key: value.clone() for key, value in params.items()}

    for layer, perm in perms.items():
        out[weights[layer - 1]] = out[weights[layer - 1]][perm]
        out[biases[layer - 1]] = out[biases[layer - 1]][perm]
        out[weights[layer]] = out[weights[layer]][:, perm]

    return out


def weight_matching(params_a, params_b, n_layers, iters=25, seed=0):
    weights, _ = _weight_keys(n_layers)

    n_hidden = n_layers - 1
    sizes = {
        layer: params_a[weights[layer - 1]].shape[0]
        for layer in range(1, n_hidden + 1)
    }

    perms = {
        layer: torch.arange(sizes[layer])
        for layer in range(1, n_hidden + 1)
    }

    rng = np.random.default_rng(seed)

    for _ in range(iters):
        changed = False
        order = list(range(1, n_hidden + 1))
        rng.shuffle(order)

        for layer in order:
            current_params_b = apply_perms(params_b, perms, n_layers)

            cost = (
                params_a[weights[layer - 1]] @ current_params_b[weights[layer - 1]].T
                + params_a[weights[layer]].T @ current_params_b[weights[layer]]
            )

            _, col = linear_sum_assignment(-cost.numpy())
            new_perm = perms[layer][torch.as_tensor(col, dtype=torch.long)]

            if not torch.equal(new_perm, perms[layer]):
                perms[layer] = new_perm
                changed = True

        if not changed:
            break

    return perms, apply_perms(params_b, perms, n_layers)


@torch.no_grad()
def activation_matching(model_a, model_b, X, n_layers):
    _, activations_a = model_a(X, return_acts=True)
    _, activations_b = model_b(X, return_acts=True)

    perms = {}

    for layer in range(1, n_layers):
        acts_a = activations_a[layer - 1] - activations_a[layer - 1].mean(0, keepdim=True)
        acts_b = activations_b[layer - 1] - activations_b[layer - 1].mean(0, keepdim=True)

        acts_a = acts_a / (acts_a.norm(dim=0, keepdim=True) + 1e-8)
        acts_b = acts_b / (acts_b.norm(dim=0, keepdim=True) + 1e-8)

        cost = (acts_a.T @ acts_b).cpu()

        _, col = linear_sum_assignment(-cost.numpy())
        perms[layer] = torch.as_tensor(col, dtype=torch.long)

    return perms
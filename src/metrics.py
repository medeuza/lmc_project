import numpy as np
import torch
import torch.nn.functional as F


def alignability(pA, pBp, n_layers):
    sims = []

    for h in range(1, n_layers):
        fA = torch.cat([pA[f"layers.{h - 1}.weight"], pA[f"layers.{h}.weight"].T], dim=1)
        fB = torch.cat([pBp[f"layers.{h - 1}.weight"], pBp[f"layers.{h}.weight"].T], dim=1)
        sims.append(F.cosine_similarity(fA, fB, dim=1))

    return torch.cat(sims).mean().item()


def assignment_cost(pA, pBp, n_layers):
    return 1.0 - alignability(pA, pBp, n_layers)


def partial_correlation(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    def residual(a, b):
        design = np.c_[np.ones_like(b), b]
        beta, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ beta

    rx = residual(x, z)
    ry = residual(y, z)

    return float(np.corrcoef(rx, ry)[0, 1])


def mediation_regression(width, align, barrier):
    def standardize(values):
        values = np.asarray(values, float)
        return (values - values.mean())/(values.std() + 1e-9)

    W = standardize(np.log2(np.asarray(width, float)))
    A = standardize(align)
    B = standardize(barrier)

    X = np.c_[np.ones_like(W), W, A]
    beta, *_ = np.linalg.lstsq(X, B, rcond=None)

    Xw = np.c_[np.ones_like(W), W]
    beta_w, *_ = np.linalg.lstsq(Xw, B, rcond=None)

    ss_total = float((B ** 2).sum()) + 1e-12

    return {
        "beta_width_alone": float(beta_w[1]),
        "beta_width_ctrl_align": float(beta[1]),
        "beta_align_ctrl_width": float(beta[2]),
        "partial_corr_align_barrier": partial_correlation(align, barrier, np.log2(np.asarray(width, float))),
        "r2_width_alone": 1.0 - float(((B - Xw @ beta_w) ** 2).sum()) / ss_total,
        "r2_width_align": 1.0 - float(((B - X @ beta) ** 2).sum()) / ss_total,
    }


def alignability_per_neuron(pA, pBp, n_layers):
    sims = []

    for h in range(1, n_layers):
        fA = torch.cat([pA[f"layers.{h - 1}.weight"], pA[f"layers.{h}.weight"].T], dim=1)
        fB = torch.cat([pBp[f"layers.{h - 1}.weight"], pBp[f"layers.{h}.weight"].T], dim=1)
        sims.append(F.cosine_similarity(fA, fB, dim=1))

    return torch.cat(sims)


def tail_alignability(pA, pBp, n_layers, q=0.10):
    sims = alignability_per_neuron(pA, pBp, n_layers)
    return float(torch.quantile(sims, q).item())


def mediation_two_mediators(width, mean_align, tail_align, barrier):
    def standardize(values):
        values = np.asarray(values, float)
        return (values - values.mean())/(values.std() + 1e-9)

    W = standardize(np.log2(np.asarray(width, float)))
    M = standardize(mean_align)
    T = standardize(tail_align)
    B = standardize(barrier)

    X1 = np.c_[np.ones_like(W), W, M]
    beta1, *_ = np.linalg.lstsq(X1, B, rcond=None)
    residual1 = B - X1 @ beta1

    X2 = np.c_[np.ones_like(W), W, M, T]
    beta2, *_ = np.linalg.lstsq(X2, B, rcond=None)
    residual2 = B - X2 @ beta2

    def ss(residual):
        return float((residual ** 2).sum())

    return {
        "beta_width_ctrl_mean": float(beta1[1]),
        "beta_width_ctrl_mean_tail": float(beta2[1]),
        "beta_tail_ctrl": float(beta2[3]),
        "resid_ss_mean_only": ss(residual1),
        "resid_ss_mean_tail": ss(residual2),
        "extra_var_explained_by_tail": 1.0 - ss(residual2)/(ss(residual1) + 1e-12),
    }


def bootstrap_mediation(width, align, barrier, n_boot=2000, seed=0):
    width = np.asarray(width)
    align = np.asarray(align, float)
    barrier = np.asarray(barrier, float)

    rng = np.random.default_rng(seed)
    idx_by_width = {
        width_value: np.where(width == width_value)[0]
        for width_value in np.unique(width)
    }

    beta_width = []
    beta_align = []

    for _ in range(n_boot):
        indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in idx_by_width.values()
            ]
        )

        result = mediation_regression(width[indices], align[indices], barrier[indices])

        beta_width.append(result["beta_width_ctrl_align"])
        beta_align.append(result["beta_align_ctrl_width"])

    def ci(values):
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "beta_width_ctrl_align_ci95": ci(beta_width),
        "beta_align_ctrl_width_ci95": ci(beta_align),
        "n_boot": int(n_boot),
    }


def within_width_analysis(width, align, barrier):
    width = np.asarray(width)
    align = np.asarray(align, float)
    barrier = np.asarray(barrier, float)

    per_width = {}
    align_residuals = []
    barrier_residuals = []
    align_nonzero = []
    barrier_nonzero = []

    for width_value in np.unique(width):
        mask = width == width_value

        align_values = align[mask]
        barrier_values = barrier[mask]

        if align_values.std() < 1e-12 or barrier_values.std() < 1e-12:
            per_width[int(width_value)] = None
        else:
            per_width[int(width_value)] = float(np.corrcoef(align_values, barrier_values)[0, 1])

        align_centered = align_values - align_values.mean()
        barrier_centered = barrier_values - barrier_values.mean()

        align_residuals.append(align_centered)
        barrier_residuals.append(barrier_centered)

        if align_values.std() > 1e-12 and barrier_values.std() > 1e-12:
            align_nonzero.append(align_centered)
            barrier_nonzero.append(barrier_centered)

    pooled = float(np.corrcoef(np.concatenate(align_residuals), np.concatenate(barrier_residuals))[0, 1])

    pooled_nonzero = (
        float(np.corrcoef(np.concatenate(align_nonzero), np.concatenate(barrier_nonzero))[0, 1])
        if align_nonzero
        else None
    )

    return {
        "per_width_r": per_width,
        "pooled_within_r_all": pooled,
        "pooled_within_r_nonzero_widths": pooled_nonzero,
    }


def multi_mediation(width, mediators, barrier):
    def standardize(values):
        values = np.asarray(values, float)
        return (values - values.mean())/(values.std() + 1e-9)

    names = list(mediators)

    W = standardize(np.log2(np.asarray(width, float)))
    B = standardize(barrier)

    X = np.column_stack([np.ones_like(W), W] + [standardize(mediators[name]) for name in names])

    beta, *_ = np.linalg.lstsq(X, B, rcond=None)

    n = len(B)
    p = X.shape[1] - 1

    ss_total = float((B ** 2).sum()) + 1e-12
    r2 = 1.0 - float(((B - X @ beta) ** 2).sum())/ss_total

    adj_r2 = None
    if n - p - 1 > 0:
        adj_r2 = 1.0 - (1.0 - r2)*(n - 1)/(n - p - 1)

    output = {"beta_width": float(beta[1]), "r2": r2, "adj_r2": adj_r2, "n": int(n), "p": int(p)}

    for index, name in enumerate(names):
        output["beta_" + name] = float(beta[2 + index])

    return output
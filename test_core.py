import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.barrier import barrier, interpolate
from src.matching import apply_perms, weight_matching
from src.metrics import (alignability, bootstrap_mediation, mediation_regression, within_width_analysis)
from src.models import MLP, get_params, make_net


W = 64
NL = 2


def _params(seed, hidden=(W,)):
    torch.manual_seed(seed)
    return get_params(MLP(list(hidden)))


def test_apply_perms_preserves_function():
    pA = _params(0)
    perm = torch.randperm(W)
    pB = apply_perms(pA, {1: perm}, NL)

    X = torch.rand(32, 784)

    outA = make_net(pA, [W], "cpu")(X)
    outB = make_net(pB, [W], "cpu")(X)

    assert (outA - outB).abs().max().item() < 1e-4


def test_planted_permutation_recovered_exactly():
    pA = _params(1)

    torch.manual_seed(2)
    perm = torch.randperm(W)

    pB = apply_perms(pA, {1: perm}, NL)
    _, pBp = weight_matching(pA, pB, NL)

    for key in pA:
        assert torch.equal(pBp[key], pA[key]), key

    assert alignability(pA, pBp, NL) == pytest.approx(1.0, abs=1e-5)


def test_planted_permutation_two_hidden_layers():
    hidden = (48, 48)
    pA = _params(3, hidden)

    torch.manual_seed(4)
    perms = {1: torch.randperm(48), 2: torch.randperm(48)}

    pB = apply_perms(pA, perms, 3)
    _, pBp = weight_matching(pA, pB, 3)

    for key in pA:
        assert torch.equal(pBp[key], pA[key]), key


def test_matching_self_is_identity():
    pA = _params(5)
    perms, pBp = weight_matching(pA, pA, NL)

    assert torch.equal(perms[1], torch.arange(W))

    for key in pA:
        assert torch.equal(pBp[key], pA[key])


def test_barrier_zero_on_identical_endpoints():
    pA = _params(6)

    X = torch.rand(64, 784)
    y = torch.randint(0, 10, (64,))

    result = barrier(pA, pA, [W], X, y, "cpu", n=5)

    assert result["loss"] == pytest.approx(0.0, abs=1e-9)
    assert result["err"] == pytest.approx(0.0, abs=1e-9)


def test_interpolate_endpoints_are_exact():
    pA = _params(7)
    pB = _params(8)

    for key in pA:
        assert torch.equal(interpolate(pA, pB, 0.0)[key], pA[key])
        assert torch.equal(interpolate(pA, pB, 1.0)[key], pB[key])


def test_alignability_of_self_is_one():
    pA = _params(9)
    assert alignability(pA, pA, NL) == pytest.approx(1.0, abs=1e-6)


def _design(rng, n_per_width=10):
    widths = np.repeat([16, 32, 64, 128, 256, 512, 1024], n_per_width)
    centered = np.log2(widths.astype(float))
    centered -= centered.mean()
    return widths, centered


def test_mediation_recovers_full_mediation():
    rng = np.random.default_rng(0)
    widths, centered = _design(rng)

    align = 0.40 + 0.015 * centered + rng.normal(0, 0.02, len(centered))
    barrier_values = 0.25 - 3.0 * (align - 0.40) + rng.normal(0, 0.03, len(centered))

    result = mediation_regression(widths, align, barrier_values)

    assert result["beta_width_alone"] < -0.4
    assert abs(result["beta_width_ctrl_align"]) < 0.15


def test_mediation_keeps_direct_effect():
    rng = np.random.default_rng(1)
    widths, centered = _design(rng)

    align = 0.45 + rng.normal(0, 0.02, len(centered))
    barrier_values = 0.9 - 0.12 * centered + rng.normal(0, 0.05, len(centered))

    result = mediation_regression(widths, align, barrier_values)

    assert abs(result["beta_width_ctrl_align"] - result["beta_width_alone"]) < 0.10


def test_bootstrap_ci_excludes_zero_for_direct_effect():
    rng = np.random.default_rng(2)
    widths, centered = _design(rng)

    align = 0.45 + rng.normal(0, 0.02, len(centered))
    barrier_values = 0.9 - 0.12 * centered + rng.normal(0, 0.05, len(centered))

    low, high = bootstrap_mediation(
        widths,
        align,
        barrier_values,
        n_boot=200,
        seed=0,
    )["beta_width_ctrl_align_ci95"]

    assert low <= high and np.isfinite([low, high]).all()
    assert high < 0


def test_within_width_handles_constant_barrier_levels():
    rng = np.random.default_rng(3)
    widths, centered = _design(rng)

    align = 0.40 + 0.015 * centered + rng.normal(0, 0.02, len(centered))
    barrier_values = 0.25 - 3.0 * (align - 0.40) + rng.normal(0, 0.03, len(centered))
    barrier_values[widths >= 512] = 0.0

    result = within_width_analysis(widths, align, barrier_values)

    assert result["per_width_r"][512] is None
    assert result["per_width_r"][1024] is None
    assert result["pooled_within_r_nonzero_widths"] < -0.5


def test_alignability_blocks_self_is_one():
    from src.similarity import alignability_blocks

    pA = _params(20)
    align_in, align_out = alignability_blocks(pA, pA, NL)

    assert align_in == pytest.approx(1.0, abs=1e-6)
    assert align_out == pytest.approx(1.0, abs=1e-6)


def test_disagreement_and_cka_are_permutation_invariant():
    from src.similarity import disagreement, linear_cka

    pA = _params(21)

    torch.manual_seed(24)
    pP = apply_perms(pA, {1: torch.randperm(W)}, NL)

    X = torch.rand(64, 784)

    assert disagreement(pA, pP, [W], X, "cpu") == 0.0
    assert linear_cka(pA, pP, [W], X, "cpu") == pytest.approx(1.0, abs=1e-4)

    pB = _params(22)
    value = linear_cka(pA, pB, [W], X, "cpu")

    assert 0.0 <= value <= 1.0 + 1e-6


def test_flatness_proxy_zero_sigma_and_deterministic():
    from src.similarity import flatness_proxy

    pA = _params(23)

    X = torch.rand(64, 784)
    y = torch.randint(0, 10, (64,))

    assert flatness_proxy(pA, [W], X, y, "cpu", sigma=0.0, n_draws=2) == 0.0

    value_a = flatness_proxy(pA, [W], X, y, "cpu", sigma=0.05, n_draws=3, seed=1)
    value_b = flatness_proxy(pA, [W], X, y, "cpu", sigma=0.05, n_draws=3, seed=1)

    assert value_a == value_b


def test_multi_mediation_finds_the_true_mediator():
    from src.metrics import multi_mediation

    rng = np.random.default_rng(5)

    widths = np.repeat([16, 32, 64, 128, 256, 512, 1024], 20)
    centered = np.log2(widths.astype(float))
    centered -= centered.mean()

    noise_candidate = rng.normal(0, 1, len(centered))
    true_candidate = 0.5 * centered + rng.normal(0, 0.3, len(centered))
    barrier_values = -1.0 * true_candidate + rng.normal(0, 0.2, len(centered))

    result = multi_mediation(
        widths,
        {"noise": noise_candidate, "true": true_candidate},
        barrier_values,
    )

    assert abs(result["beta_width"]) < 0.15
    assert result["beta_true"] < -0.6
    assert abs(result["beta_noise"]) < 0.1
    assert result["adj_r2"] is not None and result["adj_r2"] <= result["r2"]


def test_cnn_channel_perms_preserve_function():
    from src.cnn import SmallCNN, apply_channel_perms, get_params as cnn_params, make_cnn

    torch.manual_seed(30)
    channels = 8

    pA = cnn_params(SmallCNN(channels))
    pB = apply_channel_perms(pA, torch.randperm(channels), torch.randperm(channels))

    X = torch.rand(16, 1, 28, 28)

    diff = make_cnn(pA, channels, "cpu")(X) - make_cnn(pB, channels, "cpu")(X)

    assert diff.abs().max().item() < 1e-4


def test_cnn_planted_permutation_recovered_exactly():
    from src.cnn import SmallCNN, apply_channel_perms, get_params as cnn_params
    from src.cnn import weight_matching_cnn

    torch.manual_seed(31)
    channels = 8

    pA = cnn_params(SmallCNN(channels))

    torch.manual_seed(32)
    pB = apply_channel_perms(pA, torch.randperm(channels), torch.randperm(channels))

    _, recovered = weight_matching_cnn(pA, pB)

    for key in pA:
        assert torch.equal(recovered[key], pA[key]), key


def test_cnn_barrier_zero_on_identical_endpoints():
    from src.cnn import SmallCNN, barrier_cnn_full, get_params as cnn_params

    torch.manual_seed(33)
    channels = 8

    pA = cnn_params(SmallCNN(channels))

    X = torch.rand(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))

    result = barrier_cnn_full(pA, pA, channels, X, y, "cpu", n=5)

    assert result["loss"] == pytest.approx(0.0, abs=1e-9)
    assert result["err"] == pytest.approx(0.0, abs=1e-9)


def test_hub_merge_of_permuted_copies_is_lossless():
    from src.merging import hub_merge

    pA = _params(40)
    group = [pA]

    for seed in (41, 42, 43):
        torch.manual_seed(seed)
        group.append(apply_perms(pA, {1: torch.randperm(W)}, NL))

    merged = hub_merge(group, NL)

    for key in pA:
        assert torch.equal(merged[key], pA[key]), key
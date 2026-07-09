import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def save(fig, path):
    dirname = os.path.dirname(path)

    if dirname:
        os.makedirs(dirname, exist_ok=True)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_barrier_curve(res_naive, res_aligned, width, path):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.plot(res_naive["alphas"], res_naive["curve_loss"], "r-o", ms=3, label="naive")
    ax.plot(res_aligned["alphas"], res_aligned["curve_loss"], "g-o", ms=3, label="aligned")

    ax.set_xlabel(r"interpolation $\alpha$")
    ax.set_ylabel("test loss")
    ax.set_title(f"Barrier, width={width}")
    ax.legend()

    save(fig, path)


def fig_width_sweep(widths, naive_m, naive_s,
    aligned_m, aligned_s, path, ylabel="loss barrier", title="Barrier vs width"):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.array(widths)
    ax.errorbar(x, naive_m, yerr=naive_s, fmt="r-o", capsize=3, label="naive")
    ax.errorbar(x, aligned_m, yerr=aligned_s, fmt="g-o", capsize=3, label="aligned")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    save(fig, path)


def fig_mediation(widths, align_m, align_s, align_all, barrier_all, width_all, corr,
    path, barrier_label="aligned loss barrier"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.array(widths)

    ax[0].errorbar(x, align_m, yerr=align_s, fmt="b-o", capsize=3)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("hidden width")
    ax[0].set_ylabel("alignability")
    ax[0].set_title("(a) alignability vs width")

    scatter = ax[1].scatter(align_all, barrier_all, c=np.log2(width_all), cmap="viridis")
    ax[1].set_xlabel("alignability")
    ax[1].set_ylabel(barrier_label)
    ax[1].set_title(f"(b) barrier vs alignability (r={corr:.2f})")

    fig.colorbar(scatter, ax=ax[1], label="log2 width")

    save(fig, path)


def fig_lmc_over_training(steps, loss_m, loss_s, err_m, err_s, endloss_m, path):
    steps = np.array(steps)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax[0].errorbar(steps, loss_m, yerr=loss_s, fmt="g-o", capsize=3, label="loss barrier")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("training step (log)")
    ax[0].set_ylabel("aligned loss barrier", color="g")
    ax[0].tick_params(axis="y", labelcolor="g")

    ax0b = ax[0].twinx()
    ax0b.plot(steps, endloss_m, "k--o", ms=3, alpha=0.5)
    ax0b.set_ylabel("endpoint test loss", color="k")

    ax[0].set_title("(a) loss barrier")

    ax[1].errorbar(steps, err_m, yerr=err_s, fmt="b-o", capsize=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training step (log)")
    ax[1].set_ylabel("aligned error barrier")
    ax[1].set_title("(b) error barrier")

    fig.tight_layout()
    save(fig, path)


def fig_null_alignability(widths, trained_m, trained_s, null_m, null_s, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.array(widths)

    ax[0].errorbar(x, trained_m, yerr=trained_s, fmt="b-o", capsize=3, label="trained")
    ax[0].errorbar(x, null_m, yerr=null_s, fmt="-o", color="gray", capsize=3, label="untrained")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("hidden width")
    ax[0].set_ylabel("alignability")
    ax[0].set_title("(a) trained vs untrained")
    ax[0].legend()

    gap = np.asarray(trained_m) - np.asarray(null_m)
    gap_std = np.sqrt(np.asarray(trained_s) ** 2 + np.asarray(null_s) ** 2)

    ax[1].errorbar(x, gap, yerr=gap_std, fmt="k-o", capsize=3)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("hidden width")
    ax[1].set_ylabel("trained \u2212 untrained")
    ax[1].set_title("(b) alignability gap")

    fig.tight_layout()
    save(fig, path)


def fig_lmc_multi_width(series, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    for width, steps, err_mean, err_std in series:
        ax.errorbar(steps, err_mean, yerr=err_std, fmt="-o", ms=3, capsize=3, label=f"w={width}")

    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("aligned error barrier")
    ax.set_title("Error barrier over training")
    ax.legend()

    fig.tight_layout()
    save(fig, path)


def fig_cnn_channels(rows, path):
    channels = np.array([row["channels"] for row in rows], float)
    errors = np.array([row["aligned_err"] for row in rows], float)
    align = np.array([row["align"] for row in rows], float)

    rng = np.random.default_rng(0)
    x = channels * (2.0 ** rng.uniform(-0.07, 0.07, len(channels)))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    scatter = ax.scatter(x, errors, c=align, cmap="viridis", s=40, edgecolor="k", linewidth=0.4)

    for channels_count in sorted(set(channels.tolist())):
        mean_error = errors[channels == channels_count].mean()
        ax.plot([channels_count / 1.25, channels_count * 1.25], [mean_error, mean_error], "k-", lw=1.6)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("channels")
    ax.set_ylabel("aligned error barrier")
    ax.set_title("CNN: barriers by channel count")

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("alignability")

    fig.tight_layout()
    save(fig, path)


def fig_residual_candidates(labels, delta_adj_r2, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(labels))

    ax.barh(y, delta_adj_r2, color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("adj. R\u00b2 gain over width + align")
    ax.set_title("Candidate mediators")

    fig.tight_layout()
    save(fig, path)


def fig_kmerge(widths, series, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.array(widths)

    for k, mean, std in series:
        ax.errorbar(x, mean, yerr=std, fmt="-o", ms=4, capsize=3, label=f"k={k}")

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("merge penalty (error)")
    ax.set_title("Merging penalty vs width")
    ax.legend()

    fig.tight_layout()
    save(fig, path)
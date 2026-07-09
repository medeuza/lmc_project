import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from src.cnn import (alignability_cnn, barrier_cnn_full, get_img_data,
    train_cnn, weight_matching_cnn)
from src.common import load_cfg
from src.plotting import fig_cnn_channels


CHANNELS = [8, 16, 32, 64, 128]
TRAIN_SUBSET = 12000
STATS_N = 512


def main():
    cfg = load_cfg()
    channels = CHANNELS if not cfg.get("quick_test") else [8, 16]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    Xtr, ytr, Xev, yev = get_img_data(cfg["dataset"], cfg["eval_n"], device)

    subset = torch.randperm(len(Xtr))[:TRAIN_SUBSET]
    Xtr, ytr = Xtr[subset], ytr[subset]

    cfg_run = dict(cfg)
    cfg_run["dataset"] = f"{cfg['dataset']}sub{TRAIN_SUBSET // 1000}k"

    Xstats = Xev[:STATS_N]
    rows = []

    for channels_count in channels:
        print(
            f"--- channels={channels_count} "
            f"(train n={len(Xtr)}, tag dataset={cfg_run['dataset']}) ---"
        )

        for pair_id in range(cfg["k_seeds"]):
            params_a = train_cnn(channels_count, 700 + 2 * pair_id, cfg_run, Xtr, ytr, device)
            params_b = train_cnn(channels_count, 701 + 2 * pair_id, cfg_run, Xtr, ytr, device)

            naive_result = barrier_cnn_full(
                params_a,
                params_b,
                channels_count,
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
                repair=False,
            )

            _, aligned_params_b = weight_matching_cnn(params_a, params_b)

            aligned_result = barrier_cnn_full(
                params_a,
                aligned_params_b,
                channels_count,
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
                repair=False,
            )

            repair_result = barrier_cnn_full(
                params_a,
                aligned_params_b,
                channels_count,
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
                repair=True,
                stats_X=Xstats,
            )

            rows.append(
                {
                    "channels": channels_count,
                    "pair": pair_id,
                    "train_subset": TRAIN_SUBSET,
                    "naive_loss": naive_result["loss"],
                    "naive_err": naive_result["err"],
                    "aligned_loss": aligned_result["loss"],
                    "aligned_err": aligned_result["err"],
                    "repair_loss": repair_result["loss"],
                    "repair_err": repair_result["err"],
                    "align": alignability_cnn(params_a, aligned_params_b),
                }
            )

            row = rows[-1]
            print(
                f"c={channels_count:3d} k={pair_id}: "
                f"naive_err={row['naive_err']:.3f} "
                f"aligned_err={row['aligned_err']:.3f} "
                f"repair_err={row['repair_err']:.3f} | "
                f"aligned_loss={row['aligned_loss']:.2f} "
                f"repair_loss={row['repair_loss']:.2f} "
                f"align={row['align']:.3f}"
            )

        output_path = os.path.join(cfg["res_dir"], "exp8_cnn.json")
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)

    def mean_by_channels(key):
        return {
            channels_count: float(
                np.mean([row[key] for row in rows if row["channels"] == channels_count])
            )
            for channels_count in channels
        }

    print(
        "\n[CNN] aligned error barrier by channels:",
        {key: round(value, 3) for key, value in mean_by_channels("aligned_err").items()},
    )
    print(
        "[CNN] aligned+REPAIR error barrier by channels:",
        {key: round(value, 3) for key, value in mean_by_channels("repair_err").items()},
    )
    print(
        "[CNN] aligned+REPAIR loss  barrier by channels:",
        {key: round(value, 3) for key, value in mean_by_channels("repair_loss").items()},
    )
    print(
        "[CNN] alignability by channels:",
        {key: round(value, 3) for key, value in mean_by_channels("align").items()},
    )

    fig_cnn_channels(rows, os.path.join(cfg["fig_dir"], "fig8_cnn_channels.png"))


if __name__ == "__main__":
    main()
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.barrier import eval_params
from src.common import load_cfg, setup
from src.merging import hub_merge
from src.models import make_net
from src.plotting import fig_kmerge
from src.train import train_net


KS = [2, 4, 8]
N_NETS = 10


@torch.no_grad()
def ensemble_err(params_list, hidden, X, y, device):
    probs = sum(
        F.softmax(make_net(params, hidden, device)(X), dim=1)
        for params in params_list)/len(params_list)

    return float((probs.argmax(1) != y).float().mean().item())


def main():
    cfg = load_cfg()
    device, tl, Xev, yev = setup(cfg)

    num_layers = 2
    rows = []

    for width in cfg["widths"]:
        nets = [
            train_net([width], 100 + seed, cfg, tl, device)[0]
            for seed in range(N_NETS)
        ]

        for k in KS:
            for rep in range(N_NETS // k):
                group = nets[rep * k : (rep + 1) * k]

                single_errors = [
                    eval_params(params, [width], Xev, yev, device)[1]
                    for params in group
                ]

                merged_params = hub_merge(group, num_layers)

                naive_params = {
                    key: sum(params[key] for params in group) / k
                    for key in group[0]
                }

                merged_error = eval_params(merged_params, [width], Xev, yev, device)[1]
                naive_error = eval_params(naive_params, [width], Xev, yev, device)[1]
                ensemble_error = ensemble_err(group, [width], Xev, yev, device)

                mean_single_error = float(np.mean(single_errors))

                rows.append(
                    {
                        "width": width,
                        "k": k,
                        "rep": rep,
                        "err_single": mean_single_error,
                        "err_merged": merged_error,
                        "err_merged_naive": naive_error,
                        "err_ensemble": ensemble_error,
                        "merge_penalty": merged_error - mean_single_error,
                    }
                )

                row = rows[-1]
                print(
                    f"w={width:5d} k={k} rep={rep}: "
                    f"single={row['err_single']:.3f} "
                    f"merged={row['err_merged']:.3f} "
                    f"(naive {row['err_merged_naive']:.3f}) "
                    f"ensemble={row['err_ensemble']:.3f} "
                    f"penalty={row['merge_penalty']:+.3f}"
                )

    output_path = os.path.join(cfg["res_dir"], "exp11_kmerge.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    series = []

    for k in KS:
        means = []
        stds = []

        for width in cfg["widths"]:
            values = np.array(
                [
                    row["merge_penalty"]
                    for row in rows
                    if row["width"] == width and row["k"] == k
                ]
            )

            means.append(values.mean())
            stds.append(values.std())

        series.append((k, np.array(means), np.array(stds)))

    fig_kmerge(
        cfg["widths"],
        series,
        os.path.join(cfg["fig_dir"], "fig11_kmerge.png"),
    )

    print("\nEXP11 merge penalty (err_merged - err_single) by width:")

    for k, means, stds in series:
        summary = "  ".join(
            f"{width}:{mean:+.3f}"
            for width, mean in zip(cfg["widths"], means)
        )
        print(f"  k={k} (n={N_NETS // k}): {summary}")


if __name__ == "__main__":
    main()
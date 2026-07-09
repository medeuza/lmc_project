import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.barrier import barrier, eval_params
from src.common import load_cfg, setup
from src.matching import weight_matching
from src.plotting import fig_lmc_multi_width, fig_lmc_over_training
from src.train import train_net


UNTRAINED_LOSS = 2.0
WIDTHS = [16, 64, 256]
STEPS = [10, 20, 50, 100, 300, 1000, 3000]


def main():
    cfg = load_cfg()
    device, tl, Xev, yev = setup(cfg)

    num_layers = 2
    num_pairs = cfg["k_seeds"]

    rows = []
    series = []

    for width in WIDTHS:
        accum = {step: {"loss_b": [], "err_b": [], "endloss": []} for step in STEPS}

        for pair_id in range(num_pairs):
            _, snapshots_a = train_net(
                [width],
                500 + 2 * pair_id,
                cfg,
                tl,
                device,
                step_snapshots=STEPS,
            )
            _, snapshots_b = train_net(
                [width],
                501 + 2 * pair_id,
                cfg,
                tl,
                device,
                step_snapshots=STEPS,
            )

            for step in STEPS:
                key = f"step{step}"

                if key not in snapshots_a or key not in snapshots_b:
                    continue

                _, aligned_params_b = weight_matching(
                    snapshots_a[key],
                    snapshots_b[key],
                    num_layers,
                )

                result = barrier(
                    snapshots_a[key],
                    aligned_params_b,
                    [width],
                    Xev,
                    yev,
                    device,
                    cfg["alpha_pts"],
                )

                endpoint_loss, _ = eval_params(
                    snapshots_a[key],
                    [width],
                    Xev,
                    yev,
                    device,
                )

                accum[step]["loss_b"].append(result["loss"])
                accum[step]["err_b"].append(result["err"])
                accum[step]["endloss"].append(endpoint_loss)

                print(
                    f"w={width:4d} pair={pair_id} step={step:4d}: "
                    f"loss_barrier={result['loss']:.3f} "
                    f"err_barrier={result['err']:.3f} "
                    f"endloss={endpoint_loss:.3f}"
                )

        trained_steps = []
        loss_mean = []
        loss_std = []
        err_mean = []
        err_std = []
        endpoint_mean = []

        for step in STEPS:
            data = accum[step]

            if not data["loss_b"]:
                continue

            mean_endpoint_loss = float(np.mean(data["endloss"]))

            record = {
                "width": width,
                "step": step,
                "loss_barrier_mean": float(np.mean(data["loss_b"])),
                "loss_barrier_std": float(np.std(data["loss_b"])),
                "err_barrier_mean": float(np.mean(data["err_b"])),
                "err_barrier_std": float(np.std(data["err_b"])),
                "endpoint_loss_mean": mean_endpoint_loss,
                "n_pairs": len(data["loss_b"]),
                "trained": mean_endpoint_loss < UNTRAINED_LOSS,
            }

            rows.append(record)

            if record["trained"]:
                trained_steps.append(step)
                loss_mean.append(record["loss_barrier_mean"])
                loss_std.append(record["loss_barrier_std"])
                err_mean.append(record["err_barrier_mean"])
                err_std.append(record["err_barrier_std"])
                endpoint_mean.append(mean_endpoint_loss)

        if trained_steps:
            series.append((width, trained_steps, err_mean, err_std))

            if width == 16:
                fig_lmc_over_training(
                    trained_steps,
                    loss_mean,
                    loss_std,
                    err_mean,
                    err_std,
                    endpoint_mean,
                    os.path.join(cfg["fig_dir"], "fig5_lmc_over_training.png"),
                )

    output_path = os.path.join(cfg["res_dir"], "exp5.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    if series:
        fig_lmc_multi_width(series, os.path.join(cfg["fig_dir"], "fig5b_lmc_widths.png"))

    print("\nEXP5 (trained phase, error barrier by step):")

    for width in WIDTHS:
        width_rows = [row for row in rows if row["width"] == width and row["trained"]]

        if width_rows:
            summary = "  ".join(
                f"{row['step']}:{row['err_barrier_mean']:.3f}"
                for row in width_rows
            )
            print(f"  w={width:4d}: {summary}")


if __name__ == "__main__":
    main()
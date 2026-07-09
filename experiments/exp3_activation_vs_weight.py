import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.barrier import barrier
from src.common import load_cfg, setup, trained_pair
from src.matching import (activation_matching, apply_perms, weight_matching)
from src.models import make_net


def main():
    cfg = load_cfg()
    device, tl, Xev, yev = setup(cfg)

    num_layers = 2
    rows = []

    for width in cfg["widths"]:
        for pair_id in range(cfg["k_seeds"]):
            (params_a, _), (params_b, _) = trained_pair([width], pair_id, cfg, tl, device)

            _, weight_aligned_params = weight_matching(params_a, params_b, num_layers)
            weight_barrier = barrier(
                params_a,
                weight_aligned_params,
                [width],
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )["loss"]

            model_a = make_net(params_a, [width], device)
            model_b = make_net(params_b, [width], device)

            activation_perms = activation_matching(model_a, model_b, Xev, num_layers)
            activation_aligned_params = apply_perms(params_b, activation_perms, num_layers)

            activation_barrier = barrier(
                params_a,
                activation_aligned_params,
                [width],
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )["loss"]

            rows.append({
                    "width": width,
                    "pair": pair_id,
                    "weight_match": weight_barrier,
                    "activation_match": activation_barrier,
                }
            )

            print(
                f"w={width:5d} k={pair_id}: "
                f"weight={weight_barrier:.3f} "
                f"activation={activation_barrier:.3f}")

    output_path = os.path.join(cfg["res_dir"], "exp3.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    for width in cfg["widths"]:
        width_rows = [row for row in rows if row["width"] == width]

        print(
            f"w={width}: "
            f"weight={np.mean([row['weight_match'] for row in width_rows]):.3f} "
            f"activation={np.mean([row['activation_match'] for row in width_rows]):.3f}")


if __name__ == "__main__":
    main()
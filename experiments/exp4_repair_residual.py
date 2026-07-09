import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.barrier import barrier, barrier_repair
from src.common import load_cfg, setup, trained_pair
from src.matching import weight_matching
from src.metrics import assignment_cost


def main():
    cfg = load_cfg()
    device, tl, Xev, yev = setup(cfg)

    num_layers = 2
    rows = []

    for width in cfg["widths"]:
        for pair_id in range(cfg["k_seeds"]):
            (params_a, _), (params_b, _) = trained_pair([width], pair_id, cfg, tl, device)

            _, aligned_params_b = weight_matching(params_a, params_b, num_layers)

            aligned_barrier = barrier(
                params_a,
                aligned_params_b,
                [width],
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )["loss"]

            aligned_repair = barrier_repair(
                params_a,
                aligned_params_b,
                [width],
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )

            unalignability = assignment_cost(params_a, aligned_params_b, num_layers)

            naive_barrier = barrier(
                params_a,
                params_b,
                [width],
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )["loss"]

            naive_repair = barrier_repair(
                params_a,
                params_b,
                [width],
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )

            rows.append(
                {
                    "width": width,
                    "pair": pair_id,
                    "aligned": aligned_barrier,
                    "aligned_repair": aligned_repair,
                    "repair_gain": aligned_barrier - aligned_repair,
                    "unalignability": unalignability,
                    "naive": naive_barrier,
                    "naive_repair": naive_repair,
                    "repair_gain_naive": naive_barrier - naive_repair,
                }
            )

            row = rows[-1]
            print(
                f"w={width:5d} k={pair_id}: "
                f"aligned={row['aligned']:.3f} +REPAIR={row['aligned_repair']:.3f} | "
                f"naive={row['naive']:.3f} +REPAIR={row['naive_repair']:.3f} | "
                f"unalign={row['unalignability']:.3f}"
            )

    output_path = os.path.join(cfg["res_dir"], "exp4.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nEXP4 by width:")

    for width in cfg["widths"]:
        width_rows = [row for row in rows if row["width"] == width]

        print(
            f"  w={width:5d}: "
            f"repair_gain(aligned)="
            f"{np.mean([row['repair_gain'] for row in width_rows]):+.3f}  "
            f"naive={np.mean([row['naive'] for row in width_rows]):.3f} => "
            f"+REPAIR={np.mean([row['naive_repair'] for row in width_rows]):.3f}  "
            f"(gain {np.mean([row['repair_gain_naive'] for row in width_rows]):+.3f})  "
            f"unalign={np.mean([row['unalignability'] for row in width_rows]):.3f}"
        )


if __name__ == "__main__":
    main()
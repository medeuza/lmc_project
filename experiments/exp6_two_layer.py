import json
import os
import sys

sys.path.insert(0, ".")

from src.barrier import barrier
from src.common import load_cfg, setup, trained_pair
from src.matching import weight_matching
from src.metrics import alignability


def main():
    cfg = load_cfg()
    device, tl, Xev, yev = setup(cfg)

    num_layers = 3
    rows = []

    for width in cfg["two_layer_widths"]:
        hidden = [width, width]

        for pair_id in range(cfg["k_seeds"]):
            (params_a, _), (params_b, _) = trained_pair(hidden, pair_id, cfg, tl, device)

            naive_barrier = barrier(
                params_a,
                params_b,
                hidden,
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )["loss"]

            _, aligned_params_b = weight_matching(params_a, params_b, num_layers)

            aligned_barrier = barrier(
                params_a,
                aligned_params_b,
                hidden,
                Xev,
                yev,
                device,
                cfg["alpha_pts"],
            )["loss"]

            align_score = alignability(params_a, aligned_params_b, num_layers)

            rows.append(
                {
                    "width": width,
                    "pair": pair_id,
                    "naive": naive_barrier,
                    "aligned": aligned_barrier,
                    "align": align_score,
                }
            )

            print(
                f"w={width}x2 k={pair_id}: "
                f"naive={naive_barrier:.3f} "
                f"aligned={aligned_barrier:.3f} "
                f"align={align_score:.3f}"
            )

    output_path = os.path.join(cfg["res_dir"], "exp6.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
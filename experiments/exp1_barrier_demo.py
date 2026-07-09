import json
import os
import sys

sys.path.insert(0, ".")

from src.barrier import barrier
from src.common import load_cfg, setup, trained_pair
from src.matching import weight_matching
from src.metrics import alignability
from src.plotting import fig_barrier_curve


def main():
    cfg = load_cfg()
    dev, tl, Xev, yev = setup(cfg)

    width = cfg["widths"][-1]
    hidden = [width]
    num_layers = 2

    (params_a, _), (params_b, _) = trained_pair(hidden, 0, cfg, tl, dev)

    naive_result = barrier(params_a, params_b, hidden, Xev, yev, dev, cfg["alpha_pts"])

    _, aligned_params_b = weight_matching(params_a, params_b, num_layers)
    aligned_result = barrier(params_a, aligned_params_b, hidden, Xev, yev, dev, cfg["alpha_pts"])

    fig_path = os.path.join(cfg["fig_dir"], "fig1_barrier.png")
    fig_barrier_curve(naive_result, aligned_result, width, fig_path)

    output = {
        "width": width,
        "naive": naive_result["loss"],
        "aligned": aligned_result["loss"],
        "align_before": alignability(params_a, params_b, num_layers),
        "align_after": alignability(params_a, aligned_params_b, num_layers),
    }

    out_path = os.path.join(cfg["res_dir"], "exp1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("EXP1", output)


if __name__ == "__main__":
    main()
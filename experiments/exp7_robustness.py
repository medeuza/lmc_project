import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.barrier import barrier
from src.common import load_cfg, setup, trained_pair
from src.matching import weight_matching
from src.metrics import (alignability, bootstrap_mediation, mediation_regression, within_width_analysis,)


ALT_DATASET = "MNIST"
NONZERO_WIDTHS_MAX = 128


def main():
    cfg = load_cfg()
    cfg["dataset"] = ALT_DATASET

    device, tl, Xev, yev = setup(cfg)
    num_layers = 2
    rows = []

    for width in cfg["widths"]:
        for pair_id in range(cfg["k_seeds"]):
            (params_a, _), (params_b, _) = trained_pair([width], pair_id, cfg, tl, device)

            naive_result = barrier(params_a, params_b, [width], Xev, yev, device, cfg["alpha_pts"])

            _, aligned_params_b = weight_matching(params_a, params_b, num_layers)
            aligned_result = barrier(
                params_a,
                aligned_params_b,
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
                    "naive": naive_result["loss"],
                    "aligned": aligned_result["loss"],
                    "naive_err": naive_result["err"],
                    "aligned_err": aligned_result["err"],
                    "align": alignability(params_a, aligned_params_b, num_layers),
                }
            )

            row = rows[-1]
            print(
                f"[{ALT_DATASET}] w={width:5d} k={pair_id}: "
                f"loss {row['naive']:.3f}->{row['aligned']:.3f} | "
                f"err {row['naive_err']:.3f}->{row['aligned_err']:.3f} | "
                f"align={row['align']:.3f}"
            )

    output_path = os.path.join(cfg["res_dir"], "exp7_mnist.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    widths = [row["width"] for row in rows]
    align_values = [row["align"] for row in rows]

    def run_mediation(metric_key):
        barrier_values = [row[metric_key] for row in rows]

        output = mediation_regression(widths, align_values, barrier_values)
        output["corr_align_barrier"] = float(np.corrcoef(align_values, barrier_values)[0, 1])
        output["bootstrap"] = bootstrap_mediation(widths, align_values, barrier_values)
        output["within_width"] = within_width_analysis(widths, align_values, barrier_values)

        subset = [i for i, width in enumerate(widths) if width <= NONZERO_WIDTHS_MAX]

        if len({widths[i] for i in subset}) >= 3:
            subset_widths = np.array(widths)[subset]
            subset_align = np.array(align_values)[subset]
            subset_barrier = np.array(barrier_values)[subset]

            output["subset_nonzero_widths"] = mediation_regression(
                subset_widths,
                subset_align,
                subset_barrier,
            )
            output["subset_nonzero_widths"]["bootstrap"] = bootstrap_mediation(
                subset_widths,
                subset_align,
                subset_barrier,
            )
            output["subset_nonzero_widths"]["widths_used"] = sorted(
                {int(width) for width in subset_widths}
            )

        return output

    mediation_results = {
        "loss": run_mediation("aligned"),
        "err": run_mediation("aligned_err"),
    }

    mediation_path = os.path.join(cfg["res_dir"], "exp7_mnist_mediation.json")
    with open(mediation_path, "w") as f:
        json.dump(mediation_results, f, indent=2)

    def mean_by_width(key):
        return {
            width: float(np.mean([row[key] for row in rows if row["width"] == width]))
            for width in cfg["widths"]
        }

    print(
        f"\n[{ALT_DATASET}] aligned err barrier by width:",
        {key: round(value, 3) for key, value in mean_by_width("aligned_err").items()},
    )
    print(
        f"[{ALT_DATASET}] alignability by width:",
        {key: round(value, 3) for key, value in mean_by_width("align").items()},
    )

    for name in ("loss", "err"):
        result = mediation_results[name]
        bootstrap = result["bootstrap"]

        print(
            f"\n[{ALT_DATASET}] EXP7 [{name}] "
            f"corr(align, barrier) = {result['corr_align_barrier']:+.3f}"
        )
        print(f"  beta_width alone      = {result['beta_width_alone']:+.3f}")
        print(
            f"  beta_width ctrl align = {result['beta_width_ctrl_align']:+.3f}  "
            f"CI95 {np.round(bootstrap['beta_width_ctrl_align_ci95'], 3).tolist()}"
        )

        if "subset_nonzero_widths" in result:
            subset = result["subset_nonzero_widths"]
            print(
                f"  subset w<={NONZERO_WIDTHS_MAX}: beta_width ctrl align = "
                f"{subset['beta_width_ctrl_align']:+.3f}  "
                f"CI95 {np.round(subset['bootstrap']['beta_width_ctrl_align_ci95'], 3).tolist()}"
            )


if __name__ == "__main__":
    main()
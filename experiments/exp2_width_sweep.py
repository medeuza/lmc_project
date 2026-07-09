import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.barrier import barrier
from src.common import load_cfg, setup, trained_pair
from src.matching import weight_matching
from src.metrics import (alignability, bootstrap_mediation, mediation_regression, mediation_two_mediators, tail_alignability, within_width_analysis)
from src.plotting import fig_mediation, fig_width_sweep


NONZERO_WIDTHS_MAX = 128


def main():
    cfg = load_cfg()
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

            rows.append({
                    "width": width,
                    "pair": pair_id,
                    "naive": naive_result["loss"],
                    "aligned": aligned_result["loss"],
                    "naive_err": naive_result["err"],
                    "aligned_err": aligned_result["err"],
                    "align": alignability(params_a, aligned_params_b, num_layers),
                    "tail_align": tail_alignability(
                        params_a,
                        aligned_params_b,
                        num_layers,
                        q=0.10,
                    ),
                }
            )

            row = rows[-1]
            print(
                f"w={width:5d} k={pair_id}: "
                f"loss {row['naive']:.3f}->{row['aligned']:.3f} | "
                f"err {row['naive_err']:.3f}->{row['aligned_err']:.3f} | "
                f"align={row['align']:.3f}")

    results_path = os.path.join(cfg["res_dir"], "exp2.json")
    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)

    def aggregate(key):
        means = []
        stds = []

        for width in cfg["widths"]:
            values = np.array([row[key] for row in rows if row["width"] == width])
            means.append(values.mean())
            stds.append(values.std())

        return np.array(means), np.array(stds)

    naive_mean, naive_std = aggregate("naive")
    aligned_mean, aligned_std = aggregate("aligned")

    fig_width_sweep(
        cfg["widths"],
        naive_mean,
        naive_std,
        aligned_mean,
        aligned_std,
        os.path.join(cfg["fig_dir"], "fig2_width.png"))

    naive_err_mean, naive_err_std = aggregate("naive_err")
    aligned_err_mean, aligned_err_std = aggregate("aligned_err")

    fig_width_sweep(
        cfg["widths"],
        naive_err_mean,
        naive_err_std,
        aligned_err_mean,
        aligned_err_std,
        os.path.join(cfg["fig_dir"], "fig2b_width_err.png"),
        ylabel="error barrier",
        title="Error barrier vs width")

    align_mean, align_std = aggregate("align")

    widths = [row["width"] for row in rows]
    align_values = [row["align"] for row in rows]
    tail_align_values = [row["tail_align"] for row in rows]

    def run_mediation(metric_key, fig_path, barrier_label):
        barrier_values = [row[metric_key] for row in rows]
        corr = float(np.corrcoef(align_values, barrier_values)[0, 1])

        fig_mediation(
            cfg["widths"],
            align_mean,
            align_std,
            align_values,
            barrier_values,
            widths,
            corr,
            fig_path,
            barrier_label=barrier_label)

        output = mediation_regression(widths, align_values, barrier_values)
        output["corr_align_barrier"] = corr
        output.update(mediation_two_mediators(widths, align_values, tail_align_values,barrier_values))
        output["bootstrap"] = bootstrap_mediation(widths, align_values, barrier_values)
        output["within_width"] = within_width_analysis(widths, align_values, barrier_values)

        subset = [i for i, width in enumerate(widths) if width <= NONZERO_WIDTHS_MAX]

        if len({widths[i] for i in subset}) >= 3:
            subset_widths = np.array(widths)[subset]
            subset_align = np.array(align_values)[subset]
            subset_barrier = np.array(barrier_values)[subset]

            output["subset_nonzero_widths"] = mediation_regression(subset_widths, subset_align, subset_barrier)
            output["subset_nonzero_widths"]["bootstrap"] = bootstrap_mediation(subset_widths, subset_align, subset_barrier)
            output["subset_nonzero_widths"]["widths_used"] = sorted({int(width) for width in subset_widths})

        return output

    mediation_results = {
        "loss": run_mediation(
            "aligned",
            os.path.join(cfg["fig_dir"], "fig3_mediation.png"),
            "aligned loss barrier"),
        "err": run_mediation(
            "aligned_err",
            os.path.join(cfg["fig_dir"], "fig3b_mediation_err.png"),
            "aligned error barrier"),
    }

    mediation_path = os.path.join(cfg["res_dir"], "exp2_mediation.json")
    with open(mediation_path, "w") as f:
        json.dump(mediation_results, f, indent=2)

    for name in ("loss", "err"):
        result = mediation_results[name]
        bootstrap = result["bootstrap"]

        print(f"\nEXP2 [{name}] corr(align, barrier) = {result['corr_align_barrier']:+.3f}")
        print(
            f"  beta_width alone      = {result['beta_width_alone']:+.3f}  "
            f"(R^2 {result['r2_width_alone']:.3f})")
        print(
            f"  beta_width ctrl align = {result['beta_width_ctrl_align']:+.3f}  "
            f"CI95 {np.round(bootstrap['beta_width_ctrl_align_ci95'], 3).tolist()}  "
            f"(R^2 {result['r2_width_align']:.3f})")
        print(
            f"  beta_align ctrl width = {result['beta_align_ctrl_width']:+.3f}  "
            f"CI95 {np.round(bootstrap['beta_align_ctrl_width_ci95'], 3).tolist()}")

        within_width = result["within_width"]
        print(
            f"  within-width pooled r = {within_width['pooled_within_r_all']:+.3f} "
            f"(nonzero widths only: {within_width['pooled_within_r_nonzero_widths']})")

        if "subset_nonzero_widths" in result:
            subset = result["subset_nonzero_widths"]
            print(
                f"  subset w<={NONZERO_WIDTHS_MAX}: beta_width ctrl align = "
                f"{subset['beta_width_ctrl_align']:+.3f}  "
                f"CI95 {np.round(subset['bootstrap']['beta_width_ctrl_align_ci95'], 3).tolist()}")


if __name__ == "__main__":
    main()
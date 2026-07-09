import json
import os
import sys
import numpy as np
sys.path.insert(0, ".")
from src.common import load_cfg
from src.matching import weight_matching
from src.metrics import alignability, tail_alignability
from src.models import MLP, get_params
from src.plotting import fig_null_alignability
from src.train import set_seed


N_LAYERS = 2


def untrained_params(hidden, seed):
    set_seed(seed)
    return get_params(MLP(hidden))


def main():
    cfg = load_cfg()
    rows = []

    for width in cfg["widths"]:
        for pair_id in range(cfg["k_seeds"]):
            params_a = untrained_params([width], 100 + 2 * pair_id)
            params_b = untrained_params([width], 101 + 2 * pair_id)

            align_pre = alignability(params_a, params_b, N_LAYERS)

            _, matched_params_b = weight_matching(params_a, params_b, N_LAYERS)

            rows.append(
                {
                    "width": width,
                    "pair": pair_id,
                    "align_null": alignability(params_a, matched_params_b, N_LAYERS),
                    "align_pre": align_pre,
                    "tail_null": tail_alignability(params_a, matched_params_b, N_LAYERS, q=0.10),
                }
            )

            row = rows[-1]
            print(
                f"w={width:5d} k={pair_id}: "
                f"null={row['align_null']:.4f} "
                f"(pre-match {row['align_pre']:+.4f})")

    output_path = os.path.join(cfg["res_dir"], "exp9_null.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    exp2_path = os.path.join(cfg["res_dir"], "exp2.json")

    if not os.path.exists(exp2_path):
        print("\nresults/exp2.json not found")
        return

    with open(exp2_path) as f:
        trained = json.load(f)

    widths = [width for width in cfg["widths"] if any(row["width"] == width for row in trained)]

    if not widths:
        print("\nresults/exp2.json no rows")
        return

    if widths != list(cfg["widths"]):
        print(f"\ntrained overlay restricted to widths {widths} (exp2.json coverage)")

    def curve(records, key):
        means = []
        stds = []

        for width in widths:
            values = np.array([row[key] for row in records if row["width"] == width])
            means.append(values.mean())
            stds.append(values.std())

        return np.array(means), np.array(stds)

    null_mean, null_std = curve(rows, "align_null")
    trained_mean, trained_std = curve(trained, "align")
    gap = trained_mean - null_mean

    log_widths = np.log2(np.array(widths, float))

    def slope(values):
        return float(np.polyfit(log_widths, values, 1)[0])

    summary = {
        "widths": widths,
        "null_mean": null_mean.tolist(),
        "null_std": null_std.tolist(),
        "trained_mean": trained_mean.tolist(),
        "trained_std": trained_std.tolist(),
        "gap": gap.tolist(),
        "slope_trained_per_doubling": slope(trained_mean),
        "slope_null_per_doubling": slope(null_mean),
        "slope_gap_per_doubling": slope(gap),
        "n_pairs_per_width": int(cfg["k_seeds"]),
    }

    summary_path = os.path.join(cfg["res_dir"], "exp9_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    fig_null_alignability(
        widths,
        trained_mean,
        trained_std,
        null_mean,
        null_std,
        os.path.join(cfg["fig_dir"], "fig9_null_alignability.png"),
    )

    print("\nEXP9 summary")
    print("width  trained  null   gap")

    for index, width in enumerate(widths):
        print(
            f"{width:5d}  "
            f"{trained_mean[index]:.3f}   "
            f"{null_mean[index]:.3f}  "
            f"{gap[index]:.3f}")

    print(
        f"slope per doubling: "
        f"trained={summary['slope_trained_per_doubling']:+.4f}  "
        f"null={summary['slope_null_per_doubling']:+.4f}  "
        f"gap={summary['slope_gap_per_doubling']:+.4f}")


if __name__ == "__main__":
    main()
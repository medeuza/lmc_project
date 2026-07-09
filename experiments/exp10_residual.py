import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.common import load_cfg, setup, trained_pair
from src.matching import weight_matching
from src.metrics import multi_mediation, within_width_analysis
from src.models import MLP, get_params
from src.plotting import fig_residual_candidates
from src.similarity import (alignability_blocks, disagreement, flatness_proxy, linear_cka,)
from src.train import set_seed


NONZERO_WIDTHS_MAX = 128
CANDIDATE_NAMES = [
    "align_in",
    "align_out",
    "tail_align",
    "disagreement",
    "cka",
    "flatness",
    "flatness_excess",
]


def main():
    cfg = load_cfg()
    exp2_path = os.path.join(cfg["res_dir"], "exp2.json")

    if not os.path.exists(exp2_path):
        print("results/exp2.json not found")
        return

    with open(exp2_path) as f:
        base_rows = json.load(f)

    if "aligned_err" not in base_rows[0]:
        print("results/exp2.json has no error barier fields")
        return

    by_key = {(row["width"], row["pair"]): row for row in base_rows}

    device, tl, Xev, yev = setup(cfg)
    num_layers = 2
    rows = []

    for width in cfg["widths"]:
        for pair_id in range(cfg["k_seeds"]):
            if (width, pair_id) not in by_key:
                continue

            (params_a, _), (params_b, _) = trained_pair([width], pair_id, cfg, tl, device)
            _, aligned_params_b = weight_matching(params_a, params_b, num_layers)

            align_in, align_out = alignability_blocks(params_a, aligned_params_b, num_layers)

            row = dict(by_key[(width, pair_id)])
            row["align_in"] = align_in
            row["align_out"] = align_out
            row["disagreement"] = disagreement(params_a, params_b, [width], Xev, device)
            row["cka"] = linear_cka(params_a, params_b, [width], Xev, device)
            row["flatness"] = 0.5 * (flatness_proxy(params_a, [width], Xev, yev, device)
                + flatness_proxy(params_b, [width], Xev, yev, device))

            set_seed(100 + 2 * pair_id)
            init_params_a = get_params(MLP([width]))

            set_seed(101 + 2 * pair_id)
            init_params_b = get_params(MLP([width]))

            row["flatness_null"] = 0.5 * (
                flatness_proxy(init_params_a, [width], Xev, yev, device)
                + flatness_proxy(init_params_b, [width], Xev, yev, device))
            row["flatness_excess"] = row["flatness"] - row["flatness_null"]

            rows.append(row)

            print(
                f"w={width:5d} k={pair_id}: "
                f"in={align_in:.3f} out={align_out:.3f} "
                f"dis={row['disagreement']:.3f} "
                f"cka={row['cka']:.3f} "
                f"flat={row['flatness']:.3f} "
                f"(null {row['flatness_null']:.3f})")

    def analyse(metric):
        output = {}

        scopes = (
            ("all", rows),
            ("subset", [row for row in rows if row["width"] <= NONZERO_WIDTHS_MAX]),)

        for scope_name, selected_rows in scopes:
            if len({row["width"] for row in selected_rows}) < 3:
                output[scope_name] = {"skipped": "fewer than 3 width levels"}
                continue

            widths = [row["width"] for row in selected_rows]
            barriers = [row[metric] for row in selected_rows]

            def get_values(key):
                return [row[key] for row in selected_rows]

            models = {
                "M0_width_align": {"align": get_values("align")},
                "M_blocks_in_out": {
                    "align_in": get_values("align_in"),
                    "align_out": get_values("align_out"),
                },

                "M_tail": {
                    "align": get_values("align"),
                    "tail_align": get_values("tail_align"),
                },

                "M_disagreement": {
                    "align": get_values("align"),
                    "disagreement": get_values("disagreement"),
                },

                "M_cka": {
                    "align": get_values("align"),
                    "cka": get_values("cka"),
                },

                "M_flatness": {
                    "align": get_values("align"),
                    "flatness": get_values("flatness"),
                },

                "M_flatness_excess": {
                    "align": get_values("align"),
                    "flatness_excess": get_values("flatness_excess"),
                },

                "M_functional_all": {
                    "align": get_values("align"),
                    "disagreement": get_values("disagreement"),
                    "cka": get_values("cka"),
                    "flatness": get_values("flatness"),
                },
            }

            output[scope_name] = {
                name: multi_mediation(widths, mediators, barriers)
                for name, mediators in models.items()
            }

        all_widths = [row["width"] for row in rows]
        all_barriers = [row[metric] for row in rows]

        output["within_width_pooled_r"] = {
            name: within_width_analysis(all_widths,[row[name] for row in rows],all_barriers,)["pooled_within_r_all"]
            for name in ["align"] + CANDIDATE_NAMES
        }

        return output

    analysis = {
        "err": analyse("aligned_err"),
        "loss": analyse("aligned"),
    }

    output_path = os.path.join(cfg["res_dir"], "exp10_residual.json")
    with open(output_path, "w") as f:
        json.dump({"rows": rows, "analysis": analysis}, f, indent=2)

    med_path = os.path.join(cfg["res_dir"], "exp2_mediation.json")

    if os.path.exists(med_path):
        with open(med_path) as f:
            mediation = json.load(f)

        reference = mediation.get("err", {}).get("subset_nonzero_widths", {}).get("beta_width_ctrl_align")
        current = analysis["err"].get("subset", {}).get("M0_width_align", {}).get("beta_width")

        if reference is not None and current is not None:
            is_close = abs(reference - current) < 1e-9
            status = "OK" if is_close else "mismatch"
            print(
                f"\nbaseline reproduction vs exp2_mediation.json: "
                f"{current:+.4f} vs {reference:+.4f} -> {status}")

    print("\nflatness proxy by width: trained vs untrained (null)")

    for width in cfg["widths"]:
        selected_rows = [row for row in rows if row["width"] == width]

        if selected_rows:
            print(
                f"  w={width:5d}: "
                f"trained={np.mean([row['flatness'] for row in selected_rows]):.4f}  "
                f"null={np.mean([row['flatness_null'] for row in selected_rows]):.4f}  "
                f"excess={np.mean([row['flatness_excess'] for row in selected_rows]):+.4f}")

    for metric in ("err", "loss"):
        for scope in ("subset", "all"):
            scope_result = analysis[metric].get(scope, {})

            if "skipped" in scope_result or not scope_result:
                continue

            baseline = scope_result["M0_width_align"]

            def format_adj_r2(model):
                if model["adj_r2"] is None:
                    return "n/a"
                return f"{model['adj_r2']:.3f}"

            scope_label = (f" (w<={NONZERO_WIDTHS_MAX})" if scope == "subset" else "")

            print(f"\nEXP10 [{metric}] {scope}{scope_label} n={baseline['n']}")
            print(
                f"  {'M0 width+align':<18} "
                f"beta_width={baseline['beta_width']:+.3f}  "
                f"adjR2={format_adj_r2(baseline)}")

            for name, model in scope_result.items():
                if name == "M0_width_align":
                    continue

                if model["adj_r2"] is None or baseline["adj_r2"] is None:
                    delta_adj_r2 = None
                else:
                    delta_adj_r2 = model["adj_r2"] - baseline["adj_r2"]

                delta_text = "n/a" if delta_adj_r2 is None else f"{delta_adj_r2:+.3f}"

                print(
                    f"  {name:<18} "
                    f"beta_width={model['beta_width']:+.3f}  "
                    f"adjR2={format_adj_r2(model)} "
                    f"(d {delta_text})")

        within_width = analysis[metric]["within_width_pooled_r"]
        print(f"  within-width pooled r [{metric}]: " + "  ".join(f"{key}={value:+.2f}" for key, value in within_width.items()))

    candidate_analysis = analysis["err"].get("subset", analysis["err"].get("all", {}))

    if (
        candidate_analysis
        and "skipped" not in candidate_analysis
        and candidate_analysis["M0_width_align"]["adj_r2"] is not None
    ):
        baseline_adj_r2 = candidate_analysis["M0_width_align"]["adj_r2"]

        labels = []
        deltas = []

        for name, model in candidate_analysis.items():
            if name == "M0_width_align" or model["adj_r2"] is None:
                continue

            labels.append(name.replace("M_", ""))
            deltas.append(model["adj_r2"] - baseline_adj_r2)

        if labels:
            fig_residual_candidates(labels,deltas, os.path.join(cfg["fig_dir"], "fig10_residual.png"))


if __name__ == "__main__":
    main()
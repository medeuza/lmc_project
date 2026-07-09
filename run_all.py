import importlib
import sys
sys.path.insert(0, ".")

EXPERIMENTS = [
    "exp1_barrier_demo",
    "exp2_width_sweep",
    "exp3_activation_vs_weight",
    "exp4_repair_residual",
    "exp5_lmc_over_training",
    "exp6_two_layer",
    "exp7_robustness",
    "exp8_cnn",
    "exp9_null",
    "exp10_residual",
    "exp11_kmerge",
]

for name in EXPERIMENTS:
    print(name, flush=True)
    importlib.import_module(f"experiments.{name}").main()
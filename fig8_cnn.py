import json, sys
sys.path.insert(0, ".")
from src.plotting import fig_cnn_channels

rows = json.load(open("results/exp8_cnn.json"))
fig_cnn_channels(rows, "figures/fig8_cnn_channels.png")
print("figures/fig8_cnn_channels.png written (%d pairs)" % len(rows))

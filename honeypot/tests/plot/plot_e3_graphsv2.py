import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =======================
# Updated data
# =======================

rows = [
    dict(model="Gemma",  cpu_mean=7.89, cpu_peak=8.9,  ram_mean=16.21, ram_peak=16.84,
         gpu_mean=41.63, gpu_peak=77,   vram_mean=14.61, vram_peak=18.61, power_mean=159.09, power_peak=177.76),
    dict(model="LLaMA 3", cpu_mean=1.02, cpu_peak=1.2,  ram_mean=14.20, ram_peak=14.23,
         gpu_mean=34.55, gpu_peak=68,   vram_mean=20.60, vram_peak=21.61, power_mean=140.74, power_peak=164.26),
    dict(model="Zephyr", cpu_mean=1.22, cpu_peak=2.3,  ram_mean=14.52, ram_peak=15.90,
         gpu_mean=32.40, gpu_peak=64,   vram_mean=13.61, vram_peak=15.61, power_mean=136.47, power_peak=155.03),
]


df = pd.DataFrame(rows)
df["model"] = pd.Categorical(df["model"], categories=["Gemma", "LLaMA 3", "Zephyr"], ordered=True)

# =======================
# Style configuration
# =======================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

MODEL_COLORS = {
    "Gemma": '#1C3552',
    "LLaMA 3": "#555555",  # Azul pastel
    "Zephyr": "#9B1B30"    # Verde pastel
}

metrics = [
    ("CPU utilisation (%)", "cpu_mean", "cpu_peak"),
    ("RAM (GB)", "ram_mean", "ram_peak"),
    ("GPU utilisation (%)", "gpu_mean", "gpu_peak"),
    ("VRAM (GB)", "vram_mean", "vram_peak"),
    ("GPU power (W)", "power_mean", "power_peak"),
]

# =======================
# Chart
# =======================
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.2), constrained_layout=True) 

for ax, (title, mean_col, peak_col) in zip(axes, metrics):
    x = np.arange(len(df))
    mean_vals = df[mean_col].values
    peak_vals = df[peak_col].values
    err = np.maximum(peak_vals - mean_vals, 0)

    # Draw bars
    bars = ax.bar(
        x, mean_vals, yerr=err, capsize=3,
        color=[MODEL_COLORS[m] for m in df["model"]],
        edgecolor="grey", linewidth=0.7
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=0)
    ax.set_title(title, pad=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

plt.tight_layout(pad=1)
plt.savefig("figA_resource_profile_final.png", dpi=400, bbox_inches="tight")
plt.show()

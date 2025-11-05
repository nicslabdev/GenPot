import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Input configuration
# =======================
csv_path = "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/fastapi/results/resources/resources_raw.csv"

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# =======================
# Relative time (minutes from start per model)
# =======================
df_sorted = df.sort_values("timestamp").copy()
df_sorted["time_min"] = 0.0

for model in df_sorted["test_model"].unique():
    mask = df_sorted["test_model"] == model
    t0 = df_sorted.loc[mask, "timestamp"].iloc[0]
    df_sorted.loc[mask, "time_min"] = (df_sorted.loc[mask, "timestamp"] - t0).dt.total_seconds() / 60.0

# =======================
# Select key metrics
# =======================
metrics = {
    "CPU utilisation (%)": "cpu_percent_total",
    "RAM used (GB)": "ram_used_gb",
    "GPU utilisation (%)": "gpu_3_util_percent",
    "VRAM used (GB)": "gpu_3_vram_used_gb",
    "GPU temperature (°C)": "gpu_3_temp_c",
    "GPU power (W)": "gpu_3_power_w"
}

# =======================
# Style
# =======================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

MODEL_COLORS = {
    "gemma": "#1C3552",   # Azul acero oscuro
    "llama3": "#555555",  # Gris neutro
    "zephyr": "#9B1B30"   # Granate sobrio
}

# =======================
# Figure
# =======================
fig, axes = plt.subplots(len(metrics), 1, figsize=(7.5, 8.8), sharex=True)
fig.subplots_adjust(hspace=0.45)

for i, (label, col) in enumerate(metrics.items()):
    ax = axes[i]
    for model, color in MODEL_COLORS.items():
        df_model = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
        if df_model.empty:
            continue
        ax.plot(df_model["time_min"], df_model[col], lw=1.5, label=model.capitalize(), color=color)
    ax.set_ylabel(label, fontsize=9.5)
    ax.set_title(label, loc="left", fontsize=10, color="#666666", pad=4)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=8.5)
    if i < len(metrics) - 1:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Time (minutes)", fontsize=10)

axes[-1].legend(loc="upper right", frameon=False, ncol=3)
plt.tight_layout(pad=1.2)

# =======================
# Save and display
# =======================
plt.savefig("fig_timeseries_resources_all_models.png", dpi=400, bbox_inches="tight")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Input
# =======================
csv_path = "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/fastapi/results/resources/resources_raw.csv"
df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Relative time (minutes)
df_sorted = df.sort_values("timestamp").copy()
df_sorted["time_min"] = 0.0
for model in df_sorted["test_model"].unique():
    mask = df_sorted["test_model"] == model
    t0 = df_sorted.loc[mask, "timestamp"].iloc[0]
    df_sorted.loc[mask, "time_min"] = (df_sorted.loc[mask, "timestamp"] - t0).dt.total_seconds() / 60.0

# =======================
# Metrics and labels
# =======================
metrics = {
    "CPU utilisation (%)": "cpu_percent_total",
    "RAM used (GB)": "ram_used_gb",
    "GPU utilisation (%)": "gpu_3_util_percent",
    "VRAM used (GB)": "gpu_3_vram_used_gb",
    "GPU temperature (°C)": "gpu_3_temp_c",
    "GPU power (W)": "gpu_3_power_w"
}

model_order = ["gemma", "llama3", "zephyr"]
MODEL_TITLES = {"gemma": "Gemma", "llama3": "LLaMA 3", "zephyr": "Zephyr"}
MODEL_COLORS = {"gemma": "#E63946", "llama3": "#457B9D", "zephyr": "#2A9D8F"}

# =======================
# Style
# =======================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# =======================
# Create wider 3×6 grid
# =======================
fig, axes = plt.subplots(len(metrics), len(model_order), figsize=(12, 9.5), sharex=False)
fig.subplots_adjust(hspace=0.45, wspace=0.45)

for col_idx, model in enumerate(model_order):
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    if d.empty:
        continue
    color = MODEL_COLORS[model]
    for row_idx, (label, colname) in enumerate(metrics.items()):
        ax = axes[row_idx, col_idx]
        ax.plot(d["time_min"], d[colname], lw=1.6, color=color)
        if col_idx == 0:
            ax.set_ylabel(label, fontsize=9.2)
        else:
            ax.set_ylabel("")
        if row_idx == len(metrics) - 1:
            ax.set_xlabel("Time (minutes)", fontsize=9.2)
        else:
            ax.set_xlabel("")
        if row_idx == 0:
            ax.set_title(MODEL_TITLES[model], fontsize=10.5, color="#222", pad=3)
        ax.tick_params(axis="both", labelsize=8.7)

plt.tight_layout(pad=1.4)
plt.savefig("fig_resources_comparative_3x6.png", dpi=400, bbox_inches="tight")
plt.show()

print("✅ Saved figure: fig_resources_comparative_3x6.png")

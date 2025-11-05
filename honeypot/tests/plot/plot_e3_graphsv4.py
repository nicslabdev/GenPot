import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Input
# =======================
csv_path = "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/fastapi/results/resources/bk/resources_raw.csv"
df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Relative time (min) per model
df_sorted = df.sort_values("timestamp").copy()
df_sorted["time_min"] = 0.0
for model in df_sorted["test_model"].unique():
    mask = df_sorted["test_model"] == model
    t0 = df_sorted.loc[mask, "timestamp"].iloc[0]
    df_sorted.loc[mask, "time_min"] = (df_sorted.loc[mask, "timestamp"] - t0).dt.total_seconds() / 60.0

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

MODEL_COLORS = {
    "gemma": "#1C3552",   # Azul acero
    "llama3": "#555555",  # Gris neutro
    "zephyr": "#9B1B30"   # Granate sobrio
}

# =======================
# Figure (3 panels)
# =======================
fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
fig.subplots_adjust(hspace=0.35)

# === PANEL 1: CPU & RAM ===
ax1 = axes[0]
for model, color in MODEL_COLORS.items():
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    ax1.plot(d["time_min"], d["cpu_percent_total"], lw=1.4, color=color, label=model.capitalize())
ax1.set_ylabel("CPU utilisation (%)", fontsize=9.5)
ax1_twin = ax1.twinx()
for model, color in MODEL_COLORS.items():
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    ax1_twin.plot(d["time_min"], d["ram_used_gb"], lw=1.2, ls="--", color=color, alpha=0.7)
ax1_twin.set_ylabel("RAM used (GB)", fontsize=9.5, color="#666666")
ax1.set_title("System utilisation (CPU & RAM)", loc="left", fontsize=10, color="#444")

# === PANEL 2: GPU & VRAM ===
ax2 = axes[1]
for model, color in MODEL_COLORS.items():
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    ax2.plot(d["time_min"], d["gpu_3_util_percent"], lw=1.4, color=color)
ax2.set_ylabel("GPU utilisation (%)", fontsize=9.5)
ax2_twin = ax2.twinx()
for model, color in MODEL_COLORS.items():
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    ax2_twin.plot(d["time_min"], d["gpu_3_vram_used_gb"], lw=1.2, ls="--", color=color, alpha=0.7)
ax2_twin.set_ylabel("VRAM used (GB)", fontsize=9.5, color="#666666")
ax2.set_title("Accelerator load (GPU & VRAM)", loc="left", fontsize=10, color="#444")

# === PANEL 3: Temp & Power ===
ax3 = axes[2]
for model, color in MODEL_COLORS.items():
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    ax3.plot(d["time_min"], d["gpu_3_temp_c"], lw=1.4, color=color)
ax3.set_ylabel("Temperature (°C)", fontsize=9.5)
ax3_twin = ax3.twinx()
for model, color in MODEL_COLORS.items():
    d = df_sorted[(df_sorted["test_model"] == model) & (df_sorted["test_pipeline"] == "system")]
    ax3_twin.plot(d["time_min"], d["gpu_3_power_w"], lw=1.2, ls="--", color=color, alpha=0.7)
ax3_twin.set_ylabel("Power (W)", fontsize=9.5, color="#666666")
ax3.set_title("Thermal and power behaviour", loc="left", fontsize=10, color="#444")

# === Shared X ===
axes[-1].set_xlabel("Time (minutes)", fontsize=10)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.75, 1.18), frameon=False, ncol=3)

plt.tight_layout(pad=1.2)
plt.savefig("fig_resources_3panel_all_models.png", dpi=400, bbox_inches="tight")
plt.show()

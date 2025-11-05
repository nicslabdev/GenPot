#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph generation for E3 - Resource Profile
Generates:
- Fig 3a: Stacked bars (CPU/RAM/GPU/VRAM) per model & quantization
- Fig 3b: Time-series for representative 2-minute window
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

def plot_fig3a_stacked_bars(df_summary: pd.DataFrame, output_path: str):
    """
    Fig 3a: Stacked bars showing CPU/RAM/GPU/VRAM usage per model & quantization
    """
    # Group by model and quantization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    models = df_summary['model'].unique()
    x_pos = np.arange(len(df_summary))
    width = 0.6
    
    # Labels for the X-axis
    labels = [f"{row['model']}\n{row.get('quantization', 'N/A')}\n({row.get('pipeline', 'N/A')})" 
              for _, row in df_summary.iterrows()]
    
    # --- Panel 1: CPU and RAM ---
    cpu_mean = df_summary['cpu_percent_mean'].fillna(0)
    ram_mean = df_summary['ram_used_gb_mean'].fillna(0)
    
    ax1.bar(x_pos, cpu_mean, width, label='CPU %', color='#3498db', alpha=0.8)
    ax1.set_xlabel('Model & Configuration', fontweight='bold')
    ax1.set_ylabel('CPU Usage (%)', fontweight='bold', color='#3498db')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Second Y-axis for RAM
    ax1_ram = ax1.twinx()
    ax1_ram.bar(x_pos, ram_mean, width, label='RAM (GB)', color='#e74c3c', alpha=0.5)
    ax1_ram.set_ylabel('RAM Usage (GB)', fontweight='bold', color='#e74c3c')
    ax1_ram.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ram.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
    
    ax1.set_title('CPU & RAM Usage', fontsize=12, fontweight='bold')
    
    # --- Panel 2: GPU and VRAM ---
    if 'gpu_util_percent_mean' in df_summary.columns and 'vram_used_gb_mean' in df_summary.columns:
        gpu_mean = df_summary['gpu_util_percent_mean'].fillna(0)
        vram_mean = df_summary['vram_used_gb_mean'].fillna(0)
        
        ax2.bar(x_pos, gpu_mean, width, label='GPU %', color='#2ecc71', alpha=0.8)
        ax2.set_xlabel('Model & Configuration', fontweight='bold')
        ax2.set_ylabel('GPU Utilization (%)', fontweight='bold', color='#2ecc71')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Second Y-axis for VRAM
        ax2_vram = ax2.twinx()
        ax2_vram.bar(x_pos, vram_mean, width, label='VRAM (GB)', color='#9b59b6', alpha=0.5)
        ax2_vram.set_ylabel('VRAM Usage (GB)', fontweight='bold', color='#9b59b6')
        ax2_vram.tick_params(axis='y', labelcolor='#9b59b6')
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_vram.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
        
        ax2.set_title('GPU & VRAM Usage', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'GPU data not available', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('GPU & VRAM Usage', fontsize=12, fontweight='bold')
    
    plt.suptitle('Fig 3a: Resource Usage by Model & Quantization', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Fig 3a saved: {output_path}")

def plot_fig3b_timeseries(df_raw: pd.DataFrame, output_path: str, window_minutes: int = 2):
    """
    Fig 3b: Time-series of resource usage for a representative window
    """
    if len(df_raw) == 0:
        print("[WARN] No raw data available for time-series plot")
        return
    
    # Select representative window (center of the test)
    total_samples = len(df_raw)
    window_samples = window_minutes * 60  # samples per minute
    
    if total_samples > window_samples:
        start_idx = (total_samples - window_samples) // 2
        end_idx = start_idx + window_samples
        df_window = df_raw.iloc[start_idx:end_idx].copy()
    else:
        df_window = df_raw.copy()
    
    # Create relative time from the start of the window
    df_window['time_offset'] = range(len(df_window))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- CPU ---
    ax = axes[0, 0]
    if 'cpu_percent_total' in df_window.columns:
        ax.plot(df_window['time_offset'], df_window['cpu_percent_total'], 
                linewidth=1.5, color='#3498db', label='CPU %')
        ax.fill_between(df_window['time_offset'], 0, df_window['cpu_percent_total'], 
                        alpha=0.3, color='#3498db')
        ax.set_ylabel('CPU Usage (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title('CPU Usage Over Time', fontweight='bold')
    
    # --- RAM ---
    ax = axes[0, 1]
    if 'ram_used_gb' in df_window.columns:
        ax.plot(df_window['time_offset'], df_window['ram_used_gb'], 
                linewidth=1.5, color='#e74c3c', label='RAM (GB)')
        ax.fill_between(df_window['time_offset'], 0, df_window['ram_used_gb'], 
                        alpha=0.3, color='#e74c3c')
        ax.set_ylabel('RAM Usage (GB)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title('RAM Usage Over Time', fontweight='bold')
    
    # --- GPU Utilization ---
    ax = axes[1, 0]
    if 'gpu_0_util_percent' in df_window.columns:
        ax.plot(df_window['time_offset'], df_window['gpu_0_util_percent'], 
                linewidth=1.5, color='#2ecc71', label='GPU Util %')
        ax.fill_between(df_window['time_offset'], 0, df_window['gpu_0_util_percent'], 
                        alpha=0.3, color='#2ecc71')
        ax.set_ylabel('GPU Utilization (%)', fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title('GPU Utilization Over Time', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'GPU data not available', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('GPU Utilization Over Time', fontweight='bold')
    
    # --- VRAM ---
    ax = axes[1, 1]
    if 'gpu_0_vram_used_gb' in df_window.columns:
        ax.plot(df_window['time_offset'], df_window['gpu_0_vram_used_gb'], 
                linewidth=1.5, color='#9b59b6', label='VRAM (GB)')
        ax.fill_between(df_window['time_offset'], 0, df_window['gpu_0_vram_used_gb'], 
                        alpha=0.3, color='#9b59b6')
        ax.set_ylabel('VRAM Usage (GB)', fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title('VRAM Usage Over Time', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'VRAM data not available', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('VRAM Usage Over Time', fontweight='bold')
    
    # Test metadata
    if 'test_model' in df_window.columns:
        model = df_window['test_model'].iloc[0]
        pipeline = df_window.get('test_pipeline', pd.Series(['N/A'])).iloc[0]
        quant = df_window.get('test_quantization', pd.Series(['N/A'])).iloc[0]
        fig.suptitle(f'Fig 3b: Resource Usage Time-Series ({model} - {pipeline} - {quant})\n'
                    f'{window_minutes}-minute window', 
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Fig 3b: Resource Usage Time-Series\n{window_minutes}-minute window', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Fig 3b saved: {output_path}")

def plot_additional_metrics(df_raw: pd.DataFrame, output_dir: str):
    """
    Additional graphs: temperature, power, etc.
    """
    if len(df_raw) == 0:
        return
    
    df_raw['time_offset'] = range(len(df_raw))
    
    # Fig 3c: GPU Temperature & Power
    if 'gpu_0_temp_c' in df_raw.columns or 'gpu_0_power_w' in df_raw.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Temperature
        ax = axes[0]
        if 'gpu_0_temp_c' in df_raw.columns:
            ax.plot(df_raw['time_offset'], df_raw['gpu_0_temp_c'], 
                   linewidth=1.5, color='#e67e22', label='GPU Temperature')
            ax.fill_between(df_raw['time_offset'], 0, df_raw['gpu_0_temp_c'], 
                           alpha=0.3, color='#e67e22')
            ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, 
                      label='Warning (80°C)', alpha=0.7)
            ax.axhline(y=90, color='red', linestyle='--', linewidth=1, 
                      label='Critical (90°C)', alpha=0.7)
            ax.set_ylabel('Temperature (°C)', fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_title('GPU Temperature', fontweight='bold')
        
        # Power
        ax = axes[1]
        if 'gpu_0_power_w' in df_raw.columns:
            ax.plot(df_raw['time_offset'], df_raw['gpu_0_power_w'], 
                   linewidth=1.5, color='#f39c12', label='GPU Power')
            ax.fill_between(df_raw['time_offset'], 0, df_raw['gpu_0_power_w'], 
                           alpha=0.3, color='#f39c12')
            ax.set_ylabel('Power (W)', fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_title('GPU Power Consumption', fontweight='bold')
        
        plt.suptitle('Fig 3c: GPU Temperature & Power', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(output_dir, "fig3c_gpu_temp_power.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[✓] Fig 3c saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate E3 resource profile graphs")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/resources/resources_summary.csv",
        help="Path to resources_summary.csv"
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="results/resources/resources_raw.csv",
        help="Path to resources_raw.csv (for time-series)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/resources",
        help="Output directory for graphs"
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=2,
        help="Window size in minutes for time-series plot (default: 2)"
    )
    parser.add_argument(
        "--extra-plots",
        action="store_true",
        help="Generate additional plots (temperature, power)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("E3 RESOURCE PROFILE - GRAPH GENERATION")
    print(f"{'='*70}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load summary
    if not os.path.exists(args.summary_csv):
        print(f"[ERROR] Summary file not found: {args.summary_csv}")
        return
    
    df_summary = pd.read_csv(args.summary_csv)
    print(f"[INFO] Loaded summary: {len(df_summary)} configurations")
    print(f"[INFO] Models: {df_summary['model'].unique().tolist()}")
    
    # Fig 3a: Stacked bars
    fig3a_path = os.path.join(args.output_dir, "fig3a_resource_bars.png")
    plot_fig3a_stacked_bars(df_summary, fig3a_path)
    
    # Fig 3b: Time-series (requires raw data)
    if os.path.exists(args.raw_csv):
        df_raw = pd.read_csv(args.raw_csv)
        print(f"[INFO] Loaded raw data: {len(df_raw)} samples")
        
        fig3b_path = os.path.join(args.output_dir, "fig3b_timeseries.png")
        plot_fig3b_timeseries(df_raw, fig3b_path, args.window_minutes)
        
        # Extra plots
        if args.extra_plots:
            print("\n[INFO] Generating additional plots...")
            plot_additional_metrics(df_raw, args.output_dir)
    else:
        print(f"[WARN] Raw data file not found: {args.raw_csv}")
        print(f"[WARN] Skipping Fig 3b (time-series)")
    
    print(f"\n{'='*70}")
    print("✅ GRAPH GENERATION COMPLETED")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

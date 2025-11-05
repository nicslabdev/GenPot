#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph generation for E2 - Throughput & Scalability
Generates:
- Fig 2a: RPS vs p95 latency (knee curve)
- Fig 2b: Concurrency vs error rate
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10



def plot_fig2a_knee_curve(df: pd.DataFrame, output_path: str, slo_threshold: float = 300):
    """
    Fig 2a: Throughput vs Latency (Knee Curve)
    Shows the inflection point where latency (p95) starts to grow rapidly
    as throughput (RPS) increases.
    """
    # Filter inconsistent or extreme data
    df = df.dropna(subset=['rps', 'latency_p95_ms'])
    df = df[df['rps'] > 0]

    # Ensure logical order: first by model, then by RPS
    df = df.sort_values(['model', 'rps'])

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        'gemma': '#1C3552',   # Rojo
        'llama3': '#555555',  # Azul
        'zephyr': '#9B1B30'   # Verde
    }
    markers = {
        'gemma': 'o',
        'llama3': 's',
        'zephyr': '^'
    }

    # Solo una pipeline (normalmente 'api')
    df_api = df[df['pipeline'] == 'system']

    for model in df_api['model'].unique():
        group = df_api[df_api['model'] == model].sort_values('rps')
        ax.plot(
            group['rps'],
            group['latency_p95_ms'],
            marker=markers.get(model, 'o'),
            color=colors.get(model, '#888888'),
            linewidth=2.2,
            markersize=9,
            label=model.capitalize(),
            alpha=0.9
        )

        # Dibujar punto SLO si supera umbral de latencia
        knee = group[group['latency_p95_ms'] > slo_threshold]
        if not knee.empty:
            first_knee = knee.iloc[0]
            ax.scatter(
                first_knee['rps'],
                first_knee['latency_p95_ms'],
                color=colors.get(model, '#888888'),
                edgecolor='black',
                s=90,
                zorder=5,
                label=f"{model.capitalize()} SLO"
            )

    # Etiquetas y estilo
    ax.set_xlabel("RPS (Requests per Second)", fontsize=12, fontweight='bold')
    ax.set_ylabel("p95 Latency (ms)", fontsize=12, fontweight='bold')
    ax.set_title("Fig 2a: Throughput vs Latency (Knee Curve)", fontsize=14, fontweight='bold', pad=20)

    # Escala logarítmica si hay mucha dispersión
    if df['latency_p95_ms'].max() / df['latency_p95_ms'].min() > 8:
        ax.set_yscale('log')
        ax.set_ylabel("p95 Latency (ms, log scale)", fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.35)
    ax.legend(loc='lower right', frameon=True, shadow=False, fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Fig 2a saved: {output_path}")

def plot_fig2b_error_rate(df: pd.DataFrame, output_path: str):
    """
    Fig 2b: Concurrency vs Error Rate
    Shows how the error rate increases with concurrency.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Agrupar solo por modelo
    groups = df.groupby('model')
    
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#F77F00', '#9467bd', '#8c564b']  # Rojo, Azul, Verde, Naranja
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Ordenar por concurrency
        group = group.sort_values('concurrency')
        
        # Error rate total (errors + timeouts)
        ax.plot(group['concurrency'], group['error_rate'] * 100, 
                marker=marker, linewidth=2, markersize=8,
                color=color, label=label, alpha=0.8)
    
    # Línea de referencia 1% error rate
    ax.axhline(y=10.0, color='orange', linestyle='--', 
               linewidth=2, label='10% Error Rate (Warning)', alpha=0.7)
    
    # Línea de referencia 5% error rate
    ax.axhline(y=50.0, color='red', linestyle='--', 
               linewidth=2, label='50% Error Rate (Critical)', alpha=0.7)
    
    ax.set_xlabel("Concurrency (Concurrent Users)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Error Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Fig 2b: Concurrency vs Error Rate", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Configurar eje X con potencias de 2
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(['2', '4', '8', '16', '32'])
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    
    # Limitar Y axis a 100%
    ax.set_ylim(bottom=0, top=min(100, df['error_rate'].max() * 100 * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Fig 2b saved: {output_path}")

def plot_combined_2x2(df: pd.DataFrame, output_path: str, slo_threshold: float = 300):
    """
    Generates a combined 2x2 image with the 4 main graphs of E2.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color and marker configuration
    groups = df.groupby('model')
    colors = ['#1C3552', '#555555', '#9B1B30', '#F77F00']
    markers = ['o', 's', '^', 'D']
    
    # === Fig 2a: RPS vs p95 Latency (top left) ===
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        group = group.sort_values('rps')
        ax1.plot(group['rps'], group['latency_p95_ms'], 
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    ax1.set_xlabel("RPS (Requests per Second)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("p95 Latency (ms)", fontsize=12, fontweight='bold')
    ax1.set_title("Fig 5a: Throughput vs Latency", 
                 fontsize=14, fontweight='bold', pad=20)
    
    if df['latency_p95_ms'].max() / df['latency_p95_ms'].min() > 10:
        ax1.set_yscale('log')
        ax1.set_ylabel("p95 Latency (ms) - log scale", fontsize=12, fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    
    # === Fig 2b: Concurrency vs Error Rate (top right) ===
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        group = group.sort_values('concurrency')
        ax2.plot(group['concurrency'], group['error_rate'] * 100, 
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    ax2.axhline(y=10.0, color='orange', linestyle='--', 
               linewidth=2, label='10% Error Rate (Warning)', alpha=0.7)
    ax2.axhline(y=50.0, color='red', linestyle='--', 
               linewidth=2, label='50% Error Rate (Critical)', alpha=0.7)
    
    ax2.set_xlabel("Concurrency (Concurrent Users)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Error Rate (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Fig 5b: Concurrency vs Error Rate", 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks([2, 4, 8, 16, 32])
    ax2.set_xticklabels(['2', '4', '8', '16', '32'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
    ax2.set_ylim(bottom=0, top=min(100, df['error_rate'].max() * 100 * 1.1))
    
    # === Fig 2c: Concurrency vs RPS (bottom left) ===
    df['concurrency'] = pd.to_numeric(df['concurrency'], errors='coerce')
    df['rps'] = pd.to_numeric(df['rps'], errors='coerce')
    
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        group = group.sort_values('concurrency')
        ax3.plot(group['concurrency'], group['rps'], 
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    max_conc = float(df['concurrency'].max())
    rps_per_user = float(df[df['concurrency'] == 1]['rps'].mean())
    
    if not np.isnan(max_conc) and not np.isnan(rps_per_user):
        ideal_x = np.linspace(0, max_conc, 100)
        ideal_y = ideal_x * rps_per_user
        ax3.plot(ideal_x, ideal_y, 'k--', linewidth=2, alpha=0.5, label='Ideal Linear Scaling')
    
    ax3.set_xlabel("Concurrency (Concurrent Users)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("RPS (Requests per Second)", fontsize=12, fontweight='bold')
    ax3.set_title("Fig 5c: Concurrency vs RPS", 
                 fontsize=14, fontweight='bold', pad=20)
    ax3.set_xticks([2, 4, 8, 16, 32])
    ax3.set_xticklabels(['2', '4', '8', '16', '32'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    
    # === Fig 2d: Concurrency vs Queueing Delay (bottom right) ===
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        group = group.sort_values('concurrency')
        ax4.plot(group['concurrency'], group['queueing_delay_ms'], 
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    ax4.set_xlabel("Concurrency (Concurrent Users)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Queueing Delay (ms)", fontsize=12, fontweight='bold')
    ax4.set_title("Fig 5d: Concurrency vs Queueing Delay", 
                 fontsize=14, fontweight='bold', pad=20)
    ax4.set_xticks([2, 4, 8, 16, 32])
    ax4.set_xticklabels(['2', '4', '8', '16', '32'])
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Combined 2x2 plot saved: {output_path}")

def plot_additional_metrics(df: pd.DataFrame, output_dir: str):
    """
    Additional graphs useful for deep analysis.
    """
    
    # Fig 2c: Concurrency vs RPS (linear scaling)
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = df.groupby('model')
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#F77F00']  # Rojo, Azul, Verde, Naranja
    markers = ['o', 's', '^', 'D']
    
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        group = group.sort_values('concurrency')
        ax.plot(group['concurrency'], group['rps'], 
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    # Línea de scaling ideal (linear)
    # Convert to numeric to avoid DType errors
    df['concurrency'] = pd.to_numeric(df['concurrency'], errors='coerce')
    df['rps'] = pd.to_numeric(df['rps'], errors='coerce')
    
    max_conc = float(df['concurrency'].max())
    rps_per_user = float(df[df['concurrency'] == 1]['rps'].mean())
    
    if not np.isnan(max_conc) and not np.isnan(rps_per_user):
        ideal_x = np.linspace(0, max_conc, 100)
        ideal_y = ideal_x * rps_per_user
        ax.plot(ideal_x, ideal_y, 'k--', linewidth=2, alpha=0.5, label='Ideal Linear Scaling')
    
    ax.set_xlabel("Concurrency (Concurrent Users)", fontsize=12, fontweight='bold')
    ax.set_ylabel("RPS (Requests per Second)", fontsize=12, fontweight='bold')
    ax.set_title("Fig 2c: Concurrency vs RPS (Scaling)", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Configurar eje X con potencias de 2
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(['2', '4', '8', '16', '32'])
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2c_concurrency_vs_rps.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Fig 2c saved: {os.path.join(output_dir, 'fig2c_concurrency_vs_rps.png')}")
    
    # Fig 2d: Concurrency vs Queueing Delay
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (model, group) in enumerate(groups):
        label = str(model).capitalize()
        group = group.sort_values('concurrency')
        ax.plot(group['concurrency'], group['queueing_delay_ms'], 
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    ax.set_xlabel("Concurrency (Concurrent Users)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Queueing Delay (ms)", fontsize=12, fontweight='bold')
    ax.set_title("Fig 2d: Concurrency vs Queueing Delay", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Configurar eje X con potencias de 2
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(['2', '4', '8', '16', '32'])
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2d_concurrency_vs_queueing.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Fig 2d saved: {os.path.join(output_dir, 'fig2d_concurrency_vs_queueing.png')}")

def main():
    parser = argparse.ArgumentParser(description="Generate E2 load test graphs")
    parser.add_argument(
        "--summary-csv", 
        type=str, 
        default="results/load/combined_summary.csv.bk",
        help="Path to load_summary.csv"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/load",
        help="Output directory for graphs"
    )
    parser.add_argument(
        "--slo-threshold", 
        type=float, 
        default=300.0,
        help="SLO threshold for p95 latency (ms)"
    )
    parser.add_argument(
        "--extra-plots", 
        action="store_true",
        help="Generate additional plots (Fig 2c, 2d)"
    )
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.summary_csv):
        print(f"[ERROR] File not found: {args.summary_csv}")
        return
    
    df = pd.read_csv(args.summary_csv)
    
    # Filter only "system" pipeline
    df = df[df['pipeline'] == 'system']
    
    print(f"\n{'='*60}")
    print("E2 LOAD TEST - GRAPH GENERATION")
    print(f"{'='*60}")
    print(f"Input: {args.summary_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Rows: {len(df)} (filtered: pipeline=system only)")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Concurrency levels: {sorted(df['concurrency'].unique().tolist())}")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generar imagen combinada 2x2
    combined_path = os.path.join(args.output_dir, "fig2_combined_2x2.png")
    plot_combined_2x2(df, combined_path, args.slo_threshold)
    
    # Generar gráficos individuales si se solicita
    if args.extra_plots:
        print("\n[INFO] Generating individual plots...")
        fig2a_path = os.path.join(args.output_dir, "fig2a_rps_vs_p95.png")
        plot_fig2a_knee_curve(df, fig2a_path, args.slo_threshold)
        
        fig2b_path = os.path.join(args.output_dir, "fig2b_concurrency_vs_errors.png")
        plot_fig2b_error_rate(df, fig2b_path)
        
        plot_additional_metrics(df, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✅ GRAPH GENERATION COMPLETED")
    print(f"{'='*60}")
    print(f"Generated files:")
    print(f"  • {combined_path} (Combined 2x2)")
    if args.extra_plots:
        print(f"  • {fig2a_path}")
        print(f"  • {fig2b_path}")
        print(f"  • {os.path.join(args.output_dir, 'fig2c_concurrency_vs_rps.png')}")
        print(f"  • {os.path.join(args.output_dir, 'fig2d_concurrency_vs_queueing.png')}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Consistent colors per model (differentiated pastel palette)
MODEL_COLORS = {
    'gemma': '#1C3552',    # Rosa pastel (coral suave)
    'llama3': '#555555',   # Azul pastel (cielo suave)
    'zephyr': '#9B1B30',   # Verde pastel (menta suave)
}

# Readable names
MODEL_NAMES = {
    'gemma': 'Gemma',
    'llama3': 'LLaMA 3',
    'zephyr': 'Zephyr',
}


def load_summary_data(input_dir: Path) -> pd.DataFrame:
    """
    Load and combine summary files from the input directory.
    Searches for both summary_percentiles.csv.* and correctness_summary.csv.*
    
    Args:
        input_dir: Directorio con los CSV de resultados
        
    Returns:
        DataFrame consolidado con todos los datos de resumen
    """
    # Search for latency files (percentiles)
    latency_files = list(input_dir.glob('summary_percentiles.csv.*'))
    
    # Search for correctness files
    correctness_files = list(input_dir.glob('correctness_summary.csv.*'))
    
    if not latency_files and not correctness_files:
        raise FileNotFoundError(f"No se encontraron archivos summary_percentiles.csv.* o correctness_summary.csv.* en {input_dir}")
    
    print(f"📊 Encontrados archivos de resumen:")
    print(f"   - Latencia: {len(latency_files)} archivo(s)")
    for f in latency_files:
        print(f"     * {f.name}")
    print(f"   - Correctitud: {len(correctness_files)} archivo(s)")
    for f in correctness_files:
        print(f"     * {f.name}")
    
    # Load latencies
    latency_dfs = []
    for file in latency_files:
        df = pd.read_csv(file)
        # Extraer network del sufijo (.lan, .wan, etc.)
        network = file.suffix.lstrip('.')
        df['network_type'] = network
        latency_dfs.append(df)
    
    # Load correctness
    correctness_dfs = []
    for file in correctness_files:
        df = pd.read_csv(file)
        network = file.suffix.lstrip('.')
        df['network_type'] = network
        correctness_dfs.append(df)
    
    # Combine latencies
    df_latency = pd.concat(latency_dfs, ignore_index=True) if latency_dfs else pd.DataFrame()
    
    # Combine correctness
    df_correctness = pd.concat(correctness_dfs, ignore_index=True) if correctness_dfs else pd.DataFrame()
    
    # Merge by model, pipeline, network
    if not df_latency.empty and not df_correctness.empty:
        combined = pd.merge(
            df_latency, 
            df_correctness,
            on=['model', 'pipeline', 'network'],
            how='outer',
            suffixes=('_latency', '_correctness')
        )
    elif not df_latency.empty:
        combined = df_latency
    elif not df_correctness.empty:
        combined = df_correctness
    else:
        raise ValueError("No se pudieron cargar datos de latencia ni correctitud")
    
    # Add derived columns for compatibility
    if 'ttlb_p50_ms' in combined.columns:
        combined['mean_latency_s'] = combined['ttlb_p50_ms'] / 1000.0  # Convertir a segundos
    if 'ttlb_jitter_std_ms' in combined.columns:
        combined['jitter_std'] = combined['ttlb_jitter_std_ms'] / 1000.0
    if 'ttlb_jitter_iqr_ms' in combined.columns:
        combined['jitter_iqr'] = combined['ttlb_jitter_iqr_ms'] / 1000.0
    
    # Rename correctness metrics for compatibility
    if 'parse_rate' in combined.columns:
        combined['ok_parse_rate'] = combined['parse_rate']
    if 'accuracy' in combined.columns:
        combined['class_accuracy'] = combined['accuracy']
    if 'avg_similarity' in combined.columns:
        combined['mean_similarity'] = combined['avg_similarity']
    if 'avg_key_match' in combined.columns:
        combined['key_match_rate'] = combined['avg_key_match']
    
    print(f"✅ Cargados {len(combined)} registros totales")
    print(f"   Columnas disponibles: {', '.join(combined.columns.tolist()[:10])}...")
    return combined


def load_raw_data(input_dir: Path) -> pd.DataFrame:
    """
    Load all raw CSV files from the input directory.
    
    Args:
        input_dir: Directorio con los CSV de resultados
        
    Returns:
        DataFrame consolidado con todos los datos individuales
    """
    raw_files = list(input_dir.glob('latency_raw.csv.*'))
    
    if not raw_files:
        print("⚠️  No se encontraron archivos latency_raw.csv.* (opcional para algunos gráficos)")
        return pd.DataFrame()
    
    print(f"📊 Encontrados {len(raw_files)} archivos raw:")
    for f in raw_files:
        print(f"   - {f.name}")
    
    dfs = []
    for file in raw_files:
        df = pd.read_csv(file)
        # Extraer network del sufijo
        network = file.suffix.lstrip('.')
        df['network_type'] = network
        
        # Convert ttlb_ms to latency_s for compatibility
        if 'ttlb_ms' in df.columns:
            df['latency_s'] = df['ttlb_ms'] / 1000.0
        
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Cargados {len(combined)} registros individuales")
    return combined


def plot_latency_boxplot(df_raw: pd.DataFrame, df_summary: pd.DataFrame, output_dir: Path, fmt: str = 'png'):
    """
    Fig 1b: Boxplot con distribución de latencias individuales por configuración.
    Añade anotaciones de jitter (STD e IQR) tomadas de df_summary para cada configuración.

    Args:
        df_raw: DataFrame con datos individuales
        df_summary: DataFrame con datos agregados (contiene jitter_std/jitter_iqr por config)
        output_dir: Directorio de salida
        fmt: Formato de archivo
    """
    if df_raw.empty:
        print("⚠️  Omitiendo Fig 1b: No hay datos raw disponibles")
        return

    print("\n📈 Generando Fig 1b: Distribución de latencias (boxplot) por configuración...")

    # Create a combined column to order: model + pipeline + network
    df_raw['full_config'] = df_raw.apply(
        lambda row: (row['model'], row['pipeline'], row['network']),
        axis=1
    )

    # Sort by model, then pipeline, then network
    unique_configs = sorted(df_raw['full_config'].unique())

    # Prepare data, labels, colors, and jitter
    data_by_config = []
    labels = []
    colors_list = []
    jitter_std_list = []
    jitter_iqr_list = []

    for model, pipeline, network in unique_configs:
        mask = (df_raw['model'] == model) & (df_raw['pipeline'] == pipeline) & (df_raw['network'] == network)
        data = df_raw[mask]['latency_s'].values
        if len(data) > 0:
            data_by_config.append(data)
            labels.append(f"{pipeline.upper()}\n{network.upper()}")
            colors_list.append(MODEL_COLORS.get(model, '#888888'))

            # Buscar jitter en df_summary (matching by model/pipeline/network)
            row = df_summary[
                (df_summary['model'] == model) &
                (df_summary['pipeline'] == pipeline) &
                (df_summary['network'] == network)
            ]
            if not row.empty:
                jitter_std_list.append(row.iloc[0].get('jitter_std', np.nan))
                jitter_iqr_list.append(row.iloc[0].get('jitter_iqr', np.nan))
            else:
                jitter_std_list.append(np.nan)
                jitter_iqr_list.append(np.nan)

    fig, ax = plt.subplots(figsize=(16, 7))

    bp = ax.boxplot(
        data_by_config,
        labels=labels,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=7),
        widths=0.6
    )

    # Color the boxes according to the model
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add visual separators between models
    prev_model = None
    for i, (model, pipeline, network) in enumerate(unique_configs):
        if prev_model is not None and prev_model != model:
            ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        prev_model = model

    # Calculate the upper limit of the graph to position the jitter consistently
    all_data = np.concatenate(data_by_config)
    y_max = np.percentile(all_data, 95)  # Usar percentil 95 como referencia
    
    # Add jitter annotations per box (STD and IQR) - compact and elegant style
    for i, (std, iqr) in enumerate(zip(jitter_std_list, jitter_iqr_list)):
        x = i + 1
        # Posicionar justo debajo del bigote inferior del boxplot
        whisker_low = bp['caps'][i*2].get_ydata()[0]
        y_pos = whisker_low

        if not np.isnan(std) and not np.isnan(iqr):
            # Formato compacto en dos líneas (más legible)
            label = f"σ={std:.2f}s\nIQR={iqr:.2f}s"
            ax.annotate(
                label,
                xy=(x, y_pos),
                xytext=(0, -8),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=10,
                color='#222222',
                bbox=dict(
                    boxstyle='round,pad=0.45', 
                    facecolor='#f8f8f8', 
                    edgecolor='#666666',
                    linewidth=0.9,
                    alpha=0.5
                )
            )

    ax.set_xlabel('Configuration (Pipeline + Network)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Latency (seconds)', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=0, fontsize=12)
    
    # Ajustar límites del eje Y para dar espacio a las anotaciones inferiores
    y_min, y_max_current = ax.get_ylim()
    ax.set_ylim(-3, y_max_current * 1.05)  # Empezar en -3 pero mostrar desde 0
    
    # Configure ticks to start at 0
    y_ticks = ax.get_yticks()
    y_ticks = y_ticks[y_ticks >= 0]  # Filtrar solo valores >= 0
    ax.set_yticks(y_ticks)

    # Improved legend
    legend_elements = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], alpha=0.7, edgecolor='black', label=MODEL_NAMES[m])
        for m in ['gemma', 'llama3', 'zephyr'] if m in df_raw['model'].values
    ]
    legend_elements.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10, label='Mean')
    )
    # Añadir explicación del jitter en la leyenda con símbolo sigma
    legend_elements.append(
        mpatches.Patch(facecolor='#f0f0f0', edgecolor='#666666', linewidth=0.8, label='Jitter (σ, IQR)')
    )
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=12)

    plt.tight_layout()
    output_file = output_dir / f'fig1b_latency_boxplot.{fmt}'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def plot_correctness_bars(df_summary: pd.DataFrame, output_dir: Path, fmt: str = 'png'):
    """
    Fig 1c (versión combinada): Radar chart comparativo de métricas de correctitud
    para todos los modelos (Gemma, LLaMA3, Zephyr, etc.) en un solo gráfico.
    
    Args:
        df_summary: DataFrame con datos de resumen
        output_dir: Directorio de salida
        fmt: Formato de archivo ('png', 'pdf', etc.)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    print("\n📈 Generando Fig 1c: Métricas de correctitud (radar combinado)...")

    # Verify columns
    correctness_cols = ['ok_parse_rate', 'class_accuracy', 'mean_similarity', 'key_match_rate']
    if not all(col in df_summary.columns for col in correctness_cols):
        print("⚠️  Omitiendo Fig 1c: No hay datos de correctitud disponibles")
        return

    # Normalize model names
    df_summary['config'] = df_summary['model'].apply(lambda m: MODEL_NAMES.get(m, m))

    # Define labels and angular configuration
    labels = ['Parse Rate', 'Accuracy', 'Similarity', 'Key Match']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Paleta de colores (personalizable)
    colors = {
        'gemma': '#3498db',   # azul
        'llama3': '#e67e22',  # naranja
        'zephyr': '#2ecc71',  # verde
    }

    # Crear figura
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.set_ylim(0, 100)

    # Draw each model
    for model_name in df_summary['config'].unique():
        subset = df_summary[df_summary['config'] == model_name].mean(numeric_only=True)
        values = [
            subset['ok_parse_rate'] * 100,
            subset['class_accuracy'] * 100,
            subset['mean_similarity'] * 100,
            subset['key_match_rate'] * 100
        ]
        values += values[:1]

        color = colors.get(model_name.lower(), None)
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=model_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Estilo y leyenda
    ax.set_title("Fig 1c: Correctness Metrics Comparison", pad=20, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)

    plt.tight_layout()
    output_file = output_dir / f"fig1c_correctness_radar_combined.{fmt}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close(fig)




def plot_latency_vs_correctness(df_summary: pd.DataFrame, output_dir: Path, fmt: str = 'png'):
    """
    Fig 1d: Scatter plot de latencia vs métricas de correctitud promediadas por modelo.
    Agrupa todas las configuraciones de cada modelo y calcula la media de:
    - Latencia (mean_latency_s o ttlb_p50_ms)
    - Parse rate (ok_parse_rate o parse_rate)
    - Accuracy (class_accuracy o accuracy)
    - Similarity (mean_similarity o avg_similarity)
    - Key match (key_match_rate o avg_key_match)
    
    Args:
        df_summary: DataFrame con datos de resumen por configuración
        output_dir: Directorio de salida
        fmt: Formato de archivo
    """
    # Check necessary columns (with alternative names)
    required_metrics = {
        'latency': ['mean_latency_s', 'ttlb_p50_ms'],
        'parse_rate': ['ok_parse_rate', 'parse_rate'],
        'accuracy': ['class_accuracy', 'accuracy'],
        'similarity': ['mean_similarity', 'avg_similarity'],
        'key_match': ['key_match_rate', 'avg_key_match']
    }
    
    # Detectar qué columnas existen
    cols_found = {}
    for metric, possible_cols in required_metrics.items():
        for col in possible_cols:
            if col in df_summary.columns:
                cols_found[metric] = col
                break
    
    if 'similarity' not in cols_found or 'latency' not in cols_found:
        print("⚠️  Omitiendo Fig 1d: No hay datos suficientes de latencia o similarity")
        return
    
    print("\n📈 Generando Fig 1d: Latencia vs Correctitud (promediado por modelo)...")
    
    # Agrupar por modelo y calcular promedios
    model_stats = []
    for model in df_summary['model'].unique():
        df_model = df_summary[df_summary['model'] == model]
        
        # Calcular latencia promedio (convertir a segundos si es necesario)
        if cols_found['latency'] == 'ttlb_p50_ms':
            avg_latency = df_model['ttlb_p50_ms'].mean() / 1000.0
        else:
            avg_latency = df_model['mean_latency_s'].mean()
        
        # Calcular promedios de métricas de correctitud (convertir a %)
        metrics = {'model': model, 'avg_latency_s': avg_latency}
        
        for metric, col in cols_found.items():
            if metric != 'latency':
                # Si los valores están en [0,1], convertir a %
                val = df_model[col].mean()
                if val <= 1.0:
                    val *= 100
                metrics[f'avg_{metric}_pct'] = val
        
        model_stats.append(metrics)
    
    df_agg = pd.DataFrame(model_stats)
    
    # Create a unique figure combining all metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    correctness_metrics = [
        ('avg_parse_rate_pct', 'Parse Rate', 'o'),      # círculo
        ('avg_accuracy_pct', 'Accuracy', 's'),          # cuadrado
        ('avg_similarity_pct', 'Similarity', '^'),      # triángulo
        ('avg_key_match_pct', 'Key Match', 'D')         # diamante
    ]
    
    # Plot each combination of model + metric
    for metric_col, metric_label, marker in correctness_metrics:
        if metric_col not in df_agg.columns:
            continue
        
        for _, row in df_agg.iterrows():
            model = row['model']
            model_name = MODEL_NAMES.get(model, model)
            
            ax.scatter(
                row['avg_latency_s'],
                row[metric_col],
                s=300,
                c=MODEL_COLORS.get(model, '#888888'),
                marker=marker,
                alpha=0.8,
                edgecolors='black',
                linewidth=2,
                zorder=3,
                label=f"{model_name} - {metric_label}"
            )
            
            # Add a compact label near the point
            ax.annotate(
                f"{model_name[:3]}\n{metric_label[:4]}",
                (row['avg_latency_s'], row[metric_col]),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=7,
                ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                         edgecolor='gray', linewidth=0.5, alpha=0.8)
            )
    
    ax.set_xlabel('Mean Latency (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Correctness Metrics (%)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create a combined legend: first models, then metrics
    # Leyenda de modelos (por color)
    model_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODEL_COLORS.get(m, '#888888'),
               markersize=10, markeredgecolor='black', markeredgewidth=1.5, label=MODEL_NAMES.get(m, m))
        for m in df_agg['model'].unique()
    ]
    
    # Leyenda de métricas (por forma)
    metric_handles = [
        Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', markeredgewidth=1.5, label=metric_label)
        for _, metric_label, marker in correctness_metrics
        if f'avg_{metric_label.lower().replace(" ", "_")}_pct' in df_agg.columns
    ]
    
    # Combinar leyendas
    first_legend = ax.legend(handles=model_handles, title='Models', 
                            loc='upper left', framealpha=0.95, fontsize=10)
    ax.add_artist(first_legend)
    ax.legend(handles=metric_handles, title='Metrics', 
             loc='upper right', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / f'fig1d_latency_vs_correctness.{fmt}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    print(f"   Modelos analizados: {', '.join(df_agg['model'].tolist())}")
    plt.close()


def plot_jitter_comparison(df_summary: pd.DataFrame, output_dir: Path, fmt: str = 'png'):
    """
    Fig 1e (opcional): Comparación de jitter (STD e IQR).
    
    Args:
        df_summary: DataFrame con datos de resumen
        output_dir: Directorio de salida
        fmt: Formato de archivo
    """
    if 'jitter_std' not in df_summary.columns or 'jitter_iqr' not in df_summary.columns:
        print("⚠️  Omitiendo Fig 1e: No hay datos de jitter disponibles")
        return
    
    print("\n📈 Generando Fig 1e: Comparación de jitter...")
    
    # Create labels: Pipeline + Network
    df_summary['config'] = df_summary.apply(
        lambda row: f"{row['pipeline'].upper()}\n{row['network'].upper()}", 
        axis=1
    )
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(df_summary))
    width = 0.35  # Ancho de cada barra
    colors = [MODEL_COLORS.get(m, '#888888') for m in df_summary['model']]
    
    # Grouped bars: STD and IQR
    bars1 = ax.bar(x_pos - width/2, df_summary['jitter_std'], width, 
                   label='Standard Deviation', color=colors, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, df_summary['jitter_iqr'], width,
                   label='IQR', color=colors, alpha=0.5, edgecolor='black', hatch='//')
    
    # Labels on the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Solo mostrar si es significativo
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=7
                )
    
    ax.set_xlabel('Configuration (Pipeline + Network)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Jitter (seconds)', fontweight='bold', fontsize=12)
    ax.set_title('Fig 1e: Jitter Analysis (Latency Variability)', fontweight='bold', pad=20, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_summary['config'], fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Combined legend: metrics + models
    metric_legend = ax.legend(loc='upper left', framealpha=0.9, title='Metric')
    
    # Añadir leyenda de modelos
    model_legend_elements = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor='black', alpha=0.8, label=MODEL_NAMES[m])
        for m in ['gemma', 'llama3', 'zephyr'] if m in df_summary['model'].values
    ]
    ax.legend(handles=model_legend_elements, loc='upper right', framealpha=0.9, title='Model')
    ax.add_artist(metric_legend)  # Restaurar la primera leyenda
    plt.tight_layout()
    output_file = output_dir / f'fig1e_jitter_comparison.{fmt}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generador de gráficos para Experimento E1 (Latencia)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('results/latency'),
        help='Directorio con archivos CSV de entrada (default: results/latency)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/latency'),
        help='Directorio para guardar gráficos (default: results/latency)'
    )
    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Formato de salida (default: png)'
    )
    parser.add_argument(
        '--skip-raw',
        action='store_true',
        help='Omitir gráficos que requieren datos raw (boxplot)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"❌ Error: El directorio {args.input_dir} no existe")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 GENERADOR DE GRÁFICOS - EXPERIMENTO E1 (LATENCIA)")
    print("=" * 70)
    print(f"📁 Directorio de entrada: {args.input_dir.absolute()}")
    print(f"📁 Directorio de salida:  {args.output_dir.absolute()}")
    print(f"🎨 Formato de gráficos:   {args.format.upper()}")
    print("=" * 70)
    
    try:
        # Load data
        df_summary = load_summary_data(args.input_dir)
        df_raw = pd.DataFrame() if args.skip_raw else load_raw_data(args.input_dir)
        
        # Generate graphs
        # Main boxplot per configuration (uses raw + summary data for jitter)
        if not args.skip_raw and not df_raw.empty:
            plot_latency_boxplot(df_raw, df_summary, args.output_dir, args.format)
        
        plot_correctness_bars(df_summary, args.output_dir, args.format)
        plot_latency_vs_correctness(df_summary, args.output_dir, args.format)
        plot_jitter_comparison(df_summary, args.output_dir, args.format)
        
        print("\n" + "=" * 70)
        print("✅ GENERACIÓN COMPLETADA")
        print("=" * 70)
        print(f"📊 Gráficos guardados en: {args.output_dir.absolute()}")
        print("\nGráficos generados:")
        print(f"  - fig1b_latency_boxplot.{args.format}      : Distribución de latencias por configuración")
        print(f"  - fig1c_correctness_bars.{args.format}     : Métricas de correctitud")
        print(f"  - fig1d_latency_vs_correctness.{args.format}: Trade-off latencia/correctitud")
        print(f"  - fig1e_jitter_comparison.{args.format}    : Análisis de jitter (por configuración)")
        
    except Exception as e:
        print(f"\n❌ Error durante la generación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

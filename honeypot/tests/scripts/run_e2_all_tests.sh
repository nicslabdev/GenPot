#!/bin/bash
# Script para ejecutar todos los tests E2 para diferentes modelos y pipelines

set -e

echo "=========================================="
echo "E2 LOAD TESTS - BATCH EXECUTION"
echo "=========================================="
echo ""

# Configuración
MODELS=("gemma" "llama3" "zephyr")
PIPELINES=("api" "system")
CONCURRENCY="2,4,8,16,32"
DURATION=120
WARMUP=20
COOLDOWN=10

# Crear directorio de resultados
mkdir -p results/load

# Ejecutar tests para cada combinación
for model in "${MODELS[@]}"; do
    for pipeline in "${PIPELINES[@]}"; do
        echo ""
        echo "=========================================="
        echo "Testing: $model - $pipeline"
        echo "=========================================="
        
        # Nombrar el archivo de salida
        OUTPUT_PREFIX="results/load/${model}_${pipeline}"
        
        # Ejecutar test
        python run_e2_load.py \
            --model "$model" \
            --pipeline "$pipeline" \
            --concurrency-levels "$CONCURRENCY" \
            --duration-per-level "$DURATION" \
            --warmup-duration "$WARMUP" \
            --cooldown "$COOLDOWN"
        
        # Renombrar archivos de salida para esta configuración
        mv results/load/load_raw.csv "${OUTPUT_PREFIX}_raw.csv"
        mv results/load/load_summary.csv "${OUTPUT_PREFIX}_summary.csv"
        cp results/load/hardware_info.txt "${OUTPUT_PREFIX}_hardware.txt"
        
        echo "✓ Completed: $model - $pipeline"
        echo "  Files saved:"
        echo "    - ${OUTPUT_PREFIX}_raw.csv"
        echo "    - ${OUTPUT_PREFIX}_summary.csv"
        echo "    - ${OUTPUT_PREFIX}_hardware.txt"
        
        # Espera entre tests
        echo ""
        echo "Waiting 30s before next test..."
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "✅ ALL TESTS COMPLETED"
echo "=========================================="
echo ""
echo "Combining all summaries into one file..."

# Combinar todos los summaries en un solo archivo
echo "model,pipeline,concurrency,duration_sec,total_requests,rps,error_rate,timeout_rate,latency_p50_ms,latency_p95_ms,latency_p99_ms,latency_mean_ms,latency_std_ms,queueing_delay_ms" > results/load/combined_summary.csv

for model in "${MODELS[@]}"; do
    for pipeline in "${PIPELINES[@]}"; do
        tail -n +2 "results/load/${model}_${pipeline}_summary.csv" >> results/load/combined_summary.csv
    done
done

echo "✓ Combined summary saved: results/load/combined_summary.csv"
echo ""
echo "Generate graphs with:"
echo "  python plot_e2_graphs.py --summary-csv results/load/combined_summary.csv --extra-plots"
echo ""

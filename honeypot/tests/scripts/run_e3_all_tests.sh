#!/bin/bash
# Script para ejecutar todos los tests E3 Resource Profile

set -e

echo "=========================================="
echo "E3 RESOURCE PROFILE - BATCH EXECUTION"
echo "=========================================="
echo ""

# Configuración
MODELS=("llama3" "zephyr" "gemma")
PIPELINES=("system")  # Puede añadir "system" si es necesario
DURATION=3600
TARGET_RPS=1
WARMUP=300

# Crear directorio de resultados
mkdir -p results/resources/individual

echo "Test Configuration:"
echo "  Models: ${MODELS[*]}"
echo "  Pipelines: ${PIPELINES[*]}"
echo "  Duration: ${DURATION}s"
echo "  Target RPS: ${TARGET_RPS}"
echo ""

# Contador de tests
test_count=0
total_tests=$((${#MODELS[@]} * ${#PIPELINES[@]}))

# Ejecutar tests para cada combinación
for model in "${MODELS[@]}"; do
    for pipeline in "${PIPELINES[@]}"; do
            test_count=$((test_count + 1))
            
            echo ""
            echo "=========================================="
            echo "Test $test_count/$total_tests: $model - $pipeline"
            echo "=========================================="
            
            # Ejecutar test (con -u para unbuffered output)
            python -u run_e3_resources.py \
                --model "$model" \
                --pipeline "$pipeline" \
                --duration "$DURATION" \
                --target-rps "$TARGET_RPS" \
                --warmup "$WARMUP" \
                --num-workers 2 \
                --gpu-id 3
            
            # Mover archivos con nombres descriptivos
            timestamp=$(date +%Y%m%d_%H%M%S)
            mv results/resources/resources_raw.csv \
               "results/resources/individual/${model}_${pipeline}_raw.csv"
            mv results/resources/resources_summary.csv \
               "results/resources/individual/${model}_${pipeline}_summary.csv"
            cp results/resources/system_info.txt \
               "results/resources/individual/${model}_${pipeline}_system.txt"

            echo "✓ Completed: $model - $pipeline"
            echo "  Files saved to: results/resources/individual/"
            
            # Espera entre tests para enfriar el sistema
            if [ $test_count -lt $total_tests ]; then
                echo ""
                echo "Cooldown: 30 seconds..."
                sleep 30
            fi
        
    done
done

echo ""
echo "=========================================="
echo "✅ ALL E3 TESTS COMPLETED"
echo "=========================================="
echo ""
echo "Combining all summaries into one file..."

# Combinar todos los summaries en un solo archivo
echo "model,pipeline,duration_sec,target_rps,num_samples,cpu_percent_mean,cpu_percent_peak,cpu_percent_std,ram_used_gb_mean,ram_used_gb_peak,ram_percent_mean,ram_percent_peak,gpu_util_percent_mean,gpu_util_percent_peak,vram_used_gb_mean,vram_used_gb_peak,gpu_temp_c_mean,gpu_temp_c_peak,gpu_power_w_mean,gpu_power_w_peak,total_requests,successful_requests,failed_requests,success_rate,actual_rps,avg_latency_ms,test_timestamp" > results/resources/resources_summary.csv

for file in results/resources/individual/*_summary.csv; do
    # Saltar el header de cada archivo
    tail -n +2 "$file" >> results/resources/resources_summary.csv
done

echo "✓ Combined summary saved: results/resources/resources_summary.csv"
echo ""

echo "Combining all raw data into one file..."
# Obtener el header del primer archivo raw
first_raw=$(ls results/resources/individual/*_raw.csv | head -1)
head -n 1 "$first_raw" > results/resources/resources_raw.csv

# Agregar datos de todos los archivos (sin headers)
for file in results/resources/individual/*_raw.csv; do
    tail -n +2 "$file" >> results/resources/resources_raw.csv
done

echo "✓ Combined raw data saved: results/resources/resources_raw.csv"
echo ""
echo "Generate comparison graphs with:"
echo "  python plot_e3_graphs.py \\"
echo "    --summary-csv results/resources/resources_summary.csv \\"
echo "    --extra-plots"
echo ""
echo "Summary table:"
echo "----------------------------------------"
column -t -s',' results/resources/resources_summary.csv | head -20
echo ""




#!/bin/bash

API_ENDPOINT="http://192.168.43.171:5555/webapi/entry.cgi"
MODELS=("gemma" "llama3" "zephyr")
OUTPUT_DIR="results"
CSV_FILE="$OUTPUT_DIR/tiempos_respuesta.csv"

declare -a REQUESTS=(
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=password123"
  "api=SYNO.FileStation.List&method=list_share&version=2"
  "api=SYNO.Core.System&method=info&version=1"
  "api=SYNO.DownloadStation.Task&method=list&version=1"
  "api=SYNO.FileStation.List&method=list&version=2&folder_path=/volume1"
  "api=SYNO.FileStation.Info&method=get&version=2&path=/volume1/public"
  "api=SYNO.Core.User&method=get&version=1&user_name=admin"
  "api=SYNO.Core.User&method=logout&version=1&user_name=admin"
  "api=SYNO.Core.System.Utilization&method=get&version=1"
  "api=SYNO.Core.System.Status&method=network_status&version=1"
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=1234"
  "api=SYNO.API.Auth&method=logout&version=6&session=DownloadStation"
  "api=SYNO.DownloadStation.Task&method=create&version=1&uri=http://example.com/file.iso"
  "api=SYNO.DownloadStation.Task&method=delete&version=1&id=taskid_123"
  "api=SYNO.Core.Network&method=list&version=1"
  "api=SYNO.Core.Storage.Volume&method=status&version=1"
  "api=SYNO.Core.ExternalDevice&method=list&version=1"
  "api=SYNO.Core.Time&method=get&version=1"
  "api=SYNO.FakeModule.Bogus&method=nonexistent&version=1"
  "api=SYNO.Core.System&method=info&version=999"
  "api=SYNO.Core.User&method=delete&version=1&user_name="
  "api=SYNO.Core.User&metod=login&version=1"
)

mkdir -p "$OUTPUT_DIR"
echo "n_peticion,slug_peticion,modelo,tiempo_segundos,power_before_w,temperature_before_c,power_after_w,temperature_after_c" > "$CSV_FILE"

i=0
for REQ in "${REQUESTS[@]}"; do
  ((i++))
  SLUG=$(echo "$REQ" | sed 's/[^a-zA-Z0-9]/_/g')

  echo -e "\n🧪 Request $i: $REQ"
  echo "------------------------------------------------------------"

  for MODEL in "${MODELS[@]}"; do
    echo "🔹 Modelo: $MODEL"
    mkdir -p "$OUTPUT_DIR/$MODEL"

    # Mide potencia y temperatura antes
    read POWER_BEFORE TEMP_BEFORE <<< $(nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits | head -n 1 | awk -F, '{print $1" "$2}')

    START=$(date +%s)
    echo "🌐 Ejecutando: curl --silent \"$API_ENDPOINT?$REQ&model_name=$MODEL\""
    RESPONSE=$(curl --silent "$API_ENDPOINT?$REQ&model_name=$MODEL")
    END=$(date +%s)
    DURATION=$((END - START))

    # Mide potencia y temperatura después
    read POWER_AFTER TEMP_AFTER <<< $(nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits | head -n 1 | awk -F, '{print $1" "$2}')

    FILENAME="$OUTPUT_DIR/$MODEL/${i}_${SLUG}.json"
    echo "🖨️  Respuesta recibida:"
    echo "$RESPONSE"
    echo "$RESPONSE" > "$FILENAME"

    echo "$i,$SLUG,$MODEL,$DURATION,$POWER_BEFORE,$TEMP_BEFORE,$POWER_AFTER,$TEMP_AFTER" >> "$CSV_FILE"

    echo "⏱️  Tiempo: ${DURATION}s | Power: ${POWER_BEFORE}W→${POWER_AFTER}W | Temp: ${TEMP_BEFORE}ºC→${TEMP_AFTER}ºC"
    echo "💾 Guardado en: $FILENAME"
    echo "------------------------------------------------------------"
  done
done

echo -e "\n✅ Pruebas completadas."
echo "📄 Resultados guardados en carpeta: $OUTPUT_DIR"
echo "📊 CSV de tiempos: $CSV_FILE"

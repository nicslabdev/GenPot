#!/bin/bash

PORT=5555
APP_PATH="fastapi_server"
LOGFILE="fastapi_server.log"

echo "🚀 Iniciando servidor FastAPI con todos los modelos..."

# Comprobar si el puerto ya está en uso
if lsof -i :$PORT >/dev/null 2>&1; then
    PID=$(lsof -t -i :$PORT)
    echo "❌ El puerto $PORT ya está en uso por el proceso PID $PID."
    read -p "¿Quieres matarlo automáticamente? (s/n): " confirm
    if [[ $confirm == "s" || $confirm == "S" ]]; then
        kill -9 $PID
        echo "✅ Proceso $PID detenido."
    else
        echo "⚠️  No se pudo iniciar el servidor. Usa otro puerto o libera el actual."
        exit 1
    fi
fi

# Lanzar el servidor FastAPI en background y guardar logs
export CUDA_VISIBLE_DEVICES=3  
nohup uvicorn $APP_PATH:app --host 0.0.0.0 --port $PORT --reload > "$LOGFILE" 2>&1 &

echo "✅ Servidor iniciado en background en el puerto $PORT. Logs en $LOGFILE"

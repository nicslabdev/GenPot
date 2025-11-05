#!/bin/bash

PORT=${1:-5555}  # Default port is 5555 if not provided
GPU_ID=${2:-0}  # Default GPU ID is 0 if not provided
if [[ ! $PORT =~ ^[0-9]+$ ]] || [ $PORT -lt 1024 ] || [ $PORT -gt 65535 ]; then
    echo "❌ Invalid port. Must be a number between 1024 and 65535."
    exit 1
fi
APP_PATH="fastapi_server"
LOGFILE="fastapi_server.log"

echo "🚀 Starting FastAPI server with all models..."

# Check if the port is already in use
if lsof -i :$PORT >/dev/null 2>&1; then
    PID=$(lsof -t -i :$PORT)
    echo "❌ Port $PORT is already in use by process PID $PID."
    read -p "Do you want to kill it automatically? (y/n): " confirm
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
        kill -9 $PID
        echo "✅ Process $PID stopped."
    else
        echo "⚠️  Could not start the server. Use another port or free the current one."
        exit 1
    fi
fi

# Launch the FastAPI server in background and save logs
export CUDA_VISIBLE_DEVICES=$GPU_ID  
nohup uvicorn $APP_PATH:app --host 0.0.0.0 --port $PORT --reload > "$LOGFILE" 2>&1 &

echo "✅ Server started in background on port $PORT. Logs in $LOGFILE"

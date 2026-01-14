#!/bin/bash
# GPU Monitoring Script
# Usage: ./monitor_gpu.sh [interval_seconds]

INTERVAL=${1:-2}  # Default 2 seconds

echo "🔍 Monitoring GPU Utilization (Press Ctrl+C to stop)"
echo "=================================================="
echo ""

while true; do
    clear
    echo "📊 GPU Status - $(date '+%H:%M:%S')"
    echo "=================================================="
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s (%s):\n", $1, $2; printf "  GPU Util: %s%%\n", $3; printf "  Memory Util: %s%% (%s MB / %s MB)\n", $4, $5, $6; printf "  Power: %s W\n", $7; printf "  Temp: %s°C\n\n", $8}'
    
    # Show process using GPU
    echo "🔧 Processes using GPU:"
    nvidia-smi pmon -c 1 -s u | head -10
    echo ""
    echo "Press Ctrl+C to stop..."
    sleep $INTERVAL
done

#!/bin/bash
# Check if video processing is still running

echo "=== Process Status ==="
PROCESS=$(ps aux | grep -E "python.*main.py" | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "❌ No process running"
else
    echo "✓ Process is running:"
    echo "$PROCESS"
fi

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== Latest Log Output ==="
if [ -f processing.log ]; then
    tail -10 processing.log
else
    echo "No log file found"
fi

echo ""
echo "=== Output Files ==="
ls -lh output_videos/*.mp4 2>/dev/null | tail -3 || echo "No output video yet"


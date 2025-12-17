#!/bin/bash
# Run video analysis with GPU, keeping connection alive

cd /home/essashah/Desktop/football_analysis

# Run in background with nohup to keep running even if SSH disconnects
nohup python3 main.py input_videos/youtube_test.mp4 > processing.log 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "GPU will be used automatically"
echo "Monitor progress with: tail -f processing.log"
echo "Check GPU usage with: watch -n 1 nvidia-smi"
echo ""
echo "To stop: kill $PID"


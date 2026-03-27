#!/bin/bash
cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

# 绝对干净的启动方式，严禁带 CUDA_VISIBLE_DEVICES
ISAAC_PYTHON="/home/yczheng/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"

$ISAAC_PYTHON -u $root_dir/src/eval_2.py \
    --maxActions 150 \
    --eval_save_path $root_dir/unfixed_logs/scene \
    --dataset_path $root_dir/DATASET/uavondataset.json \
    --is_fixed false \
    --gpu_id 0 \
    --batchSize 1 \
    --simulator_tool_port 30000
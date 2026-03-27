cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

ISAAC_PYTHON="/home/yczheng/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"

$ISAAC_PYTHON -u $root_dir/src/eval_cliph.py \
    --maxActions 150 \
    --eval_save_path $root_dir/CLIP_logs/scene \
    --dataset_path $root_dir/DATASET/uavondataset.json \
    --is_fixed  true\
    --gpu_id 0 \
    --batchSize 1 \
    --simulator_tool_port 30000
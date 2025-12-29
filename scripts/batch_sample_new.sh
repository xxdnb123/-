#!/usr/bin/env bash
# 仅做参数包装：把原来 sh 需要的 5 个参数在这里填好，然后原样调用原脚本。
# 不改任何逻辑、不加别的参数。

set -euo pipefail

# ==== 在这里填好原脚本需要的 5 个参数 ====
CONFIG_FILE="configs/sampling.yml"   # 1) 配置文件
RESULT_PATH="/home/user9/xxd_data/project/targetdiff-main/generate_res"                # 2) 输出目录
NODE_ALL=10                           # 3) 总 worker 数（单机单卡 = 1）
NODE_THIS=0                          # 4) 当前 worker 编号（0..NODE_ALL-1）
START_IDX=0                          # 5) 起始 data_id（通常 0）

# ==== 原始批处理脚本路径（按你的仓库路径调整，如有不同） ====
ORIGINAL_SH="scripts/batch_sample_diffusion.sh"

# ==== 直接把 5 个参数原样传给原脚本 ====
bash "$ORIGINAL_SH" "$CONFIG_FILE" "$RESULT_PATH" "$NODE_ALL" "$NODE_THIS" "$START_IDX"


python -m scripts.sample_diffusion configs/sampling.yml -i 0 --batch_size 50 --result_path outputs

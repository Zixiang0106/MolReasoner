#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH -o output/infer_all_back.out
#SBATCH -e output/infer_all_back.err
module load cuda/12.3.2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source activate llama

export WANDB_DISABLED="true"
export SWANLAB_MODE="disabled"
export DISABLE_VERSION_CHECK=1

# 需要跑的模型（不含 molt5 / scibert）
MODELS=(
    "Qwen3-8B"
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-7B"
    "Qwen2-7B-Instruct"
)

# 简单的模板选择：按模型家族挑一个你在 LLaMA-Factory 里可用的 template
pick_template () {
  local m="$1"
  if [[ "$m" == *"Qwen"* || "$m" == *"DeepSeek"* ]]; then
    echo "qwen"
  elif [[ "$m" == *"Llama-3"* || "$m" == *"Llama-3.1"* ]]; then
    echo "llama3"
  elif [[ "$m" == *"llama-2"* || "$m" == *"Llama-2"* ]]; then
    echo "chatml"   # 也可根据你数据改成 alpaca/chatml/llama2 等
  else
    echo "qwen"
  fi
}

mkdir -p output

for model in "${MODELS[@]}"; do
  tpl=$(pick_template "$model")

  echo ">>> Running inference for $model (template=$tpl)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/infer.py \
      --model_name_or_path "/mnt/ceph/users/zlu10/llm/models/$model" \
      --template "$tpl" \
      --dataset tox_test \
      --save_name "output/${model}_tox_test.json" \
      --batch_size 128 \
      --max_new_tokens 4096 \
      --preprocessing_num_workers 8 \
      --cutoff_len 4096
done
# 1) 进入你装 vLLM 的环境
conda activate infer

# 2) 确保没有加载系统 CUDA
module purge
unset LD_LIBRARY_PATH

# 3) 一些稳态开关 + 日志
export CUDA_VISIBLE_DEVICES=0          # 先单卡排错
export VLLM_LOG_LEVEL=DEBUG
export VLLM_TORCH_CUDA_GRAPH=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1               # 先禁用 IB，排除通讯干扰
export NCCL_P2P_DISABLE=0

# 4) 起 vLLM（保持你的参数）
python -m vllm.entrypoints.api_server \
  --model /mnt/ceph/users/zlu10/llm/MolReasoner/LLaMA-Factory/output \
  --tokenizer /mnt/ceph/users/zlu10/llm/MolReasoner/LLaMA-Factory/output \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85

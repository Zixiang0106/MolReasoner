#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 40:00:00
#SBATCH -C a100
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4 
#SBATCH --cpus-per-task=16
#SBATCH -o output/infer.out
#SBATCH -e output/infer.err
source activate llama
# pip install -e ".[torch,metrics]" --no-build-isolation
# pip3 install deepspeed

export WANDB_DISABLED="true"
export SWANLAB_MODE="disabled"
export DISABLE_VERSION_CHECK=1


CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/infer.py \
    --model_name_or_path ./caption \
    --template qwen \
    --dataset tox_test \
    --save_name output/tox_test.json \
    --batch_size 512 \
    --max_new_tokens 2048 \
    --preprocessing_num_workers 8 \
    --cutoff_len 4096 

# Note: Replace `xxxx.json` with your desired output file name.
# Note: Replace model_name_or_path with the path to your trained model.
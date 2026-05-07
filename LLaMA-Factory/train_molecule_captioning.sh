#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 40:00:00
#SBATCH -C a100
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4 
#SBATCH --cpus-per-task=16
#SBATCH -o output/sft.out
#SBATCH -e output/sft.err
source activate llama
#pip install -e ".[torch,metrics]" --no-build-isolation
#pip3 install deepspeed

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

export WANDB_DISABLED="true"
export SWANLAB_MODE="disabled"
llamafactory-cli train  /mnt/ceph/users/zlu10/llm/MolReasoner/LLaMA-Factory/examples/train_full/train_molecule_captioning/sft.yml
#CUDA_VISIBLE_DEVICfuES=0,1 FORCE_TORCHRUN=1 llamafactory-cli train  LLaMA-Factory/examples/train_full/train_molecule_captioning/sft.yml
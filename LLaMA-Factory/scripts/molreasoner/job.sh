#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
module load cuda/12.3.2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source activate llama
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u infer_molreasoner.py
#!/bin/bash

#SBATCH --job-name="torch_trainer_train"
#SBATCH -p batch
#SBATCH -o runjob.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH --cpus-per-gpu=2


# Transformer
model_config="Tfm"
optim_config="Tfm"
trainer_config="Tfm"


# Number of nodes
nnodes=1
# Number of tasks per node
nproc_per_node=2
run_ip=20000


# Optimizations
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=600
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun TTVenv/bin/torchrun \
    --nnodes $nnodes \
    --nproc_per_node $nproc_per_node \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$run_ip \
    src/train.py \
    --model_config_name $model_config \
    --optim_config_name $optim_config \
    --trainer_config_name $trainer_config

# # cp reload
# srun TTVenv/bin/torchrun \
#     --nnodes $nnodes \
#     --nproc_per_node $nproc_per_node \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $head_node_ip:$run_ip \
#     src/train_from_cp.py \
#     --model_save_path "model_ckpts/Tfm/" \
#     --step "7500"

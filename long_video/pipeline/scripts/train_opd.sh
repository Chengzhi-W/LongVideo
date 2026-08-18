#!/bin/bash

RANK=${RANK:=0}
PORT=${MASTER_PORT:-"19002"}
MASTER_ADDR=${MASTER_ADDR:-"29.232.224.126"}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

cd /apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing
export https_proxy="http://star-proxy.oa.com:3128"
pip install wandb==0.24.0
pip install diffusers==0.31.0
pip install lmdb
pip install easydict
pip install ftfy
pip install omegaconf

export PYTHONPATH='/apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing'
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment}"
export WANDB_API_KEY
wandb online

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    train.py \
    --config_path configs/causal_forcing_opd_chunkwise.yaml \
    --logdir logsv1/causal_forcing_opd_chunkwise

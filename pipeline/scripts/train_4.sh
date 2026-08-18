#!/bin/bash

RANK=${RANK:=4}
PORT=${MASTER_PORT:-"19002"}
MASTER_ADDR=${MASTER_ADDR:-"29.232.243.152"}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

cd /apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing
export https_proxy="http://star-proxy.oa.com:3128"
pip install wandb==0.24.0
pip install diffusers==0.31.0
pip install lmdb

export PYTHONPATH='/apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing'
# 原始transformers版本是5.5.4
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment}"
export WANDB_API_KEY
# wandb online
wandb online
# Launch command (parameters automatically read from accelerate_multi_node.yaml)


torchrun \
    --nproc_per_node=8 \
    --nnodes=8 \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    train.py \
    --config_path configs/causal_forcing_dmd_chunkwise.yaml \
    --logdir logs/causal_forcing_dmd_fakear_chunkwise
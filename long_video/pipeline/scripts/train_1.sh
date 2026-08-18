#!/bin/bash

RANK=${RANK:=1}
PORT=${MASTER_PORT:-"19002"}
MASTER_ADDR=${MASTER_ADDR:-"29.191.208.44"}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

cd /apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing
export https_proxy="http://star-proxy.oa.com:3128"
pip install wandb==0.24.0
pip install diffusers==0.31.0
pip install lmdb
pip install easydict
pip install ftfy
pip install omegaconf
# pip install /apdcephfs_gy31/share_301869871/taiji/xxuanhe/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

export PYTHONPATH='/apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing'
# 原始transformers版本是5.5.4
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment}"
export WANDB_API_KEY
# wandb online
wandb online
# wandb offline
# Launch command (parameters automatically read from accelerate_multi_node.yaml)


torchrun \
    --nproc_per_node=8 \
    --nnodes=3 \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    train.py \
    --config_path configs/causal_forcing_opd_chunkwise.yaml \
    --logdir logs-stage1.5/causal_forcing_opd_chunkwise
#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file accelerate_config.yaml train.py --dataset OREGANO --embedding-dim 256 --hidden-size 512 \
    --num-layers 6 --batch-size 1024 --lr 5e-4 --dropout 0.1 --num-epoch 30 --save-dir "model_1" \
    --no-filter-gen --label-smooth 0.25 --encoder --save-interval 2 --l-punish --trainset "6_rev_rule" \
    --prob 0.15 --beam-size 256 --test-batch-size 8 --warmup 3 --iter
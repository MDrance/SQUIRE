#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

accelerate launch --config_file accelerate_config.yaml train.py --test --dataset FB15K237 \
	--beam-size 512 --save-dir "model_1" --ckpt "ckpt_30.pt" \
	--test-batch-size 8 --encoder --l-punish --no-filter-gen --self-consistency
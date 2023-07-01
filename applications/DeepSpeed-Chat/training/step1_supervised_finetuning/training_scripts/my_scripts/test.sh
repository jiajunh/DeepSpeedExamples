#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
    --model_name_or_path bigscience/bloom-560m \
    --per_device_train_batch_size 2　\
    --per_device_eval_batch_size 2 \
    --max_seq_len 512 \
    --gradient_accumulation_steps 16 \
    --lora_dim 128 \
    --zero_stage $ZERO_STAGE \
    --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

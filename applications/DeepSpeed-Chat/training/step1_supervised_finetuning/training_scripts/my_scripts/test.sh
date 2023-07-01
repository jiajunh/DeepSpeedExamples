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
    --model_name_or_path facebook/opt-350m \
    --per_device_train_batch_size 8　\
    --per_device_mini_train_batch_size=8 \
    --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
    --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

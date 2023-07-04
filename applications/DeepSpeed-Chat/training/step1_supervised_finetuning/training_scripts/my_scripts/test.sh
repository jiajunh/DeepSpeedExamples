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
    # --model_name_or_path bigscience/bloom-560m \
    # --per_device_train_batch_size 2　\
    # --per_device_eval_batch_size 2 \
    # --max_seq_len 512 \
    # --learning_rate 10e-6 \
    # --weight_decay 0. \
    # --gradient_accumulation_steps 1 \
    # --offload True \
    # --lora_dim 128 \
    # --zero_stage $ZERO_STAGE \
    # --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
    --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path bigscience/bloom-560m \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 256 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

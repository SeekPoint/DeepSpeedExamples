#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --hostfile /hostfile --master_addr=192.168.1.7 --master_port=29701 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-125m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 128 \
   --per_device_eval_batch_size 128 \
   --max_seq_len 128 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --disable_dropout \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
    2>&1 | tee $OUTPUT/training-ph2-2nodes2gpus-125m-2ndnode-ZERO_STAGE$ZERO_STAGE.log

#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0

python prompt_eval.py \
    --model_name_or_path_baseline facebook/opt-125mb \
    --model_name_or_path_finetune ../applications/DeepSpeed-Chat/output/actor-models/1.3b


#直接在
python3 training/step1_supervised_finetuning/prompt_eval.py \
--model_name_or_path_baseline ~/hf_model/opt-125m  \
--model_name_or_path_finetune output/actor-models/125m \
2>&1 | tee output/actor-models/125m/eval-ph1-1ode2gpus-opt125.log


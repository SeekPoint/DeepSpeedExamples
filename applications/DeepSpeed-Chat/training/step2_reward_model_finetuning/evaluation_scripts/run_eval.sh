#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
python  rw_eval.py \
    --model_name_or_path


=====#直接在
(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat$
python3 training/step2_reward_model_finetuning/rw_eval.py \
    --model_name_or_path_baseline ~/hf_model/opt-125m \
    --model_name_or_path_finetune output/reward-models/125m  \
    2>&1 | tee output/reward-models/125m/eval-ph2-1node2gpus-opt125.log
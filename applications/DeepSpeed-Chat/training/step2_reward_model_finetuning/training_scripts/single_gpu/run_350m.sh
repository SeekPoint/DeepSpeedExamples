#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# 这两行代码将从命令行中获取的前两个参数赋值给变量
OUTPUT=$1
ZERO_STAGE=$2

# 检查OUTPUT和ZERO_STAGE是否为空，如果为空，则为它们设置默认值。
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

# 创建一个名为OUTPUT的目录，如果该目录已经存在，-p 选项将防止报错。
mkdir -p $OUTPUT

# 使用deepspeed库来运行main.py
deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# �����д��뽫���������л�ȡ��ǰ����������ֵ������
OUTPUT=$1
ZERO_STAGE=$2

# ���OUTPUT��ZERO_STAGE�Ƿ�Ϊ�գ����Ϊ�գ���Ϊ��������Ĭ��ֵ��
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

# ����һ����ΪOUTPUT��Ŀ¼�������Ŀ¼�Ѿ����ڣ�-p ѡ���ֹ����
mkdir -p $OUTPUT

# ʹ��deepspeed��������main.py
deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

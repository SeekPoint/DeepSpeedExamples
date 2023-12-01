#================================================直接由ds启动！
#
#注意:
#1 对于action= store_XXX的参数
#    --gradient_checkpointing True
#    --disable_dropout False
#===不能填写 False、True true false
#
#2  --deepspeed  源码并没有这个参数, bash中有
#
#~/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning$

#===ph1===z0
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Yuan ke

set -x
#The set -x option instructs the shell to print each command or pipeline before executing it,
# preceded by a special prompt (usually +).

savefolder=~/yk_repo/ds/_log_tmps_/  #等号两边不能有空格！！！
#如果文件夹不存在，创建文件夹
if [ ! -d $savefolder ]; then
  mkdir $savefolder  #上面的路径不要加引号！！！
fi

binf="pytorch_model.bin"


cd ~/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
echo now in: $Cur_Dir

for ((i=0; i<=3; i++))
do

echo "===ph1===zero_stage$i"
deepspeed main.py \
   --data_path Dahoas/rm-static  \
   --data_split 2,4,4 \
   --data_output_path ~/ds_exp \
   --model_name_or_path /home/amd00/hf_model/opt-125m \
   --per_device_train_batch_size 13 \
   --per_device_eval_batch_size 3 \
   --max_seq_len 249 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2 \
   --disable_dropout \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 2 \
   --seed 1234 \
   --deepspeed \
   --gradient_checkpointing \
   --zero_stage $i \
   --lora_dim 19 \
   --output_dir ${savefolder}actor+opt125m+Z$i \
    2>&1 | tee ${savefolder}ph1_training-1node2gpus-Z$i.log

if [ ! -f ${savefolder}actor+opt125m+Z$i/$binf ]; then
  echo ${savefolder}actor+opt125m+Z$i/$binf not exist!
  exit 1
fi

done


cd ~/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning
echo now in: $Cur_Dir

for ((i=0; i<=3; i++))
do

echo "===ph2===zero_stage$i"

deepspeed main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path /home/amd00/hf_model/opt-125m \
   --num_padding_at_beginning 5 \
   --per_device_train_batch_size 9 \
   --per_device_eval_batch_size 7 \
   --max_seq_len 133 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --disable_dropout \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 2 \
   --seed 1234 \
   --zero_stage $i \
   --deepspeed \
   --gradient_checkpointing \
   --output_dir ${savefolder}reward+opt125m+Z$i \
   2>&1 | tee ${savefolder}ph2_training-1node2gpus-Z$i.log

if [ ! -f ${savefolder}reward+opt125m+Z$i/$binf ]; then
  echo ${savefolder}reward+opt125m+Z$i/$binf not exist!
  exit 1
fi

done


cd ~/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning
echo now in: $Cur_Dir

for ((i=0; i<=3; i++))
do
echo "===ph3===zero_stage$i"
deepspeed main.py \
 --data_path Dahoas/rm-static \
 --data_split 2,4,4 \
 --actor_model_name_or_path ${savefolder}actor+opt125m+Z$i \
 --critic_model_name_or_path ${savefolder}reward+opt125m+Z$i \
 --num_padding_at_beginning 1 \
 --per_device_train_batch_size 13 \
 --per_device_mini_train_batch_size 11 \
 --generation_batch_numbers 3 \
 --ppo_epochs 2 \
 --max_answer_seq_len 199 \
 --max_prompt_seq_len 211 \
 --actor_learning_rate 9.65e-6 \
 --critic_learning_rate 5e-6 \
 --num_train_epochs 2 \
 --lr_scheduler_type cosine \
 --gradient_accumulation_steps 2 \
 --disable_actor_dropout \
 --num_warmup_steps 5 \
 --deepspeed \
 --seed 1234 \
 --enable_hybrid_engine \
 --actor_zero_stage $i \
 --critic_zero_stage $i \
 --enable_ema \
 --preprocessing_num_workers 2 \
 --enable_hybrid_engine \
 --unpin_actor_parameters \
 --release_inference_cache \
 --inference_tp_size 1 \
 --tp_gather_partition_size 2 \
 --actor_gradient_checkpointing \
 --critic_gradient_checkpointing \
 --actor_lora_dim 23 \
 --critic_lora_dim 27 \
 --enable_ema \
 --output_dir ${savefolder}rlhf+opt125m+ac$i+cr$i+its1 \
 2>&1 | tee ${savefolder}ph3_training-1node2gpus-a$ic$i.log

#上面${savefolder} 必须要有{}  #只有z3才可以=2  注释不能写上上面 \之中！！！

if [ ! -f ${savefolder}rlhf+opt125m+ac$i+cr$i+its1/actor/$binf ]; then
  echo ${savefolder}rlhf+opt125m+ac$i+cr$i+its1/actor/$binf not exist!
  exit 1
fi

if [ ! -f ${savefolder}rlhf+opt125m+ac$i+cr$i+its1/actor_ema/$binf ]; then
  echo ${savefolder}rlhf+opt125m+ac$i+cr$i+its1/actor_ema/$binf not exist!
  exit 1
fi

if [ ! -f ${savefolder}rlhf+opt125m+ac$i+cr$i+its1/critic/$binf ]; then
  echo ${savefolder}rlhf+opt125m+ac$i+cr$i+its1/critic/$binf not exist!
  exit 1
fi
done

# its=2会导致出错！！！z0123
#echo "===ph3===zero_stage3="
#deepspeed main.py \
# --data_path Dahoas/rm-static \
# --data_split 2,4,4 \
# --actor_model_name_or_path ${savefolder}actor+opt125m+Z3 \
# --critic_model_name_or_path ${savefolder}reward+opt125m+Z3 \
# --num_padding_at_beginning 1 \
# --per_device_train_batch_size 13 \
# --per_device_mini_train_batch_size 11 \
# --generation_batch_numbers 3 \
# --ppo_epochs 2 \
# --max_answer_seq_len 199 \
# --max_prompt_seq_len 211 \
# --actor_learning_rate 9.65e-6 \
# --critic_learning_rate 5e-6 \
# --num_train_epochs 2 \
# --lr_scheduler_type cosine \
# --gradient_accumulation_steps 2 \
# --disable_actor_dropout \
# --num_warmup_steps 5 \
# --deepspeed \
# --seed 1234 \
# --enable_hybrid_engine \
# --actor_zero_stage 3 \
# --critic_zero_stage 3 \
# --enable_ema \
# --preprocessing_num_workers 2 \
# --enable_hybrid_engine \
# --unpin_actor_parameters \
# --release_inference_cache \
# --inference_tp_size 2 \
# --tp_gather_partition_size 2 \
# --actor_gradient_checkpointing \
# --critic_gradient_checkpointing \
# --actor_lora_dim 23 \
# --critic_lora_dim 27 \
# --enable_ema \
# --output_dir ${savefolder}rlhf+opt125m+ac3+cr3+its2 \
# 2>&1 | tee ${savefolder}ph3_training-1node2gpus-a3c3.log
#
#if [ ! -f ${savefolder}rlhf+opt125m+ac$i+cr$i+its2/$binf ]; then
#  echo ${savefolder}rlhf+opt125m+ac$i+cr$i/$binf not exist!
#  exit 1
#fi
#done

#for ((i=0; i<=3; i++))
#do
#  for ((j=0; j<=3; k++))
#  do
#  echo "===ph2===zero_stage===$i===$j"
#
#  done
#
#done
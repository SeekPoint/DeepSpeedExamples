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
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 128 \
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
   --lora_dim 16 \
   --output_dir ~/yk_repo/ds/_log_tmps_/actor+opt125m+Z$i \
    2>&1 | tee ~/yk_repo/ds/_log_tmps_/ph1_training-1node2gpus-Z$i.log

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
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 128 \
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
   --output_dir ~/yk_repo/ds/_log_tmps_/reward+opt125m+Z$i \
   2>&1 | tee ~/yk_repo/ds/_log_tmps_/ph2_training-1node2gpus-Z$i.log

done


cd ~/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning
echo now in: $Cur_Dir

for ((i=0; i<=3; i++))
do
echo "===ph3===zero_stage===$i="
deepspeed main.py \
 --data_path Dahoas/rm-static \
 --data_split 2,4,4 \
 --actor_model_name_or_path ~/yk_repo/ds/_log_tmps_/actor+opt125m+Z3 \
 --critic_model_name_or_path ~/yk_repo/ds/_log_tmps_/reward+opt125m+Z3 \
 --num_padding_at_beginning 1 \
 --per_device_train_batch_size 16 \
 --per_device_mini_train_batch_size 8 \
 --generation_batch_numbers 2 \
 --ppo_epochs 2 \
 --max_answer_seq_len 128 \
 --max_prompt_seq_len 128 \
 --actor_learning_rate 9.65e-6 \
 --critic_learning_rate 5e-6 \
 --num_train_epochs 2 \
 --lr_scheduler_type cosine \
 --gradient_accumulation_steps 2 \
 --disable_actor_dropout \
 --num_warmup_steps 10 \
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
 --actor_lora_dim 64 \
 --critic_lora_dim 32 \
 --enable_ema \
 --output_dir ~/yk_repo/ds/_log_tmps_/rlhf+opt125m+ac$i+cr$i \
 2>&1 | tee ~/yk_repo/ds/_log_tmps_/ph3_training-1node2gpus-a$ic$i.log
done

#for ((i=0; i<=3; i++))
#do
#  for ((j=0; j<=3; k++))
#  do
#  echo "===ph2===zero_stage===$i===$j"
#  deepspeed main.py \
#   --data_path Dahoas/rm-static \
#   --data_split 2,4,4 \
#   --actor_model_name_or_path ~/yk_repo/ds/_log_tmps_/actor+opt125m+Z3 \
#   --critic_model_name_or_path ~/yk_repo/ds/_log_tmps_/reward+opt125m+Z3 \
#   --num_padding_at_beginning 1 \
#   --per_device_train_batch_size 16 \
#   --per_device_mini_train_batch_size 8 \
#   --generation_batch_numbers 2 \
#   --ppo_epochs 2 \
#   --max_answer_seq_len 128 \
#   --max_prompt_seq_len 128 \
#   --actor_learning_rate 9.65e-6 \
#   --critic_learning_rate 5e-6 \
#   --num_train_epochs 2 \
#   --lr_scheduler_type cosine \
#   --gradient_accumulation_steps 2 \
#   --disable_actor_dropout \
#   --num_warmup_steps 10 \
#   --deepspeed \
#   --seed 1234 \
#   --enable_hybrid_engine \
#   --actor_zero_stage $i \
#   --critic_zero_stage $j \
#   --enable_ema \
#   --preprocessing_num_workers 2 \
#   --enable_hybrid_engine \
#   --unpin_actor_parameters \
#   --release_inference_cache \
#   --inference_tp_size 1 \
#   --tp_gather_partition_size 2 \
#   --actor_gradient_checkpointing \
#   --critic_gradient_checkpointing \
#   --actor_lora_dim 64 \
#   --critic_lora_dim 32 \
#   --enable_ema \
#   --output_dir ~/yk_repo/ds/_log_tmps_/rlhf+opt125m+ac$i+cr$j \
#   2>&1 | tee ~/yk_repo/ds/_log_tmps_/ph3_training-1node2gpus-a$ic$j.log
#
#  done
#
#done
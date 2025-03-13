#!/bin/bash
set -x  

# parameters
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH=4096
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=64
LEARNING_RATE=2e-6
WEIGHT_DECAY=0.001
NUM_EPOCHS=1
TRAIN_SET="raftstudy/ultrafeedback_pairs"
EVAL_SET="raftstudy/rmbench_eval"
OUTPUT_DIR="./models/llama3_ul_rm"
GRADIENT_CHECKPOINTING=True
OPTIMIZER="paged_adamw_32bit"
LR_SCHEDULER="cosine"
SAVE_STEPS=999999
EVAL_STEPS=200
PADDING_MODE="add_pad"
DEEPSEED_CONFIGS="./deepspeed_configs/deepspeed_3.json"

# running the script
accelerate launch ./train_bradley_terry_rm.py \
  model_name=$MODEL_NAME \
  max_length=$MAX_LENGTH \
  per_device_train_batch_size=$TRAIN_BATCH_SIZE \
  per_device_eval_batch_size=$EVAL_BATCH_SIZE \
  gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
  learning_rate=$LEARNING_RATE \
  weight_decay=$WEIGHT_DECAY \
  num_train_epochs=$NUM_EPOCHS \
  train_set_path=$TRAIN_SET \
  eval_set_path=$EVAL_SET \
  output_path=$OUTPUT_DIR \
  gradient_checkpointing=$GRADIENT_CHECKPOINTING \
  optim=$OPTIMIZER \
  lr_scheduler_type=$LR_SCHEDULER \
  save_every_steps=$SAVE_STEPS \
  eval_every_steps=$EVAL_STEPS \
  padding_mode=$PADDING_MODE \
  deepspeed=$DEEPSEED_CONFIGS

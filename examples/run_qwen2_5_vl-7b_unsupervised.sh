#!/bin/bash
set -x

# Unsupervised GRPO training with Cross-Attention based rewards
# This script uses SOTA score computed from attention maps as reward signal

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

FORMAT_PROMPT="\nPlease reason step by step, and put your final answer within \boxed{}."

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m verl.trainer.main \
    config=examples/config_unsupervised.yaml \
    data.train_files=MMR1/MMR1-Math-RL-Data-v0@train \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=16 \
    worker.reward.n=16 \
    worker.reward.use_unsupervised_reward=true \
    trainer.experiment_name=qwen2_5_vl_7b_unsupervised \
    trainer.n_gpus_per_node=8 \
    data.val_batch_size=500 \
    data.max_pixels=1204224 \
    trainer.total_episodes=15 \
    trainer.save_limit=7 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true

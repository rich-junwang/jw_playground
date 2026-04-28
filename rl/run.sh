#!/bin/zsh


# Training
torchrun --standalone --nproc_per_node=8 dpo/grpo_train_from_scratch.py

# Eval
python dpo/grpo_evaluation.py

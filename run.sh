#!/bin/bash

#conda activate py39

pretrained=sshleifer/tiny-gpt2
#pretrained=bigscience/bloom-560m
#pretrained=bigscience/bloom-7b1

task_alias=open_llm_vi
tasks=arc_vi,hellaswag_vi,mmlu_vi,truthfulqa_mc_vi
#tasks=arc_vi
#tasks=hellaswag_vi
#tasks=mmlu_vi
#tasks=truthfulqa_vi

python main.py \
    --model hf-auto \
    --model_alias gpt2 \
    --tasks ${tasks} \
    --task_alias ${task_alias} \
    --model_args pretrained=${pretrained}

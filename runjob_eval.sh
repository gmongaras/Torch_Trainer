#!/bin/bash

#SBATCH --job-name="torch_trainer_eval"
#SBATCH -p batch
#SBATCH -o runjob_eval.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH --cpus-per-gpu=2

model_name=model_ckpts/Tfm
eval_names=hellaswag,piqa,arc_easy,arc_challenge,race,humaneval,mbpp,triviaqa,nq_open
step=25_000

TTVenv/bin/python \
    src/eval.py \
    --model_save_path $model_name \
    --eval_names $eval_names \
    --step $step
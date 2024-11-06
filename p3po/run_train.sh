#!/bin/bash
#
# Usage:
# bash run_train.sh $task $num_tracked_points $hz

export HYDRA_FULL_ERROR=1

# ARGS:
task=$1
num_tracked_points=$2
hz=$3


if [[ $hz == 10 ]]; then
    minlength=34
elif [[ $hz == 15 ]]; then
    minlength=23
elif [[ $hz == 30 ]]; then
    minlength=12
else
    echo "hz=$hz not supported! exiting ..."
    exit
fi


p3po_task_name=$task
run_name="${task}_3d_abs_actions_minlength${minlength}_${hz}hz"
task="${task}_3d_abs_actions_minlength${minlength}_${hz}hz_closed_loop_dataset"

keypoints_type=3
point_dimensions=3
temporal_agg=true
history_len=10
gaussian_augmentation_std=0.02
num_rand_drop_history=3
num_train_steps=200010
save_every_steps=5000
delta_actions=false


# command
python train.py \
    agent=baku \
    suite=dexterous \
    dataloader=p3po_general \
    suite.hidden_dim=256 \
    suite.task.tasks=[$task] \
    suite.history_len=$history_len \
    suite.num_train_steps=$num_train_steps \
    suite.save_every_steps=$save_every_steps \
    point_dimensions=$point_dimensions \
    use_wandb=true \
    run_name=$run_name \
    p3po_task_name=$p3po_task_name \
    num_tracked_points=$num_tracked_points \
    dataloader.bc_dataset.gaussian_augmentation_std=$gaussian_augmentation_std \
    dataloader.bc_dataset.num_rand_drop_history=$num_rand_drop_history \
    temporal_agg=$temporal_agg \
    keypoints_type=$keypoints_type \
    delta_actions=$delta_actions

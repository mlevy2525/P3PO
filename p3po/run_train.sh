#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1


# pick_block_20241103a
# --------
p3po_task_name="pick_block_20241103a_30hz_nooccpoints"
run_name="pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz"
task="pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz_closed_loop_dataset"
num_tracked_points=8


# these probably won't change as much
unproject_depth=true
temporal_agg=true
point_dimensions=3
history_len=10
gaussian_augmentation_std=0.1
num_train_steps=150010
save_every_steps=5000


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
    temporal_agg=$temporal_agg \
    unproject_depth=$unproject_depth

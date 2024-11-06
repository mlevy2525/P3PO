#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1


# push_block_20241031c
# --------
# p3po_task_name="push_block_20241031c_30hz"
# task="push_block_20241031c_30hz_3d_abs_actions_minlength33_10hz_closed_loop_dataset"
# run_name="push_block_20241031c_10hz_3d_abs_actions_minlength33"
# num_tracked_points=8
# bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.02/183759_push_block_20241031c_10hz_3d_abs_actions_minlength33_3d/snapshot/250000.pt"

# open_drawer_20241101b
# --------
# p3po_task_name="open_drawer_20241103b_30hz"
# task="open_drawer_20241103b_30hz_3d_abs_actions_minlength33_10hz_closed_loop_dataset"
# run_name="open_drawer_20241103b_30hz_3d_abs_actions_minlength33_10hz"
# num_tracked_points=29
# bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.04/152317_open_drawer_20241103b_30hz_3d_abs_actions_minlength33_10hz_3d/snapshot/100000.pt"

# pick_block_20241103a
# --------
p3po_task_name="pick_block_20241103a_30hz_nooccpoints"
task="pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz_closed_loop_dataset"
run_name="pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz"
num_tracked_points=8
bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.05/154107_pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz_std0.1_repeatpad_3d/snapshot/100000.pt"


# these probably won't change as much
keypoints_type=3
temporal_agg=true
point_dimensions=3
history_len=10


# command
python train.py \
    agent=baku \
    suite=dexterous \
    dataloader=p3po_general \
    run_name=$run_name \
    p3po_task_name=$p3po_task_name \
    num_tracked_points=$num_tracked_points \
    bc_weight=$bc_weight \
    suite.hidden_dim=256 \
    suite.task.tasks=[$task] \
    suite.history_len=$history_len \
    point_dimensions=$point_dimensions \
    use_wandb=false \
    eval_only=true \
    load_bc=true \
    temporal_agg=$temporal_agg \
    keypoints_type=$keypoints_type \
    hydra.run.dir=/tmp/hydra-junk \
    hydra.sweep.dir=/tmp/hydra-junk \
    hydra.launcher.submitit_folder=/tmp/hydra-junk

#!/bin/bash


export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1


minlength=23
hz=15
# p3po_task_name="open_oven_20241107c"
# p3po_task_name="pick_block_20241107a"
p3po_task_name="collect_ball_20241107e"
task="${p3po_task_name}_3d_abs_actions_minlength${minlength}_${hz}hz_closed_loop_dataset"
run_name="${p3po_task_name}_3d_abs_actions_minlength${minlength}_${hz}hz"
num_tracked_points=10
# bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.08/140739_open_oven_20241107c_3d_abs_actions_minlength12_30hz_3d/snapshot/100000.pt"
# bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.08/140739_pick_block_20241107a_3d_abs_actions_minlength12_30hz_3d/snapshot/200000.pt"
# bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.08/152453_pick_block_20241107a_3d_abs_actions_minlength34_10hz_3d/snapshot/200000.pt"
# bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.08/161219_pick_block_20241107a_3d_abs_actions_minlength23_15hz_3d/snapshot/200000.pt"
bc_weight="/home/ademi/P3PO/p3po/exp_local/2024.11.08/161219_collect_ball_20241107e_3d_abs_actions_minlength23_15hz_3d/snapshot/200000.pt"

# these probably won't change as much
delta_actions=false
temporal_agg=true
history_len=10
keypoints_type=3
point_dimensions=3


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
    delta_actions=$delta_actions \
    hydra.run.dir=/tmp/hydra-junk \
    hydra.sweep.dir=/tmp/hydra-junk \
    hydra.launcher.submitit_folder=/tmp/hydra-junk

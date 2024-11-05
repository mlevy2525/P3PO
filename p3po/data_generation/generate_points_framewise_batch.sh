#!/bin/bash
#
# this script runs all jobs in the background! to kill these scripts, you must
# fetch their pids or kill them with something like `pkill python`


# pick_block_20241103a
# --------

CUDA_VISIBLE_DEVICES=0 python generate_points_framewise.py \
    --preprocessed_data_dir /home/ademi/hermes/data/pick_block_20241103a_preprocessed_30hz \
    --task_name pick_block_20241103a_30hz_nooccpoints \
    --num_tracked_points 8 \
    --unproject_depth &

CUDA_VISIBLE_DEVICES=1 python generate_points_framewise.py \
    --preprocessed_data_dir /home/ademi/hermes/data/pick_block_20241103a_preprocessed_30hz \
    --task_name pick_block_20241103a_30hz_nooccpoints \
    --num_tracked_points 8 &


# open_drawer_20241103b
# --------

CUDA_VISIBLE_DEVICES=2 python generate_points_framewise.py \
    --preprocessed_data_dir /home/ademi/hermes/data/open_drawer_20241103b_preprocessed_30hz \
    --task_name open_drawer_20241103b_30hz \
    --num_tracked_points 29 \
    --unproject_depth &

CUDA_VISIBLE_DEVICES=3 python generate_points_framewise.py \
    --preprocessed_data_dir /home/ademi/hermes/data/open_drawer_20241103b_preprocessed_30hz \
    --task_name open_drawer_20241103b_30hz \
    --num_tracked_points 29 &

#!/bin/bash
#
# Usage:
# bash generate_train_dataset.sh $task $num_tracked_points

task=$1
num_tracked_points=$2

preprocessdir=/home/ademi/hermes/data/${task}_preprocessed

cd data_generation/
python generate_points_framewise.py \
    --preprocessed_data_dir $preprocessdir \
    --task_name $task \
    --num_tracked_points $num_tracked_points \
    --keypoints_type 3
cd ../

python generate_dataset.py \
    --preprocessed_data_dir $preprocessdir \
    --task_name $task \
    --history_length 10 \
    --subsample 3 \
    --framewise_cotracker \
    --keypoints_type 3

python generate_dataset.py \
    --preprocessed_data_dir $preprocessdir \
    --task_name $task \
    --history_length 10 \
    --subsample 2 \
    --framewise_cotracker \
    --keypoints_type 3

python generate_dataset.py \
    --preprocessed_data_dir $preprocessdir \
    --task_name $task \
    --history_length 10 \
    --subsample 1 \
    --framewise_cotracker \
    --keypoints_type 3

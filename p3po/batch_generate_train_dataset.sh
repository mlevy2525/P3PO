#!/bin/bash
#
# this script runs all jobs in the background! to kill these scripts, you must
# fetch their pids or kill them with something like `pkill python`

tmux new-session -d -s 0 'CUDA_VISIBLE_DEVICES=0 bash generate_train_dataset.sh pick_block_20241107a 9; bash'
tmux new-session -d -s 1 'CUDA_VISIBLE_DEVICES=1 bash generate_train_dataset.sh open_oven_20241107c 6; bash'
tmux new-session -d -s 2 'CUDA_VISIBLE_DEVICES=2 bash generate_train_dataset.sh collect_ball_20241107e 10; bash'

tmux new-session -d -s 3 'CUDA_VISIBLE_DEVICES=3 bash generate_train_dataset.sh pick_block_20241107b 9; bash'
tmux new-session -d -s 4 'CUDA_VISIBLE_DEVICES=0 bash generate_train_dataset.sh open_oven_20241107d 6; bash'
tmux new-session -d -s 5 'CUDA_VISIBLE_DEVICES=1 bash generate_train_dataset.sh collect_ball_20241107f 10; bash'

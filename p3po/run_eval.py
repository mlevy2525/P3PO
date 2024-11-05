import subprocess

devices = range(4)  
device_idx = 0
run_name = "pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz"
tasks = ["pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz_closed_loop_dataset"]
p3po_task_name = "pick_block_20241103a_30hz_nooccpoints"
num_tracked_points = 8
point_dimensions = 3
use_wandb = "false"
history_len = 10
bc_weight = "/home/ademi/P3PO/p3po/exp_local/2024.11.05/145900_pick_block_20241103a_30hz_nooccpoints_3d_abs_actions_minlength33_10hz_std0.1_repeatpad_notempagg_3d/snapshot/80000.pt"

command = (
    f"CUDA_VISIBLE_DEVICES={devices[device_idx % len(devices)]} HYDRA_FULL_ERROR=1 "
    f"python train.py agent=baku suite=dexterous dataloader=p3po_general "
    f"suite.hidden_dim=256 suite.task.tasks={tasks} "
    f"point_dimensions={point_dimensions} use_wandb={use_wandb} suite.history_len={history_len} "
    f"run_name={run_name} p3po_task_name={p3po_task_name} num_tracked_points={num_tracked_points} "
    f"eval_only=true load_bc=true bc_weight={bc_weight} temporal_agg=false"
)
print(f"Running on device {devices[device_idx % len(devices)]}: {command}")
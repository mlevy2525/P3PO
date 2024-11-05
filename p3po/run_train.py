import subprocess

devices = range(4)  
device_idx = 0
run_name = "open_drawer_20241103b_30hz_3d_abs_actions_minlength33_10hz"
tasks = ["open_drawer_20241103b_30hz_3d_abs_actions_minlength33_10hz_closed_loop_dataset"]
p3po_task_name = "open_drawer_20241103b_30hz"
num_tracked_points = 29
point_dimensions = 3
use_wandb = "true"
history_len = 1

command = (
    f"CUDA_VISIBLE_DEVICES={devices[device_idx % len(devices)]} HYDRA_FULL_ERROR=1 "
    f"python train.py agent=baku suite=dexterous dataloader=p3po_general "
    f"suite.hidden_dim=256 suite.task.tasks={tasks} "
    f"point_dimensions={point_dimensions} use_wandb={use_wandb} suite.history_len={history_len} "
    f"run_name={run_name} p3po_task_name={p3po_task_name} num_tracked_points={num_tracked_points}"
)
print(f"Running on device {devices[device_idx % len(devices)]}: {command}")
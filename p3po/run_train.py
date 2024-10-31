import subprocess

devices = range(4)  
device_idx = 0      
run_name = "pick_and_place_bread_1030a_abs_actions_history10"
tasks = ["pick_and_place_bread_1030a_3d_abs_actions_minlength12_closed_loop_dataset"]
p3po_task_name = "pick_and_place_bread_1030a"
num_tracked_points = 31
point_dimensions = 3
use_wandb = "true"
history_len = 10
out_log_dir = "/home/ademi/P3PO/logs"
_out = _err = f"{out_log_dir}/{run_name}_train.log"

with open(_out, "wb") as out, open(_err, "wb") as err:
    command = (
        f"CUDA_VISIBLE_DEVICES={devices[device_idx % len(devices)]} HYDRA_FULL_ERROR=1 "
        f"python train.py agent=baku suite=dexterous dataloader=p3po_general "
        f"suite.hidden_dim=256 suite.task.tasks={tasks} "
        f"point_dimensions={point_dimensions} use_wandb={use_wandb} suite.history_len={history_len} "
        f"run_name={run_name} p3po_task_name={p3po_task_name} num_tracked_points={num_tracked_points}"
    )
    print(f"Running on device {devices[device_idx % len(devices)]}: {command}")
    p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)

p.wait()
print('returncode', p.returncode)

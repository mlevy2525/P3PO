#!/usr/bin/env python3

import os
import subprocess
from P3PO.p3po.config_singleton import ConfigSingleton

def main():
    eval_process = None  # To track the running eval.py process

    print("Task Manager Initialized")
    print("Enter task name to start evaluation or 'reset' to reset the robot or 'stop' to terminate the ongoing evaluation.")

    while True:
        # Get user input
        user_input = input("Enter a task name or 'stop': ").strip()

        # Handle "stop" input to terminate the current eval process
        if user_input.lower() == "stop":
            if eval_process and eval_process.poll() is None:
                print("Stopping the ongoing evaluation process...")
                eval_process.terminate()
                eval_process.wait()
                print("Evaluation process terminated.")
            else:
                print("No evaluation process is currently running.")
            continue

        # Handle "reset" input to reset the robot
        # if it is reset, run python reset.py suite=xarm_env_reset
        if user_input.lower() == "reset":
            try:
                print("Resetting the robot...")
                subprocess.run(["python", "reset.py", "suite=xarm_env_reset"])
                print("Robot reset successfully.")
            except Exception as e:
                print(f"Failed to reset the robot: {e}")
            continue

        # Handle task name input
        if user_input:
            # TODO: use llm to classify task type
            # task_name = llm.classify_task(user_input)

            # For now, hack
            # task_name = '0902_pickmug_anything'
            task_name, des_object = user_input.split(',')[0].strip(), user_input.split(',')[1].strip()
            if task_name ==  '1220_pick_bottle_from_fridge_12p':
                model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_pick_bottle_fridge_7p2/snapshot/90000.pt'
                expert_background = 'cam4_robot_7p'
                expert_object = 'bottle'
            elif task_name == '1223_place_bottle_on_ground':
                # model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_place_bottle/snapshot/100000.pt'
                model_path = '/data/bobby/1223_place_bottle_on_ground/100000.pt'
                expert_background = 'cam4_robot_place_bottle'
                expert_object = 'bottle'
            elif task_name == '1220_pick_bottle_from_side_door':
                model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_pick_bottle_side_door/snapshot/100000.pt'
                expert_background = 'cam4_robot_7p'
                expert_object = 'bottle'
            elif task_name == '0103_place_bottle_on_ground':
                # model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_place_bottle_reset/snapshot/100000.pt'
                model_path = '/data/bobby/0103_place_bottle_on_ground/100000.pt'
                expert_background = 'cam4_robot_place_bottle'
                expert_object = 'bottle'
            elif task_name == '1220_pick_bottle_from_fridge_new':
                # model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_pick_bottle_side_door_new/snapshot/150000.pt'
                model_path = '/data/bobby/1220_pick_bottle_from_fridge_new/150000.pt'
                expert_background = ''
                expert_object = '' 
            elif task_name == '1220_pick_bottle_from_side_door_new':
                # model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_pick_bottle_fridge_new/snapshot/140000.pt'
                model_path = '/data/bobby/1220_pick_bottle_from_side_door_new/140000.pt'
                expert_background = ''
                expert_object = ''
            elif task_name == '0105_place_side_door_bottle_on_ground':
                # model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_0105_place_side_door/snapshot/112000.pt
                model_path = '/data/bobby/0105_place_side_door_bottle_on_ground/112000.pt'
                expert_background = ''
                expert_object = ''

            # Validate model path
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue

            import yaml

            # Define the dictionary
            info_dict = {
                "task_name": task_name,
                "model_path": model_path,
                "desired_object": des_object
            }

            # Write the dictionary to a YAML file
            with open('/home/aadhithya/bobby_wks/P3PO/p3po/current_info.yaml', 'w') as f:
                yaml.dump(info_dict, f)

            # Terminate any existing eval process before starting a new one
            if eval_process and eval_process.poll() is None:
                print("Stopping the existing evaluation process...")
                eval_process.terminate()
                eval_process.wait()
                print("Previous evaluation process terminated.")

            # Update the global configuration using the singleton
            ConfigSingleton({
                "task_name": task_name,
                "model_path": model_path,
                "agent": "baku",
                "suite": "xarm_env",
                "dataloader": "p3po_xarm",
                "use_proprio": False,
                "hidden_dim": 256,
                "expert_background": expert_background,
                "expert_object": expert_object,
            })

            # Start the new eval process
            try:
                print(f"Starting eval.py for task: {task_name}, model: {model_path}")
                eval_process = subprocess.Popen(
                    [
                        "python",
                        "eval.py",
                        f"agent=baku",
                        f"suite=xarm_env",
                        f"dataloader=p3po_xarm",
                        f"suite.task.tasks=[{task_name}]",
                        "use_proprio=false",
                        "suite.hidden_dim=256",
                        f"bc_weight={model_path}",
                    ]
                )
            except Exception as e:
                print(f"Failed to start eval.py: {e}")
        else:
            print("Invalid input. Please enter a task name or 'stop'.")

if __name__ == "__main__":
    main()

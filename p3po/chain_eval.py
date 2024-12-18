#!/usr/bin/env python3

import os
import subprocess

def main():
    eval_process = None  # To track the running eval.py process

    print("Task Manager Initialized")
    print("Enter task name to start evaluation or 'stop' to terminate the ongoing evaluation.")

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

        # Handle task name input
        if user_input:
            # TODO: use llm to classify task type
            # task_name = llm.classify_task(user_input)

            # For now, hack
            task_name = '0902_pickmug_anything'
            # model_path = f"/path/to/model/{task_name}.pt"  # Adjust this logic if model paths vary
            model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_mug_anything/snapshot/100000.pt'

            # Validate model path
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue

            # Terminate any existing eval process before starting a new one
            if eval_process and eval_process.poll() is None:
                print("Stopping the existing evaluation process...")
                eval_process.terminate()
                eval_process.wait()
                print("Previous evaluation process terminated.")

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

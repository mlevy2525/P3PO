import os
import subprocess
import zmq
import cv2
from PIL import Image
import base64
import numpy as np
import io
# import sys
# sys.path.append('/mnt/robotlab/siddhant/P3PO')
# print(sys.path)

from P3PO.p3po.config_singleton import ConfigSingleton
from get_image import CameraCapture

def serialize_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def main():
    eval_process = None  # To track the running eval.py process
    initialize = True

    # Initialize ZMQ context and socket
    context = zmq.Context()
    server_socket = context.socket(zmq.REQ)
    server_socket.connect("tcp://172.24.71.224:6000")

    print("Task Manager Initialized")
    prompt = "Take all bottles out of the fridge"

    if not prompt:
        print("Invalid input. Please enter a valid prompt.")
        return

    camera_capture = CameraCapture()

    # serialize images
    serialize_images = []
    task_complete = True

    # # Save and send images to the server
    # image_paths = []
    # for idx, image in enumerate(images):
    #     image_path = f"camera_{idx}.jpg"
    #     cv2.imwrite(image_path, image)
    #     image_paths.append(image_path)

    while True:
        if initialize:
            initialize = False
            request = {
                "image": serialize_images,
                "image_path": "",
                "query": f"Initialize: {prompt}"
            }
            server_socket.send_json(request)
            response = server_socket.recv_json()
            print("Initialized with response: ", response)
            # import ipdb; ipdb.set_trace()
            continue
        if task_complete:
            # Request the next subtask from the server
            task_complete = False
            print("Capturing images...")
            images = camera_capture.capture_images()
            for idx, image in enumerate(images):
                image = image[:,:,::-1]
                # # convert np array to image
                image = Image.fromarray(image)
                serialized_image = serialize_image(image)
                serialize_images.append(serialized_image)
            print("Requesting next subtask...")
            request = {
                    "image": serialize_images,
                    "image_path": "",
                    "query": f"Next task"
                }
            server_socket.send_json(request)
            task = server_socket.recv_json()
            print("Received task: ", task)
            task = task['result'].lower()

            if not task or task == "stop":
                print("All tasks completed.")
                exit()

            # task_name = task.get("name")
            # model_path = task.get("model_path")
            # get what is inside "[]"
            task, des_object = task[1:-1].split(',')[0].strip(), task[1:-1].split(',')[1].strip()
            if task == 'pick_bottle_from_side_door_of_fridge':
                task_name = '1220_pick_bottle_from_side_door'
                model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_pick_bottle_side_door/snapshot/100000.pt'
            elif task == 'place_bottle_on_ground':
                task_name = '1223_place_bottle_on_ground'
                model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_place_bottle/snapshot/100000.pt'
            elif task == 'pick_bottle_from_fridge':
                task_name =  '1220_pick_bottle_from_fridge_12p'
                model_path = '/mnt/robotlab/siddhant/P3PO/snapshot_pick_bottle_fridge_7p2/snapshot/90000.pt'

            print(f"Task: {task_name}, Model: {model_path}, Object: {des_object}")

            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue

            # Terminate any existing eval process before starting a new one
            if eval_process and eval_process.poll() is None:
                print("Stopping the existing evaluation process...")
                eval_process.terminate()
                eval_process.wait()
                print("Previous evaluation process terminated.")

            import yaml

            # Define the dictionary
            info_dict = {
                "task_name": task_name,
                "model_path": model_path,
                "desired_object": des_object
            }

            # Write the dictionary to a YAML file
            with open('/mnt/robotlab/siddhant/P3PO/p3po/current_info.yaml', 'w') as f:
                yaml.dump(info_dict, f)

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
                continue

        frame_count = 0
        while not task_complete:
            frame_count += 1

            if frame_count % 200 == 0:
                # Capture a frame to send to the server for judging
                judge_serialized_images = []
                images = camera_capture.capture_images()
                for idx, image in enumerate(images):
                    image = image[:,:,::-1]
                    # # convert np array to image
                    image = Image.fromarray(image)
                    serialized_image = serialize_image(image)
                    judge_serialized_images.append(serialized_image)

                print("Sending images to the server for judging...")
                request = {
                    "image": judge_serialized_images,
                    "image_path": "",
                    "query": f"Judge: {task_name}"
                }

                server_socket.send_json(request)
                response = server_socket.recv_json()
                print("Received response from server: ", response)
                is_complete = True if response['result'] == '1' else False

                if is_complete:
                    task_complete = True
                    print(f"Task '{task_name}' completed successfully.")
                    eval_process.terminate()
                    eval_process.wait()
                    break

if __name__ == "__main__":
    main()

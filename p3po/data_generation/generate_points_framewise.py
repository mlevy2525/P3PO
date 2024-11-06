import argparse
import json
import os
import pickle
import shutil
import sys
import yaml

import cv2
import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append('../')
from points_class import PointsClass

CAM_ID = 0
LOG_FOLDER = "outputs/"
COORDS_FOLDER = "../../coordinates"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', type=str, required=True, help='path to demo data folder with preprocessed outputs')
    parser.add_argument('--task_name', type=str, required=True, help='task name')
    parser.add_argument('--num_tracked_points', type=int, required=True, help='number of tracked points in this task')
    parser.add_argument('--keypoints_type', type=float, choices=[2, 2.5, 3], required=True, help='type of points: 2d for xy, 2.5d for xyz, 3d for unprojected(xyz)')
    args = parser.parse_args()

    save_images = True
    gt_depth = True

    preprocessed_data_dir = args.preprocessed_data_dir
    task_name = args.task_name
    num_tracked_points = args.num_tracked_points
    if args.keypoints_type != 2.5:
        args.keypoints_type = int(args.keypoints_type)
    keypoints_type = args.keypoints_type
    dimensions = 2 if keypoints_type == 2 else 3

    with open("../cfgs/suite/p3po.yaml") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    cfg['task_name'] = task_name
    cfg['num_tracked_points'] = num_tracked_points
    cfg['keypoints_type'] = keypoints_type
    cfg['dimensions'] = dimensions

    print(f'using preprocessed_data_dir={preprocessed_data_dir}, cfg={cfg}')

    coords = pickle.load(open(os.path.join(COORDS_FOLDER, "coords", f"{task_name}.pkl"), 'rb'))
    if len(coords) != num_tracked_points:
        raise RuntimeError('Number of points found in PKL does not match the number specified !!')

    points_class = PointsClass(**cfg)

    out_dir = os.path.join(LOG_FOLDER, f'{task_name}_{str(keypoints_type)}d')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(preprocessed_data_dir, "valid_demo_ids.json"), "r") as f:
        valid_demo_ids = json.load(f)
        valid_demo_ids = sorted(map(int, valid_demo_ids))

    graphs_list = []
    trajectories = {}
    with tqdm(total=len(valid_demo_ids)) as pbar:
        for demo_id in valid_demo_ids:
            directory = f"demonstration_{demo_id}"
            pbar.set_postfix({'directory': directory})

            path = os.path.join(preprocessed_data_dir, directory, f"cam_{CAM_ID}_rgb_images")
            depth_path = os.path.join(preprocessed_data_dir, directory, f"cam_{CAM_ID}_depth.h5")

            try:
                with open(os.path.join(preprocessed_data_dir, directory, f"image_indices_cam_{CAM_ID}.pkl"), "rb") as f:
                    image_indices = pickle.load(f)
                image_indices = [entry[1] for entry in image_indices]
                images = [np.array(Image.open(os.path.join(path, f"frame_{file_num:05d}.png"))) for file_num in image_indices]
            except:
                print(f"Failed to load images from {path}")
                continue
            with h5py.File(depth_path, 'r') as h5_file:
                depth_dataset = h5_file['depth_images']
                depth_images = [depth_dataset[idx] for idx in image_indices]

            save_dir = os.path.join(out_dir, directory)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

            graphs = []
            frames = []
            points_class.reset_episode()
            for idx, image in enumerate(images):
                points_class.add_to_image_list(image)

                if idx == 0:
                    points_class.find_semantic_similar_points()
                    points_class.track_points(is_first_step=True)

                points_class.track_points()

                if gt_depth:
                    depth_image = depth_images[0] / 1000
                    points_class.set_depth(depth_image)
                else:
                    points_class.get_depth()

                graph = points_class.get_points()
                graphs.append(graph)

                if save_images:
                    image = points_class.plot_image()[-1]
                    frames.append(image)
                    cv2.imwrite(os.path.join(save_dir, f"{task_name}_{idx}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if len(frames) > 0:
                iio.imwrite(os.path.join(out_dir, f"{directory}_{task_name}.gif"), frames, duration=3, format='gif', loop=0)
            trajectories[directory.split('_')[-1]] = graphs
            pbar.update(1)

    file_path = os.path.join(preprocessed_data_dir, f"{task_name}_{str(keypoints_type)}d-keypoints.pkl")
    with open(str(file_path), 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved points to {file_path}")

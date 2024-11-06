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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', type=str, required=True, help='path to demo data folder with preprocessed outputs')
    parser.add_argument('--task_name', type=str, required=True, help='task name')
    parser.add_argument('--num_tracked_points', type=int, required=True, help='number of tracked points in this task')
    parser.add_argument('--unproject_depth', action='store_true', default=False, help='whether to use unprojected depth points (3d)')
    args = parser.parse_args()

    save_images = True
    gt_depth = True

    preprocessed_data_dir = args.preprocessed_data_dir
    task_name = args.task_name
    num_tracked_points = args.num_tracked_points
    unproject_depth = args.unproject_depth

    with open("../cfgs/suite/p3po.yaml") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    cfg['task_name'] = task_name
    cfg['num_tracked_points'] = num_tracked_points
    cfg['unproject_depth'] = unproject_depth

    print(f'using preprocessed_data_dir={preprocessed_data_dir}, cfg={cfg}')

    coords = pickle.load(open(f'../../coordinates/coords/{task_name}.pkl', 'rb'))
    if len(coords) != num_tracked_points:
        raise RuntimeError('Number of points found in PKL does not match the number specified !!')

    points_class = PointsClass(**cfg)

    point_type = '3d' if unproject_depth else '2.5d'
    frames_dir = f'{task_name}_{point_type}_frames_framewise'
    gifs_dir = f'{task_name}_{point_type}_gifs_framewise'

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    if os.path.exists(gifs_dir):
        shutil.rmtree(gifs_dir)
    os.makedirs(gifs_dir, exist_ok=True)

    with open(os.path.join(preprocessed_data_dir, "valid_demo_ids.json"), "r") as f:
        valid_demo_ids = json.load(f)
        valid_demo_ids = sorted(map(int, valid_demo_ids))

    graphs_list = []
    trajectories = {}
    directories = sorted([d for d in os.listdir(preprocessed_data_dir) if d.startswith('demonstration_')])
    with tqdm(total=len(valid_demo_ids)) as pbar:
        for demo_id in valid_demo_ids:
            directory = f"demonstration_{demo_id}"
            pbar.set_postfix({'directory': directory})

            path = f'{preprocessed_data_dir}/{directory}/cam_3_rgb_images'
            depth_path = f'{preprocessed_data_dir}/{directory}/cam_3_depth.h5'

            try:
                image_indices = pickle.load(open(f'{preprocessed_data_dir}/{directory}/image_indices_cam_3.pkl', 'rb'))
                image_indices = [entry[1] for entry in image_indices]
                images = [np.array(Image.open(f'{path}/frame_{str(file_num).zfill(5)}.png')) for file_num in image_indices]
            except:
                print(f"Failed to load images from {path}")
                continue
            with h5py.File(depth_path, 'r') as h5_file:
                depth_dataset = h5_file['depth_images']
                depth_images = [depth_dataset[idx] for idx in image_indices]

            if os.path.exists(f'{frames_dir}/{directory}'):
                shutil.rmtree(f'{frames_dir}/{directory}')
            os.makedirs(f'{frames_dir}/{directory}')

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
                    cv2.imwrite(f'{frames_dir}/{directory}/{task_name}_{idx}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if len(frames) > 0:
                iio.imwrite(f'{gifs_dir}/{directory}_{task_name}.gif', frames, duration=3, format='gif', loop=0)
            trajectories[directory.split('_')[-1]] = graphs
            pbar.update(1)

    file_path = f'{preprocessed_data_dir}/{task_name}_{point_type}_cov3_framewise.pkl'
    with open(str(file_path), 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved points to {file_path}")

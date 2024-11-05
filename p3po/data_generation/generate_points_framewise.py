import os
os.environ['MUJOCO_GL'] = 'egl'
import pickle
from pathlib import Path
import sys
sys.path.append('../')
from points_class import PointsClass
import yaml
import numpy as np
import cv2
import imageio
import h5py
from PIL import Image
import pickle as pkl
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', type=str, required=True, help='path to demo data folder with preprocessed outputs')
    args = parser.parse_args()
    
    save_images = True
    gt_depth = True
    orig_bgr = True
    
    preprocessed_data_dir = args.preprocessed_data_dir

    with open("../cfgs/suite/p3po.yaml") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    task_name = cfg['task_name']
    coords = pkl.load(open(f'../../coordinates/coords/{task_name}.pkl', 'rb'))
    cfg['num_tracked_points'] = len(coords)
    print(f"Number of tracked points: {cfg['num_tracked_points']}")
    points_class = PointsClass(**cfg)
    dimensions = cfg['dimensions']

    if os.path.exists(f'{task_name}_{dimensions}d_frames_framewise'):
        import shutil
        shutil.rmtree(f'{task_name}_{dimensions}d_frames_framewise')
    os.makedirs(f'{task_name}_{dimensions}d_frames_framewise', exist_ok=True)
    if os.path.exists(f'{task_name}_{dimensions}d_gifs_framewise'):
        import shutil
        shutil.rmtree(f'{task_name}_{dimensions}d_gifs_framewise')
    os.makedirs(f'{task_name}_{dimensions}d_gifs_framewise', exist_ok=True)

    graphs_list = []
    trajectories = {}
    for directory in tqdm(os.listdir(preprocessed_data_dir)):
        if 'demonstration' not in directory:
            continue
        path = f'{preprocessed_data_dir}/{directory}/cam_3_rgb_images'
        depth_path = f'{preprocessed_data_dir}/{directory}/cam_3_depth.h5'
        print(f"Loading frames from {path}")
        try:
            image_indices = pkl.load(open(f'{preprocessed_data_dir}/{directory}/image_indices_cam_3.pkl', 'rb'))
            image_indices = [entry[1] for entry in image_indices]
            images = [np.array(Image.open(f'{path}/frame_{str(file_num).zfill(5)}.png')) for file_num in image_indices]
        except:
            print(f"Failed to load images from {path}")
            continue
        with h5py.File(depth_path, 'r') as h5_file:
            depth_dataset = h5_file['depth_images']
            depth_images = [depth_dataset[idx] for idx in image_indices]
        graphs = []

        points_class.reset_episode()

        if orig_bgr:
            points_class.add_to_image_list(images[0][:,:,::-1])
        else:
            points_class.add_to_image_list(images[0])

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

        if os.path.exists(f'{task_name}_{dimensions}d_frames_framewise/{directory}'):
            import shutil
            shutil.rmtree(f'{task_name}_{dimensions}d_frames_framewise/{directory}')
        os.makedirs(f'{task_name}_{dimensions}d_frames_framewise/{directory}')
        
        if save_images:
            image = points_class.plot_image()[-1]
            cv2.imwrite(f'{task_name}_{dimensions}d_frames_framewise/{directory}/{task_name}_0.png', image)

        for idx, image in enumerate(images[1:]):
            if orig_bgr:
                points_class.add_to_image_list(image[:,:,::-1])
            else:
                points_class.add_to_image_list(image)
            points_class.track_points()
            if gt_depth:
                depth_image = depth_images[idx] / 1000
                points_class.set_depth(depth_image)
            else:
                points_class.get_depth()
            
            graph = points_class.get_points()
                
            graphs.append(graph)
            if save_images:
                image = points_class.plot_image()[-1]
                cv2.imwrite(f'{task_name}_{dimensions}d_frames_framewise/{directory}/{task_name}_{idx+1}.png', image)

        with imageio.get_writer(f'{task_name}_{dimensions}d_gifs_framewise/{directory}_{task_name}.gif', mode='I', duration=4) as writer:  
            for filetask_name in os.listdir(f'{task_name}_{dimensions}d_frames_framewise/{directory}'):
                if filetask_name.endswith(".png"):
                    image = imageio.imread(f'{task_name}_{dimensions}d_frames_framewise/{directory}/{filetask_name}')
                    writer.append_data(image)

        trajectories[directory.split('_')[-1]] = graphs

    file_path = f'{preprocessed_data_dir}/{task_name}_{dimensions}d_concatdepth_framewise.pkl'
    with open(str(file_path), 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved points to {file_path}")
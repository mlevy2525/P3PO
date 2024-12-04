import sys
sys.path.append("../")

import pickle
import cv2
import yaml
import imageio
from matplotlib import pyplot as plt

import torch

from points_class import PointsClass

# TODO: If you want to use gt depth, set to True and set the key for the depth in the pickle
# To use gt depth, the depth must be in the same pickle as the images
# We assume the input depth is in the form width x height
use_gt_depth = False
gt_depth_key = "depth"

# add images to a list
# image_paths = ["/home/aadhithya/bobby_wks/multiagent/black_plate.png", "/home/aadhithya/bobby_wks/multiagent/white_plate.png", "/home/aadhithya/bobby_wks/multiagent/blue_plate.png"]
# image_paths = ["/home/aadhithya/bobby_wks/multiagent/cropped_image1.png", "/home/aadhithya/bobby_wks/multiagent/cropped_image2.png", "/home/aadhithya/bobby_wks/multiagent/cropped_image3.png", "/home/aadhithya/bobby_wks/multiagent/cropped_image4.png", "/home/aadhithya/bobby_wks/multiagent/cropped_image5.png"]
image_paths = ["/home/aadhithya/bobby_wks/pick_white_plate.png"]

write_images = True

with open("../cfgs/suite/p3po.yaml") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Initialize the PointsClass object
original_task_name = cfg['task_name']
# cfg['task_name'] = "plate_14p"
cfg['task_name'] = "rack_and_robot"
cfg['num_points'] = 10
points_class = PointsClass(**cfg)
episode_list = []

mark_every = 8

for i in range(len(image_paths)):
    image = cv2.imread(image_paths[i])
    points_class.add_to_image_list(image)
    points_class.find_semantic_similar_points()
    points_class.track_points(is_first_step=True)
    points_class.track_points(one_frame=(mark_every == 1))

    if use_gt_depth:
        points_class.set_depth(depth[0])
    else:
        points_class.get_depth()

    points_list = []
    points = points_class.get_points()
    print("Gotten points", points)
    points_list.append(points[0])

    if write_images:
        image = points_class.plot_image()
        cv2.imwrite(f"/home/aadhithya/bobby_wks/P3PO/p3po/data_generation/videos/{cfg['task_name']}_rack_and_robot_{i}.png", image[0])
        # imageio.imwrite(f"{image_paths[i]}", image[0])
        print(f"Image saved to {image_paths[i]}")

    episode_list.append(torch.stack(points_list))
    points_class.reset_episode()
import numpy as np
import dm_env
from dm_env import specs
import torch
import cv2
import os
import sys
sys.path.append("../data_generation/")
from points_class import PointsClass

class P3POWrapper(dm_env.Environment):
    def __init__(self, env, pixel_keys, depth_keys, training_keys, points_class):
        self._env = env

        self.pixel_keys = pixel_keys
        self.training_keys = training_keys
        self.depth_keys = depth_keys
        self.points_class = points_class

        self.observation_spec = self._env.observation_spec
        obs = self.reset().observation
        self._obs_spec["graph"] = specs.BoundedArray(
            shape=obs["graph"].shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="graph",
        )
        os.makedirs("/home/ademi/plotted_images", exist_ok=True)

    def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        obs = observation.observation
        self.points_class.reset_episode()

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        self.points_class.find_semantic_similar_points()
        self.points_class.track_points(is_first_step=True)
        self.points_class.track_points()
        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()

        obs['graph'] = self.points_class.get_points()[-1]
        points_dimensions = self.points_class.num_tracked_points * self.points_class.dimensions
        fingertips_dimensions = obs['fingertips'].shape[0]
        if points_dimensions + fingertips_dimensions > obs['graph'].shape[0]:
            obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(obs['fingertips'])])
        else:
            obs['graph'][points_dimensions:points_dimensions+fingertips_dimensions] = torch.tensor(obs['fingertips'])
        obs = self._env._replace(observation, observation=obs)

        return obs

    def step(self, action):
        observation = self._env.step(action)
        obs = observation.observation

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        self.points_class.track_points()
        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()

        obs['graph'] = self.points_class.get_points()[-1]
        points_dimensions = self.points_class.num_tracked_points * self.points_class.dimensions
        obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(obs['fingertips'])])
        index_pose = np.eye(4)
        index_pose[:3, 3] = obs['fingertips'][0:3]
        img = self.points_class.plot_image(index_pose=index_pose)[-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"/home/ademi/plotted_images/image_{self._env._step}.png", img)

        obs = self._env._replace(observation, observation=obs)
        return obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
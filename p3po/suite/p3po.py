import numpy as np
import dm_env
from dm_env import specs

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

    def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        obs = observation.observation
        self.points_class.reset_episode()

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        # TODO: get object_bbox from reasoning server
        object_bbox = None

        image = obs[self.pixel_keys[0]]
        # # plot image
        # print("plotting image")
        # print(image.shape) # 360, 360, 3
        # import matplotlib.pyplot as plt
        # image_folder = "/home/aadhithya/bobby_wks/P3PO/saved_images"
        # import time
        # plt.imsave(f"{image_folder}/training_image_{time.time()}.png", image)
        # plt.close()
        # exit()

        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        self.points_class.find_semantic_similar_points(object_bbox)
        self.points_class.track_points(is_first_step=True)
        self.points_class.track_points()
        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()

        obs['graph'] = self.points_class.get_points()[-1]
        obs = self._env._replace(observation, observation=obs)
        # # plot
        # image = self.points_class.plot_image()
        # exit()

        return obs

    def step(self, action):
        print("step")
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
        obs = self._env._replace(observation, observation=obs)

        # image = self.points_class.plot_image()
        # exit()

        return obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
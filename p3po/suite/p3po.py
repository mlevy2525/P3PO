import numpy as np
import dm_env
from dm_env import specs
import torch
import cv2
import os
import sys
sys.path.append("../data_generation/")
from points_class import PointsClass
import pickle as pkl

class P3POWrapper(dm_env.Environment):
    def __init__(self, env, pixel_keys, depth_keys, training_keys, points_class, closed_loop_dataset_path, steps_per_obs):
        self._env = env

        self.pixel_keys = pixel_keys
        self.step_count = 0
        self.steps_per_obs = steps_per_obs
        self.training_keys = training_keys
        self.depth_keys = depth_keys
        self.points_class = points_class
        self.nearest_neighbor_state = False
        self.save_nearest_neighbor_info = False
        if self.nearest_neighbor_state or self.save_nearest_neighbor_info:
            self.closed_loop_dataset = pkl.load(open(closed_loop_dataset_path, "rb"))
        self.nearest_neighbor_info = []
        self.fingertips = []

        self.observation_spec = self._env.observation_spec
        obs = self.reset().observation
        self._obs_spec["graph"] = specs.BoundedArray(
            shape=obs["graph"].shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="graph",
        )
        # if os.path.exists("/home/ademi/eval_dump/plotted_images"):
        #     import shutil
        #     shutil.rmtree("/home/ademi/eval_dump/plotted_images")
        # os.makedirs("/home/ademi/eval_dump/plotted_images", exist_ok=True)

    def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        obs = observation.observation
        self.step_count = 0
        self.points_class.reset_episode()

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        self.points_class.find_semantic_similar_points()
        self.points_class.track_points(is_first_step=True)
        self.points_class.track_points(one_frame=(self.steps_per_obs < 8), step_size=self.steps_per_obs)

        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()

        obs['graph'] = self.points_class.get_points()[-1]
        points_dimensions = self.points_class.num_tracked_points * self.points_class.dimensions
        if 'fingertips' in obs:
            self.fingertips = [obs['fingertips']]
            obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(obs['fingertips'])])
        else:
            obs['graph'] = obs['graph'][:points_dimensions]
        
        if self.nearest_neighbor_state or self.save_nearest_neighbor_info:
            min_distance = float('inf')
            min_idx = 0
            min_episode_idx = 0
            for episode_idx, episode in enumerate(self.closed_loop_dataset['observations']):
                graph = torch.tensor(episode['graph'])
                curr_graph = obs['graph'].unsqueeze(0)
                min_idx = torch.abs(curr_graph - graph).sum(-1).argmin()
                distance = torch.abs(curr_graph - graph).sum(-1)[min_idx]
                if distance < min_distance:
                    min_distance = distance
                    min_idx = min_idx
                    min_episode_idx = episode_idx
            if self.nearest_neighbor_state:
                obs['graph'] = torch.tensor(self.closed_loop_dataset['observations'][min_episode_idx]['graph'][min_idx])
            if self.save_nearest_neighbor_info:
                demo_num = self.closed_loop_dataset['demo_num'][min_episode_idx]
                subsample = self.closed_loop_dataset['subsample']
                self.nearest_neighbor_info.append((demo_num, min_idx*subsample))

        obs = self._env._replace(observation, observation=obs)

        return obs

    def step(self, action):
        self.step_count += 1
        observation = self._env.step(action)
        obs = observation.observation

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()
        if 'fingertips' in obs:
            self.fingertips.append(obs['fingertips'])

        if (self.step_count) % self.steps_per_obs == 0:
            self.points_class.track_points(one_frame=(self.steps_per_obs < 8), step_size=self.steps_per_obs)

            obs['graph'] = self.points_class.get_points(last_n_frames=self.steps_per_obs)[-self.steps_per_obs:]
            points_dimensions = self.points_class.num_tracked_points * self.points_class.dimensions
            if 'fingertips' in obs:
                #TODO: MAKE SURE THIS WORKS
                obs['graph'] = torch.concatenate([obs['graph'][:, :points_dimensions], torch.tensor(self.fingertips[-self.steps_per_obs:])])
            else:
                obs['graph'] = obs['graph'][:, :points_dimensions]

            if self.nearest_neighbor_state or self.save_nearest_neighbor_info:
                min_distance = float('inf')
                min_idx = 0
                min_episode_idx = 0
                for episode_idx, episode in enumerate(self.closed_loop_dataset['observations']):
                    graph = torch.tensor(episode['graph'])
                    curr_graph = obs['graph'].unsqueeze(0)
                    min_idx = torch.abs(curr_graph - graph).sum(-1).argmin()
                    distance = torch.abs(curr_graph - graph).sum(-1)[min_idx]
                    if distance < min_distance:
                        min_distance = distance
                        min_idx = min_idx
                        min_episode_idx = episode_idx
                if self.nearest_neighbor_state:
                    obs['graph'] = torch.tensor(self.closed_loop_dataset['observations'][min_episode_idx]['graph'][min_idx])
                if self.save_nearest_neighbor_info:
                    demo_num = self.closed_loop_dataset['demo_num'][min_episode_idx]
                    subsample = self.closed_loop_dataset['subsample']
                    self.nearest_neighbor_info.append((demo_num, min_idx*subsample))

        if 'fingertips' in obs:
            index_pose = np.eye(4)
            index_pose[:3, 3] = obs['fingertips'][0:3]
            middle_pose = np.eye(4)
            middle_pose[:3, 3] = obs['fingertips'][3:6]
            ring_pose = np.eye(4)
            ring_pose[:3, 3] = obs['fingertips'][6:9]
            thumb_pose = np.eye(4)
            thumb_pose[:3, 3] = obs['fingertips'][9:12]
            finger_poses = [index_pose, middle_pose, ring_pose, thumb_pose]
            img = self.points_class.plot_image(finger_poses=finger_poses)[-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"/home/ademi/eval_dump/plotted_images/image_{self._env._step}.png", img)
        if self.save_nearest_neighbor_info:
            with open("/home/ademi/eval_dump/nearest_neighbor_info.pkl", "wb") as f:
                pkl.dump(self.nearest_neighbor_info, f)

            obs = self._env._replace(observation, observation=obs)
        else:
            obs = observation
        return obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
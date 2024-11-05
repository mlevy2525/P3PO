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
    def __init__(self, env, pixel_keys, depth_keys, training_keys, points_class, closed_loop_dataset_path):
        self._env = env

        self.pixel_keys = pixel_keys
        self.training_keys = training_keys
        self.depth_keys = depth_keys
        self.points_class = points_class
        self.nearest_neighbor_state = False
        self.save_nearest_neighbor_info = True
        if self.nearest_neighbor_state or self.save_nearest_neighbor_info:
            self.closed_loop_dataset = pkl.load(open(closed_loop_dataset_path, "rb"))
        self.nearest_neighbor_info = []

        self.observation_spec = self._env.observation_spec
        obs = self.reset().observation
        self._obs_spec["graph"] = specs.BoundedArray(
            shape=obs["graph"].shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="graph",
        )
        if os.path.exists("/home/ademi/eval_dump/plotted_images"):
            import shutil
            shutil.rmtree("/home/ademi/eval_dump/plotted_images")
        os.makedirs("/home/ademi/eval_dump/plotted_images", exist_ok=True)

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
        obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(obs['fingertips'])])
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
        # if 'ik_fingertips' in obs:
        #     obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(action)])
        # else:
        #     obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(obs['fingertips'])])
        obs['graph'] = torch.concatenate([obs['graph'][:points_dimensions], torch.tensor(action)])

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

        # index_pose = np.eye(4)
        # index_pose[:3, 3] = obs['fingertips'][0:3]
        # middle_pose = np.eye(4)
        # middle_pose[:3, 3] = obs['fingertips'][3:6]
        # ring_pose = np.eye(4)
        # ring_pose[:3, 3] = obs['fingertips'][6:9]
        # thumb_pose = np.eye(4)
        # thumb_pose[:3, 3] = obs['fingertips'][9:12]
        # finger_poses = [index_pose, middle_pose, ring_pose, thumb_pose]
        # img = self.points_class.plot_image(finger_poses=finger_poses, finger_color=(0, 255, 0))[-1]

        # index_pose = np.eye(4)
        # index_pose[:3, 3] = obs['ik_fingertips'][0:3]
        # middle_pose = np.eye(4)
        # middle_pose[:3, 3] = obs['ik_fingertips'][3:6]
        # ring_pose = np.eye(4)
        # ring_pose[:3, 3] = obs['ik_fingertips'][6:9]
        # thumb_pose = np.eye(4)
        # thumb_pose[:3, 3] = obs['ik_fingertips'][9:12]
        # finger_poses = [index_pose, middle_pose, ring_pose, thumb_pose]
        # img = self.points_class.plot_image(finger_poses=finger_poses, finger_color=(0, 0, 255))[-1]

        index_pose = np.eye(4)
        index_pose[:3, 3] = action[0:3]
        middle_pose = np.eye(4)
        middle_pose[:3, 3] = action[3:6]
        ring_pose = np.eye(4)
        ring_pose[:3, 3] = action[6:9]
        thumb_pose = np.eye(4)
        thumb_pose[:3, 3] = action[9:12]
        finger_poses = [index_pose, middle_pose, ring_pose, thumb_pose]
        img = self.points_class.plot_image(finger_poses=finger_poses, finger_color=(255, 0, 0))[-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(f"/home/ademi/eval_dump/plotted_images/image_{self._env._step}.png", img)
        if self.save_nearest_neighbor_info:
            with open("/home/ademi/eval_dump/nearest_neighbor_info.pkl", "wb") as f:
                pkl.dump(self.nearest_neighbor_info, f)

        obs = self._env._replace(observation, observation=obs)
        return obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
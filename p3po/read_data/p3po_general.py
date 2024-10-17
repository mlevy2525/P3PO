import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        processed_path,
        suite,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        training_keys,
        intermediate_goal_step=30,
        store_actions=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._intermediate_goal_step = intermediate_goal_step
        self._keys = training_keys

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = []
        self._paths.extend([Path(path) / f"general/{task}.pkl" for task in tasks])

        self._graph_paths = []
        self._graph_paths.extend([Path(processed_path) / f"points/{task}.pkl" for task in tasks])

        paths = {}
        graph_paths = {}
        idx = 0
        for path in self._paths:
            paths[idx] = path
            graph_paths[idx] = self._graph_paths[idx]
            idx += 1
        del self._paths
        del self._graph_paths
        self._paths = paths
        self._graph_paths = graph_paths

        # store actions
        if store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for (_graph_idx, _path_idx) in zip(self._graph_paths, self._paths):
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            graph_data = pkl.load(open(str(self._graph_paths[_graph_idx]), "rb"))
            if "episode_list" in graph_data:
                graph_data = graph_data['episode_list']
                for i, episode in enumerate(graph_data):
                    data['observations'][i]['graph'] = episode

            for i in range(len(data["observations"])):
                if isinstance(data['observations'][i]['graph'], list):
                    data['observations'][i]['graph'] = np.array(data['observations'][i]['graph'])

            observations = (
                data["observations"]
            )
            actions = data["actions"]

            if "task_emb" in data:
                task_emb = data["task_emb"]
            else:
                task_emb = 0
                
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                if isinstance(actions[i], list):
                    actions[i] = np.array(actions[i])
                    actions[i] = actions[i].reshape(actions[i].shape[0], -1)

                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i][self._keys[0]])
                    ),
                )
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1] if "states" in data else 100
                )
                self._num_samples += (
                    len(observations[i][self._keys[0]])
                )

                # store actions
                if store_actions:
                    self.actions.append(actions[i])

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
            "proprioceptive": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]
        if "task_emb" in episodes:
            task_emb = episodes["task_emb"]
        else:
            task_emb = 0

        # Sample obs, action
        sample_idx = np.random.randint(
            0, len(observations[self._keys[0]]) - self._history_len
        )
        sampled_input = {}
        for key in self._keys:
            sampled_input[key] = observations[key][
                sample_idx : sample_idx + self._history_len
            ]
            if self._obs_type == "pixels":
                sampled_input[key] = torch.stack(
                    [
                        self.aug(sampled_input[key][i])
                        for i in range(len(sampled_input[key]))
                    ]
                )

        if self._temporal_agg:
            # arrange sampled action to be of shape (history_len, num_queries, action_dim)
            sampled_action = np.zeros(
                (self._history_len, self._num_queries, actions.shape[-1])
            )
            num_actions = (
                self._history_len + self._num_queries - 1
            )  # -1 since its num_queries including the last action of the history
            act = np.zeros((num_actions, actions.shape[-1]))
            act[
                : min(len(actions), sample_idx + num_actions) - sample_idx
            ] = actions[sample_idx : sample_idx + num_actions]
            sampled_action = np.lib.stride_tricks.sliding_window_view(
                act, (self._num_queries, actions.shape[-1])
            )
            sampled_action = sampled_action[:, 0]
        else:
            sampled_action = actions[sample_idx : sample_idx + self._history_len]

        # prompt
        if self._prompt == None or self._prompt == "text":
            sampled_input['actions'] = self.preprocess["actions"](sampled_action)
            sampled_input['task_emb'] = task_emb
            return sampled_input
        elif self._prompt == "goal":
            prompt_episode = self._sample_episode(env_idx)
            prompt_observations = prompt_episode["observation"]
            prompt_pixel = self.aug(prompt_observations["pixels"][-1])[None]
            prompt_action = prompt_episode["action"][-1:]

            sampled_input['actions'] = self.preprocess["actions"](sampled_action)
            sampled_input['task_emb'] = task_emb
            sampled_input['prompt_pixels'] = prompt_pixel
            sampled_input['prompt_actions'] = self.preprocess["actions"](prompt_action)
            return sampled_input
        elif self._prompt == "intermediate_goal":
            prompt_episode = episodes
            prompt_observations = prompt_episode["observation"]
            intermediate_goal_step = (
                self._intermediate_goal_step + np.random.randint(-30, 30)
            )
            goal_idx = min(
                sample_idx + intermediate_goal_step,
                len(prompt_observations["pixels"]) - 1,
            )
            prompt_pixel = self.aug(prompt_observations["pixels"][goal_idx])[None]
            prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]

            sampled_input['actions'] = self.preprocess["actions"](sampled_action)
            sampled_input['task_emb'] = task_emb
            sampled_input['prompt_pixels'] = prompt_pixel
            sampled_input['prompt_actions'] = self.preprocess["actions"](prompt_action)
            return sampled_input

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]
        actions = episode["action"]
        if "task_emb" in episode:
            task_emb = episode["task_emb"]
        else:
            task_emb = 0

        input_shape = observations[self._keys[0]].shape

        # observation
        if self._prompt == None or self._prompt == "text":
            prompt_pixel = None
            prompt_action = None
        elif self._prompt == "goal":
            if self._obs_type == "pixels":
                prompt_pixel = np.transpose(observations["pixels"][-1:], (0, 3, 1, 2))
            else:
                prompt_action = observations(self._keys[0])[-1:]
            prompt_action = None
        elif self._prompt == "intermediate_goal":
            goal_idx = min(
                step + self._intermediate_goal_step, len(observations[self._keys[0]]) - 1
            )
            if self._obs_type == "pixels":
                prompt_pixel = np.transpose(observations["pixels"][goal_idx : goal_idx + 1], (0, 3, 1, 2))
            else:
                prompt_action = observations(self._keys[0])[goal_idx : goal_idx + 1]
            prompt_action = None

        return {
            "prompt_pixels": prompt_pixel,
            "prompt_actions": (
                self.preprocess["actions"](prompt_action)
                if prompt_action is not None
                else None
            ),
            "task_emb": task_emb,
        }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples

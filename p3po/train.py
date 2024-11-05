#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import cv2
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
import wandb
import pickle as pkl

def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    for key in cfg.training_keys:
        obs_shape[key] = obs_spec[key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
         
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        if self.cfg.eval_only:
            self.cfg.eval = True

        self.cfg.run_name = self.cfg.run_name + '_' + str(self.cfg.point_dimensions) + 'd'
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.project_name,  # Add the project name to your config
                config=dict(self.cfg),
                dir=str(self.work_dir),
                name=self.cfg.run_name,  # Add a unique run name
            )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len * self.cfg.suite.action_repeat
        )
        if self.cfg.eval:
            # self.cfg.suite.task_make_fn.max_episode_len = (10000)
            self.cfg.suite.task_make_fn.max_episode_len = (150)
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)
        if self.cfg.use_p3po:
            from suite.p3po import P3POWrapper
            from points_class import PointsClass
            import yaml

            with open(f"{self.cfg.root_dir}/p3po/cfgs/suite/p3po.yaml") as stream:
                try:
                    points_cfg = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            points_cfg["root_dir"] = self.cfg.root_dir
            points_cfg["dimensions"] = self.cfg.point_dimensions
            points_cfg["task_name"] = self.cfg.p3po_task_name
            points_cfg["num_tracked_points"] = self.cfg.num_tracked_points

            points_class = PointsClass(**points_cfg)
            for i in range(len(self.env)):
                self.env[i] = P3POWrapper(self.env[i], self.cfg.suite.pixel_keys, self.cfg.depth_keys, self.cfg.training_keys, points_class, closed_loop_dataset_path=dataset_iterable._paths[0], steps_per_obs=self.cfg.steps_per_obs)


        # create agent
        if self.cfg.dataloader.bc_dataset._target_ == "read_data.p3po_general.BCDataset":
            from dm_env import specs
            action_spec = specs.BoundedArray(
                            (self.expert_replay_loader.dataset._max_action_dim,),
                            np.float32,
                            self.stats["actions"]["min"],
                            self.stats["actions"]["max"],
                            "action",
                        )
        else:
            action_spec = self.env[0].action_spec()

        self.agent = make_agent(
            self.env[0].observation_spec(), action_spec, cfg
        )

        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx

        # Discretizer for BeT
        if repr(self.agent) != "mtact":
            if repr(self.agent) == "rt1" or self.cfg.agent.policy_head in [
                "bet",
                "vqbet",
            ]:
                self.agent.discretize(
                    self.expert_replay_loader.dataset.actions,
                    self.expert_replay_loader.dataset.preprocess,
                )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        use_residual_setpoints = False
        alpha = 1.0
        print(f"{use_residual_setpoints=} {alpha=}")

        self.agent.train(False)
        episode_rewards = []
        successes = []

        num_envs = self.envs_till_idx
        self.vinn = False
        self.open_loop = False
        if self.vinn or self.open_loop:
            self.closed_loop_dataset = pkl.load(open("/home/ademi/P3PO/expert_demos/general/open_oven_1025_densehandle_3d_abs_actions_closed_loop_dataset.pkl", "rb"))

        for env_idx in range(num_envs):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            residual = np.zeros(12, dtype=np.float32)

            while eval_until_episode(episode):
                torch.cuda.empty_cache()
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)
                
                if self.open_loop:
                    ol_step = 0

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    if self.vinn:
                        curr_graph = time_step.observation["graph"].unsqueeze(0)
                        min_distance = float('inf')
                        min_idx = 0
                        min_episode_idx = 0
                        for episode_idx, episode in enumerate(self.closed_loop_dataset['observations']):
                            if episode_idx != 38:
                                continue
                            graph = torch.tensor(episode['graph'])
                            min_idx = torch.abs(curr_graph - graph).sum(-1).argmin()
                            distance = torch.abs(curr_graph - graph).sum(-1)[min_idx]
                            if distance < min_distance:
                                min_distance = distance
                                min_idx = min_idx
                                min_episode_idx = episode_idx
                        print('closest episode:', min_episode_idx)
                        action = self.closed_loop_dataset['actions'][min_episode_idx][min_idx].flatten()
                        print('action:', action)
                    elif self.open_loop:
                        action = self.closed_loop_dataset['actions'][0][ol_step].flatten()
                        ol_step += 1
                    else:
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action = self.agent.act(
                                time_step.observation,
                                prompt,
                                self.stats,
                                step,
                                self.global_step,
                                eval_mode=True,
                            )
                            if use_residual_setpoints:
                                action = action + alpha * residual
                    if self.open_loop: 
                        ol_step += 1
                    # plots = self.env[0].points_class.plot_image(last_n_frames=self.cfg.steps_per_obs)
                    for i in range(self.cfg.steps_per_obs):
                        # cv2.imwrite("plot_" + str(step) + ".png", plots[i][:,:,::-1])
                        if not time_step.last():
                            if use_residual_setpoints:
                                time_step = self.env[env_idx].step(action[i] + residual * alpha)
                            else:
                                time_step = self.env[env_idx].step(action[i])
                            self.video_recorder.record(self.env[env_idx])
                            total_reward += time_step.reward
                            step += 1

                        if 'fingertips' in time_step.observation:
                            residual = (time_step.observation["fingertips"] - action[i])
                    # print(f"execution residual per fingertip: [{np.linalg.norm(residual[0:3]).item():.4f}, {np.linalg.norm(residual[3:6]).item():.4f}, {np.linalg.norm(residual[6:9]).item():.4f}, {np.linalg.norm(residual[9:12]).item():.4f}]")

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_step}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - num_envs):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[:num_envs]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
        
        if self.cfg.use_wandb:
            wandb.log({
                    "eval/episode_reward": np.mean(episode_rewards[:num_envs]),
                    "eval/success": np.mean(successes),
                    "eval/episode_length": step * self.cfg.suite.action_repeat / episode,
                    "eval/episode": self.global_episode,
                    "eval/step": self.global_step,
                })


        self.agent.train(True)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)

        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            if (
                self.cfg.eval
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # update
            if not self.cfg.eval_only:
                metrics = self.agent.update(self.expert_replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

                # log
                if log_every_step(self.global_step):
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                        log("total_time", total_time)
                        log("actor_loss", metrics["actor_loss"])
                        log("step", self.global_step)
                    
                    # Log metrics to WandB
                    if self.cfg.use_wandb:
                        wandb.log({
                            "train/actor_loss": metrics["actor_loss"],
                            "train/total_time": self.timer.total_time(),
                            "train/step": self.global_step,
                        })

                # save snapshot
                if save_every_step(self.global_step):
                    self.save_snapshot()

            self._global_step += 1
        if self.cfg.use_wandb:
            wandb.finish()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode", "stats"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=False)


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    # Load weights
    if cfg.load_bc:
        snapshots = {}
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)

    workspace.train()


if __name__ == "__main__":
    main()

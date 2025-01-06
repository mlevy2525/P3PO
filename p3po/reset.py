#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

class EnvironmentReset:
    def __init__(self, cfg):
        self.cfg = cfg
        # Initialize environment
        self.env, _ = hydra.utils.call(self.cfg.suite.task_make_fn)

    def reset_env(self):
        return self.env[0].reset(flag=1)

if __name__ == "__main__":
    from P3PO.p3po.config_singleton import ConfigSingleton
    import hydra

    @hydra.main(config_path="cfgs", config_name="config_eval")
    def main(cfg):
        # ConfigSingleton(cfg)
        env_reset = EnvironmentReset(cfg)

        print("Resetting environment...")
        env_reset.reset_env()
        print("Environment reset successfully.")

    main()

"""Implements a wrapper around real robot replay"""

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from gymnasium import spaces

from dexterous.envs.base import BaseDexterousArmEnv
from dexterous.envs.constants import CAM_TO_INTRINSICS, ALLEGRO_HOME, FRANKA_HOME_CARTESIAN, KINOVA_HOME_CARTESIAN

class DexterousRealArmEnv(BaseDexterousArmEnv):

    CAM_HOST = "172.24.71.240"
    CAM_PORT = 10005
    CAM_ID = 3
    CAM_TYPE = "realsense"
    CAM_INTRINSICS = CAM_TO_INTRINSICS[CAM_TYPE]

    def __init__(
        self, use_object: bool = False, actor_name: str = "hand_and_arm", render_obs: bool = True,
        arm_type: str = "franka", use_robot=True
    ):
        assert arm_type in ["franka", "kinova"]
        self.arm_type = arm_type
        self.render_obs = render_obs
        self.use_object = use_object
        self.actor_name = actor_name
        self.use_robot = use_robot

        if use_object:
            print(
                "use_object=True is not supported yet! This will have different behavior in real than in "
                "sim, since real does not have knowledge of the object's intrinsics and must be extracted"
            )

        # arm
        if "arm" in actor_name:
            if arm_type == "kinova":
                if use_robot:
                    from openteach.robot.kinova import KinovaArm
                    self.arm = KinovaArm()
                self.num_arm_joints = 6
                self.arm_home_position = KINOVA_HOME_CARTESIAN
            elif arm_type == "franka":
                if use_robot:
                    from openteach.robot.franka import FrankaArm
                    self.arm = FrankaArm()
                self.num_arm_joints = 7
                self.arm_home_position = FRANKA_HOME_CARTESIAN
        else:
            self.arm = None
            self.num_arm_joints = 0

        # hand
        if "hand" in actor_name:
            if use_robot:
                from openteach.robot.allegro.allegro import AllegroHand
                self.hand = AllegroHand()
            self.num_hand_joints = 16
        else:
            self.hand = None
            self.num_hand_joints = 0

        self.num_dofs = self.num_arm_joints + self.num_hand_joints

        self.action_space = spaces.Box(
            low=np.array([-1] * self.num_dofs, dtype=np.float32),  # Actions are 12 + 7
            high=np.array([1] * self.num_dofs, dtype=np.float32),
            dtype=np.float32,
        )

        print(f"= arm: {arm_type} | hand: allegro")
        print(f"* number of arm joints: {self.num_arm_joints}")
        print(f"* number of hand joints: {self.num_hand_joints}")
        print(f"* number of dofs total: {self.num_dofs}")

        # camera
        if use_robot:
            from openteach.utils.network import ZMQCameraSubscriber
            self.image_subscriber = ZMQCameraSubscriber(
                host=DexterousRealArmEnv.CAM_HOST,
                port=DexterousRealArmEnv.CAM_PORT + DexterousRealArmEnv.CAM_ID,
                topic_type="RGB",
            )
        self.image_transform = T.Compose([
            T.Resize((DexterousRealArmEnv.CAM_INTRINSICS.H, DexterousRealArmEnv.CAM_INTRINSICS.W)),
        ])

        print("resetting to home position")
        self.reset()

    def reset(self):
        if self.use_robot:
            self.arm.move_coords(self.arm_home_position, duration=10)
            self.hand.move(ALLEGRO_HOME)
        return self.get_state()

    def get_state(self):
        obs = {}
        if self.render_obs:
            obs["pixels"] = self.compute_observation(obs_type="image")
        obs["features"] = self.compute_observation(obs_type="position")
        return obs

    def compute_observation(self, obs_type):
        if obs_type == "image":
            if self.use_robot:
                image, _ = self.image_subscriber.recv_rgb_image()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image, "RGB")
                return np.asarray(self.image_transform(image))
            else:
                return 255 * np.ones((512, 512, 3), np.uint8)

        if obs_type == "position":
            if self.use_robot:
                arm_joint_positions = self.arm.get_joint_position()
                hand_joint_positions = self.hand.get_joint_position()
                joint_positions = np.concatenate([arm_joint_positions[:self.num_arm_joints], hand_joint_positions], axis=0)
                return joint_positions
            else:
                return np.zeros((self.num_dofs))

        # TODO: if we want to extract state from object, should implement `obs_type="position_object"`
        if obs_type == "velocity" or obs_type == "position_object":
            print(f"self.compute_observation: {obs_type=} not implemented in real yet")

    def step(self, action):
        """Steps an action in the real environment"""
        # action: should be a concatenation of [arm, hand] commands
        # - arm: cartesian set points of 3+4 translation+quaternion
        # - hand: joint positions of 4*4 allegro finger angles
        assert (
            action.shape[0] == 7 + self.num_hand_joints
        ), "Action given should be 22 dimensional action space with kinova actions at first"

        if self.arm is not None:
            self.arm.move_coords(action[:7], duration=1)

        if self.hand is not None:
            self.hand.move(action[7:])

        return self.get_state()

    def step_control(self, action): # Don't move the arm with move_coords but just move with arm_control
        assert (
            action.shape[0] == 7 + self.num_hand_joints
        ), "Action given should be 22 dimensional action space with kinova actions at first"

        if self.arm is not None:
            self.arm.arm_control(action[:7])

        if self.hand is not None:
            self.hand.move(action[7:])

        return self.get_state()
    
    def step_joints(self, action_joints): # action in joint values
        assert (
            action_joints.shape[0] == self.num_arm_joints + self.num_hand_joints
        ), "Action given should be 22 dimensional action space with kinova actions at first"

        if self.arm is not None:
            self.arm.move(action_joints[:7])

        if self.hand is not None:
            self.hand.move(action_joints[7:])

        return self.get_state()
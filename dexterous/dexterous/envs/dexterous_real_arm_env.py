"""Implements a wrapper around real robot replay"""

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.transform import Rotation
from gymnasium import spaces

import sys
sys.path.append("/home/ademi/hermes")
from hermes.environments.base import BaseDexterousArmEnv
from hermes.pose_estimation_ar.constants import CAM_TO_INTRINSICS, MARKER_LENGTH
from hermes.utils.real_constants import ALLEGRO_HOME, FRANKA_HOME_CARTESIAN, KINOVA_HOME_CARTESIAN
from hermes.inverse_kinematics.allegro_with_arm_sgd_ik_solver import FingertipArmIKSolverSGD
from hermes.calibration.calibrate_base import CalibrateBase
from hermes.replayers.hermes_real import convert_Tpose_to_7dof, map_ik_allegro_mount_to_franka_cartesian_pose


class DexterousRealArmEnv(BaseDexterousArmEnv):

    CAM_HOST = "172.24.71.240"
    CAM_PORT = 10005
    CAM_ID = 3
    CAM_TYPE = "realsense"
    CAM_INTRINSICS = CAM_TO_INTRINSICS[CAM_TYPE]
    MARKER_SIZE = MARKER_LENGTH

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

        self.ik_solver = FingertipArmIKSolverSGD(arm_type='franka')

        base_calibr = CalibrateBase(
            host=DexterousRealArmEnv.CAM_HOST,
            calibration_pics_dir='/data/hermes/base_calibration',
            cam_idx=DexterousRealArmEnv.CAM_ID,
            marker_size=DexterousRealArmEnv.MARKER_SIZE,
            marker_id=0
        )
        self.H_B_C = base_calibr.calibrate(save_transform=True, save_img=True)
        base_calibr.get_calibration_error_in_2d(base_to_camera=self.H_B_C)

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
        obs["fingertips"] = self.compute_observation(obs_type="fingertips")
        return obs

    def render(self, mode):
        image = self.compute_observation(obs_type="image")
        return image
    
    def get_joint_positions(self):
        arm_joint_positions = self.arm.get_joint_position()
        hand_joint_positions = self.hand.get_joint_position()
        joint_positions = np.concatenate([arm_joint_positions[:self.num_arm_joints], hand_joint_positions], axis=0)
        return joint_positions
    
    def get_fingertips_in_camera(self):
        joint_positions = self.get_joint_positions()
        robot_fingertips_in_base = self.ik_solver.forward_kinematics(
            current_joint_positions=joint_positions)
        robot_fingertips_in_camera = []
        for hfb in robot_fingertips_in_base:
            finger = self.H_B_C @ hfb
            finger = convert_Tpose_to_7dof(finger)
            robot_fingertips_in_camera.extend(finger[:3])
        robot_fingertips_in_camera = np.array(robot_fingertips_in_camera)
        return robot_fingertips_in_camera
    
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
        
        if obs_type == "fingertips":
            if self.use_robot:
                return self.get_fingertips_in_camera()
            else:
                return np.zeros((12))


        # TODO: if we want to extract state from object, should implement `obs_type="position_object"`
        if obs_type == "velocity" or obs_type == "position_object":
            print(f"self.compute_observation: {obs_type=} not implemented in real yet")

    def step(self, action):
        # action is 12 dimensonal, 3 for each fingertip
        assert action.shape[0] == 12, "Action given should be 12 dimensional action space"

        cur_fingertips_in_camera = self.get_fingertips_in_camera()
        target_fingertips_in_camera = cur_fingertips_in_camera + action # actions are delta actions

        # put the target fingertips in base frame
        H_F_Bs = []
        for i in range(4):
            target_fingertip_in_camera = target_fingertips_in_camera[i*3:(i+1)*3]
            H_F_C = np.eye(4)
            H_F_C[:3, 3] = target_fingertip_in_camera
            H_F_B = np.linalg.pinv(self.H_B_C) @ H_F_C
            H_F_Bs.append(H_F_B)
        H_F_Bs = np.stack(H_F_Bs, axis=0)   

        # use ik to get target joint positions corresponding to target fingertips
        target_joint_positions, _ = self.ik_solver.inverse_kinematics(
            H_F_Bs, self.get_joint_positions(), threshold=1e-3, learning_rate=1e-2,
            finger_arm_weight=100, max_iterations=100, max_grad_norm=1,
        )

        # move the arm to target eef pose corresponding to target joint positions
        H_E_I = self.ik_solver.get_endeff_pose(target_joint_positions, 'allegro_mount') # endeffector-wrt-IK
        arm_pose_from_ik = np.concatenate(
            [H_E_I[:3, 3], Rotation.from_matrix(H_E_I[:3,:3]).as_quat()], axis=0
        )
        H_E_F = map_ik_allegro_mount_to_franka_cartesian_pose(H_E_I = H_E_I) # endeffector-wrt-cartesian-franka
        # Convert homogenous matrix to cart pose
        arm_pose = np.concatenate(
            [H_E_F[:3, 3], Rotation.from_matrix(H_E_F[:3,:3]).as_quat()], axis=0
        )
        print(f'Arm Pose from H_E_I: {arm_pose_from_ik}')
        print(f'Arm Pose from H_E_F: {arm_pose}')
        action = np.concatenate([arm_pose, target_joint_positions[self.num_arm_joints:]], axis=0)
        # input("Press Enter to take an action...")
        obs = self.step_control(action)
        return obs, 0, 0, {}

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
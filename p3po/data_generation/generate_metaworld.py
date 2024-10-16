import metaworld
import random
import  metaworld.policies as policies
import cv2
import numpy as np

import pickle
from pathlib import Path
from collections import deque

import os
os.environ["MUJOCO_GL"] = "egl"

import matplotlib
import sys
import mujoco

num_demos = 35
frame_stack = 1
img_size = 512
action_repeat = 2

SAVE_DATA_PATH = Path('../../expert_demos/metaworld')

from mw_info import env_names, POLICY, CAMERA, NUM_STEPS

for env_name in env_names:
	print(f"Generating demo for: {env_name}")
	# Initialize policy
	policy = POLICY[env_name]()

	# Initialize env
	ml1 = metaworld.MT1(env_name, seed=10) # Construct the benchmark, sampling tasks
	env = ml1.train_classes[env_name](render_mode="rgb_array")  # Create an environment with task `pick_place`

	cam_id = mujoco.mj_name2id(env.mujoco_renderer.model,
                               mujoco.mjtObj.mjOBJ_CAMERA,
                               "corner2")
	env.mujoco_renderer.camera_id = cam_id

	images_list = list()
	states_list = list()
	actions_list = list()
	rewards_list = list()
	depths_list = list()

	episode = 0
	failed = 0
	while episode < num_demos:
		print(f"Episode {episode}")
		images = list()
		depths = list()
		states = list()
		actions = list()
		rewards = list()
		image_stack = deque([], maxlen=frame_stack)
		goal_achieved = 0

		# Set random goal
		task = ml1.train_tasks[episode + failed]
		env.set_task(task)  # Set task

		# Reset env
		state,info = env.reset()  # Reset environment
		env.step(np.zeros(env.action_space.shape))  # Reset environment
		num_steps = NUM_STEPS[env_name]
		for step in range(num_steps//action_repeat):
			# Get state
			states.append(state)
			# Get frames
			frame = env.mujoco_renderer.render("rgb_array")[::-1, :]
			depth = env.mujoco_renderer.render("depth_array")[::-1, :]

			# # Convert from [0 1] to depth in meters, see links below:
			# # http://stackoverflow.com/a/6657284/1461210
			# # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
			extent = env.model.stat.extent
			near = env.model.vis.map.znear * extent
			far = env.model.vis.map.zfar * extent
			depth = near / (1 - depth * (1 - near / far))
			depth = depth * (depth < 10) ## set values > 100 as zero
	
			frame = cv2.resize(frame, (img_size, img_size))
			depth = cv2.resize(depth, (img_size, img_size))

			#Save frame
			images.append(frame)
			depths.append(depth)

			# Get action
			action = policy.get_action(state)
			action = np.clip(action, -1.0, 1.0)
			actions.append(action)
			# Act in the environment
			for _ in range(action_repeat):
				state, reward, terminated, truncated, info = env.step(action)
			rewards.append(reward)
			goal_achieved += info['success'] 

		# Store trajectory
		if goal_achieved > 0:
			episode = episode + 1
			observation = {}
			observation["pixels"] = np.array(images, dtype=np.uint8)
			observation["depth"] = np.array(depths)
			observation["state"] = np.array(states)
			images_list.append(observation)
			states_list.append(np.array(states))
			actions_list.append(np.array(actions))
			rewards_list.append(np.array(rewards))
		else:
			failed += 1
			print("Failed episode, skipping")

	path_name = env_name.split("-")[:-1]
	if len(path_name) > 1:
		path_name = str.join("_", path_name)
	else:
		path_name = path_name[0]
	file_path = SAVE_DATA_PATH / f'{path_name}.pkl'
	payload = {
		'observations': images_list,
		'states': states_list,
		'actions': actions_list,
		'rewards': rewards_list
	}

	with open(str(file_path), 'wb') as f:
		pickle.dump(payload, f)
	print(f"Saved data to {file_path}")
		
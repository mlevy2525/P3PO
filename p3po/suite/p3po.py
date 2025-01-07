import numpy as np
import dm_env
from dm_env import specs

import sys
sys.path.append("../data_generation/")
from points_class import PointsClass

import zmq
import numpy as np
import cv2
from PIL import Image
import io
import base64
import torch

from P3PO.p3po.config_singleton import ConfigSingleton

# print(ConfigSingleton.get_config())
# print each key and its value
# for key in ConfigSingleton.get_config():
#     print(key, ":", ConfigSingleton.get_config()[key])
current_task = ConfigSingleton.get_config()["suite"]['task']['tasks'][0]
# current_weight = ConfigSingleton.get_config()["bc_weight"]

def serialize_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

class P3POWrapper(dm_env.Environment):
    def __init__(self, env, pixel_keys, depth_keys, training_keys, points_class):
        self._env = env

        self.pixel_keys = pixel_keys
        self.training_keys = training_keys
        self.depth_keys = depth_keys
        self.points_class = points_class
        self.isFirstStep = True
        self.isFirstReset = True

        self.observation_spec = self._env.observation_spec
        obs = self.reset(flag=0).observation
        self._obs_spec["graph"] = specs.BoundedArray(
            shape=obs["graph"].shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="graph",
        )

    def reset(self, **kwargs):
        # import ipdb; ipdb.set_trace()
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::p3po reset")
        observation = self._env.reset(**kwargs)
        obs = observation.observation
        self.points_class.reset_episode()

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        if self.isFirstReset:
            self.isFirstReset = False
            import yaml
            with open("/home/aadhithya/bobby_wks/P3PO/p3po/cfgs/suite/p3po.yaml") as stream:
                try:
                    cfg = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # Now read the current_info file
            with open('/home/aadhithya/bobby_wks/P3PO/p3po/current_info.yaml', 'r') as stream:
                try:
                    info = yaml.safe_load(stream)
                    print(info)
                    desired_object = info['desired_object']
                except yaml.YAMLError as exc:
                    print(f"Error reading YAML file: {exc}")

            # connect to reasoning server at port 8888 at 172.24.71.224 ############################
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://172.24.71.224:6000")

            # Initialize the PointsClass object
            if current_task in ['1223_place_bottle_on_ground', '0103_place_bottle_on_ground', '1220_pick_bottle_from_fridge_new', '0106_pick_bottle_from_fridge', '0106_pick_bottle_from_fridge_good']: #'cylindial_bottle' #'plate_12p'  #"plate_14p"
                cfg['task_name'] = 'bottle' 
                cfg['num_points'] = 5
            elif current_task in ['1220_pick_bottle_from_side_door', '0105_place_side_door_bottle_on_ground', '1220_pick_bottle_from_side_door_new']:
                cfg['task_name'] = 'cylindial_bottle'
                cfg['num_points'] = 6
            points_class = PointsClass(**cfg)

            # # # # send the image to the reasoning server and get the bounding box ########################
            image = obs[self.pixel_keys[0]]
            frame = image
            image = image[:,:,::-1]

            # # convert np array to image
            image = Image.fromarray(image)
            serialized_image = serialize_image(image)
            # get bottle type from user
            # bottle_type = input("Enter the bottle type: ")
            bottle_type = 'bottle'
            request = {
                "image": serialized_image,
                "image_path": "",
                "query": f"Get the bounding box of the {desired_object}"
                # "query": "Get the bounding box of the coke bottle"
            }
            socket.send_json(request)
            response = socket.recv_json()
            bbox_plate = response["result"]
            bbox_plate = bbox_plate[:-1] if len(bbox_plate) == 5 else bbox_plate
            print("bbox_plate", bbox_plate)
            print(type(bbox_plate))
            # # exit()

            points_class.add_to_image_list(frame)
            points_class.find_semantic_similar_points(object_bbox = bbox_plate)
            print("found semantic similar points", points_class.semantic_similar_points)
            print(type(points_class.semantic_similar_points))
            first_points = points_class.semantic_similar_points
            first_points_num = cfg['num_points']

            # initialize a new point tracking
            # cfg["task_name"] = 'cam4_robot_place_bottle' if current_task == '1223_place_bottle_on_ground' else 'cam4_robot_7p' #'cam4_robot_7p' #'cam1_robot_7p' #'robot_and_rack_8p'  #"rack_and_robot"
            # cfg["num_points"] = 6 if current_task == '1223_place_bottle_on_ground' else 7 #7 #8 #10
            # cfg["task_name"] = 'cam4_robot_place_bottle_8p' if current_task in ['1223_place_bottle_on_ground', '0103_place_bottle_on_ground'] else 'cam4_robot_8p'
            # cfg["num_points"] = 8
            if current_task in ['1220_pick_bottle_from_fridge_new', '0106_pick_bottle_from_fridge', '0106_pick_bottle_from_fridge_good']:
                cfg['task_name'] = 'cam4_robot_7p'
                cfg['num_points'] = 7
            elif current_task in ['0105_place_side_door_bottle_on_ground']:
                cfg['task_name'] = 'cam4_robot_place_side_door_bottle'
                cfg['num_points'] = 7
            elif current_task in ['1223_place_bottle_on_ground', '0103_place_bottle_on_ground']:
                cfg['task_name'] = 'cam4_robot_place_bottle_8p'
                cfg['num_points'] = 8
            elif current_task in ['1220_pick_bottle_from_side_door_new']:
                cfg['task_name'] = 'cam1_robot_new'
                cfg['num_points'] = 7
            points_class = PointsClass(**cfg)

            request = {
                "image": serialized_image,
                "image_path": "",
                "query": f"Get the bounding box of the robot (only get one)"
                # "query": "Get the bounding box of the coke bottle"
            }
            socket.send_json(request)
            response = socket.recv_json()
            bbox_plate = response["result"]
            bbox_plate = bbox_plate[:-1] if len(bbox_plate) == 5 else bbox_plate
            print("bbox_plate", bbox_plate)
            print(type(bbox_plate))
            points_class.add_to_image_list(frame)
            points_class.find_semantic_similar_points(object_bbox=bbox_plate)

            # points_class.add_to_image_list(frame)
            # points_class.find_semantic_similar_points()
            second_points = points_class.semantic_similar_points
            second_points_num = cfg['num_points']
            total_points_num = first_points_num + second_points_num
            print("found semantic similar points", second_points)
            # append first points to the semantic similar points
            # total_points = torch.cat((first_points, second_points), dim=0)
            total_points = torch.cat((second_points, first_points), dim=0)
            print("total_points", total_points)

            self.points_class.num_points = total_points_num
            # self.points_class.tracks = total_points
        #     self.points_class.semantic_similar_points = torch.Tensor([[  0.,  69., 113.],
        # [  0.,  89.,  94.],
        # [  0.,  90.,  55.],
        # [  0.,  87.,  59.],
        # [  0.,  76.,  66.],
        # [  0., 154., 238.],
        # [  0., 126., 222.],
        # [  0., 130., 189.],
        # [  0., 113., 202.],
        # [  0., 166., 190.]])
            self.points_class.semantic_similar_points = total_points
        #     self.points_class.semantic_similar_points = torch.Tensor([[  0.,  69., 109.],
        # [  0.,  90.,  85.],
        # [  0.,  90.,  45.],
        # [  0.,  87.,  55.],
        # [  0.,  81.,  61.],
        # [  0., 169., 259.],
        # [  0., 151., 249.],
        # [  0., 151., 216.],
        # [  0., 135., 225.],
        # [  0., 189., 219.]])

        # frame = obs[self.pixel_keys[0]]
        # shape = frame.shape
        # frame = frame[(shape[0] - int(shape[0] * (1 - .25))) : , (shape[1] - int(shape[1] * (1 - .3))) : ]
        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        # self.points_class.find_semantic_similar_points(object_bbox=bbox_plate)
        # self.points_class.find_semantic_similar_points()
        # print("found semantic similar points--------------------", self.points_class.semantic_similar_points)
        self.points_class.track_points(is_first_step=True)
        self.points_class.track_points()
        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()

        print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;multiobject p3po done")
        # plot
        image = self.points_class.plot_image()
        # exit()

        obs['graph'] = self.points_class.get_points()[-1]
        obs = self._env._replace(observation, observation=obs)


        return obs

    def step(self, action):
        # print("p3po current step's action is: ", action)
        if self.isFirstStep:
            self.isFirstStep = False
            print("p3po first step")

        observation = self._env.step(action)
        obs = observation.observation

        if obs[self.pixel_keys[0]].shape[0] == 3:
            obs[self.pixel_keys[0]] = np.transpose(obs[self.pixel_keys[0]], (1, 2, 0))

        self.points_class.add_to_image_list(obs[self.pixel_keys[0]])
        self.points_class.track_points()
        # import ipdb; ipdb.set_trace()
        if len(self.depth_keys) > 0 and self.depth_keys[0] in obs:
            self.points_class.set_depth(obs[self.depth_keys[0]])
        else:
            self.points_class.get_depth()

        obs['graph'] = self.points_class.get_points()[-1]
        obs = self._env._replace(observation, observation=obs)

        # plot
        image = self.points_class.plot_image()
        # import ipdb; ipdb.set_trace()
        # exit()

        return obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
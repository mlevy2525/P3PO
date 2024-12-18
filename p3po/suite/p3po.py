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
        obs = self.reset(flag=1).observation
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
            with open("/mnt/robotlab/siddhant/P3PO/p3po/cfgs/suite/p3po.yaml") as stream:
                try:
                    cfg = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # connect to reasoning server at port 8888 at 172.24.71.224 ############################
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://172.24.71.224:6000")

            # Initialize the PointsClass object
            cfg['task_name'] = 'bottle' #'plate_12p'  #"plate_14p"
            cfg['num_points'] = 5 #12 #14
            points_class = PointsClass(**cfg)

            # # # # send the image to the reasoning server and get the bounding box ########################
            image = obs[self.pixel_keys[0]]
            frame = image

            # # convert np array to image
            image = Image.fromarray(image)
            serialized_image = serialize_image(image)
            request = {
                "image": serialized_image,
                "image_path": "",
                "query": "Get the bounding box of the bottle"
            }
            socket.send_json(request)
            response = socket.recv_json()
            bbox_plate = response["result"]
            bbox_plate = bbox_plate[:-1] if len(bbox_plate) == 5 else bbox_plate
            print("bbox_plate", bbox_plate)
            print(type(bbox_plate))
            # # exit()


            # TODO: get object_bbox from reasoning server
            # bbox_plate = [288.23318481, 126.03538513, 359.97091675, 257.01574707]
            # bbox_plate = [246.25900269, 103.23423004, 317.26687622, 219.7673645 ]
            # bbox_plate = [131.79145813,191.11541748,192.15452576,265.63516235]
            # bbox_plate = [141.8319091796875, 246.34805297851562, 184.5408935546875, 309.7018737792969]
            # bbox_plate = [117.40306854, 176.61445618, 169.01112366, 244.98728943] # letter cup
            # bbox_plate = [202.83328247070312, 279.2685546875, 217.8385009765625, 331.46942138671875]
            # bbox_plate = [244.5233612060547, 92.18417358398438, 268.80908203125, 157.75901794433594]
            # bbox_plate = [217.63137817382812, 82.5777587890625, 237.97607421875, 148.17152404785156]
            # bbox_plate = [233.92860412597656, 58.02970504760742, 255.9622039794922, 124.33071899414062]
            points_class.add_to_image_list(frame)
            points_class.find_semantic_similar_points(object_bbox = bbox_plate)
            print("found semantic similar points", points_class.semantic_similar_points)
            print(type(points_class.semantic_similar_points))
            first_points = points_class.semantic_similar_points
            first_points_num = cfg['num_points']

            # initialize a new point tracking
            cfg["task_name"] = 'cam4_robot' #'robot_and_rack_8p'  #"rack_and_robot"
            cfg["num_points"] = 5 #8 #10
            points_class = PointsClass(**cfg)
            points_class.add_to_image_list(frame)
            points_class.find_semantic_similar_points()
            second_points = points_class.semantic_similar_points
            second_points_num = cfg['num_points']
            total_points_num = first_points_num + second_points_num
            print("found semantic similar points", second_points)
            # append first points to the semantic similar points
            # total_points = torch.cat((first_points, second_points), dim=0)
            total_points = torch.cat((second_points, first_points), dim=0)
            print("total_points", total_points)

            self.points_class.num_points = total_points_num
            self.points_class.tracks = total_points
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
        print("p3po current step's action is: ", action)
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
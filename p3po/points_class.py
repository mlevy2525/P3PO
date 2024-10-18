import numpy as np
import sys
import pickle
from PIL import Image
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utilities.correspondence import Correspondence
from utilities.depth import Depth

class PointsClass():
    def __init__(self, root_dir, task_name, device, width, height, image_size_multiplier, ensemble_size, dift_layer, dift_steps, num_points, **kwargs):
        """
        Initialize the Points Class for finding key points in the episode.

        Parameters:
        -----------
        root_dir : str
            The root directory for the github repository.

        task_name : str
            The name of the task done by the robot.
            
        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU acceleration).
            
        width : int
            The width that should be used in the correspondence model.
            
        height : int
            The height that should be used in the correspondence model.

        image_size_multiplier : int
            The size multiplier for the image in the correspondence model.
            
        ensemble_size : int
            The size of the ensemble for the DIFT model.
            
        dift_layer : int
            The specific layer of the DIFT model to use for feature extraction.

        dift_steps : int
            The number of steps or iterations for feature extraction in the DIFT model.
        """

        # Set up the correspondence model and find the expert image features
        self.correspondence_model = Correspondence(device, root_dir + "/dift/", width, height, image_size_multiplier, ensemble_size, dift_layer, dift_steps)
        try:
            self.initial_coords = np.array(pickle.load(open("%s/coordinates/coords/%s.pkl" % (root_dir, task_name), "rb")))
        except Exception as e:
            print(e)
            print("Setting coordinates to random values")
            if num_points == -1:
                num_points = 100
            self.initial_coords = np.random.rand(num_points, 3) * 256
            self.initial_coords[:, 0] = 0

        try:
            expert_image = Image.open("%s/coordinates/images/%s.png" % (root_dir, task_name)).convert('RGB')
        except Exception as e:
            print(e)
            print("Setting expert image to random values")
            expert_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))

        self.expert_correspondence_features = self.correspondence_model.set_expert_correspondence(expert_image)

        # Set up the depth model
        self.depth_model = Depth(root_dir + "/Depth-Anything-V2/", device)

        # Set up cotracker
        sys.path.append(root_dir + "/co-tracker/")
        from cotracker.predictor import CoTrackerOnlinePredictor
        self.cotracker = CoTrackerOnlinePredictor(checkpoint=root_dir + "/co-tracker/checkpoints/cotracker2v1.pth", window_len=16).to(device)


        self.transform = transforms.Compose([ 
                            transforms.PILToTensor()])
        self.image_list = torch.tensor([]).to(device)
        self.depth = np.array([])

        if num_points == -1:
            self.num_points = self.initial_coords.shape[0]
        else:
            self.num_points = num_points

        self.device = device

    # Image passed in here must be in RGB format
    def add_to_image_list(self, image):
        """
        Add an image to the image list for finding key points.

        Parameters:
        -----------
        image : np.ndarray
            The image to add to the image list. This image must be in RGB format.
        """

        pil_image = Image.fromarray((image))
        transformed = (self.transform(pil_image) / 255)

        # We only want to track the last 16 images so pop the first one off if we have more than 16
        if self.image_list.shape[0] > 0 and self.image_list.shape[1] == 16:
            self.image_list = self.image_list[:, 1:]

        # If it is the first image you want to repeat until the whole array is full
        # Otherwise it will just add the new image to the end of the array
        while self.image_list.shape[0] == 0 or self.image_list.shape[1] < 16:
            self.image_list = torch.cat((self.image_list, transformed.unsqueeze(0).unsqueeze(0).clone().to(self.device)), dim=1)

    def reset_episode(self):
        """
        Reset the image list for finding key points.
        """

        self.image_list = torch.tensor([]).to(self.device)
        self.depth = torch.tensor([]).to(self.device)

    def find_semantic_similar_points(self):
        """
        Find the semantic similar points between the expert image and the current image.
        """

        self.semantic_similar_points = self.correspondence_model.find_correspondence(self.expert_correspondence_features, self.image_list[0, -1], self.initial_coords)

    def get_depth(self, last_n_frames=1):
        """
        Get the depth map for the current image using Depth Anything. Depth is height x width.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode
        """

        self.depth = np.zeros((last_n_frames, self.image_list.shape[3], self.image_list.shape[4]))
        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            numpy_image = self.image_list[0, frame_idx].cpu().numpy().transpose(1, 2, 0) * 255
            depth = self.depth_model.get_depth(numpy_image)
            self.depth[frame_idx] = depth

    def set_depth(self, depth):
        """
        If you are using ground truth depth, you can set the depth here.

        Parameters:
        -----------
        depth : np.ndarray
            The depth map for the current image. Depth is height x width.
        """

        if self.image_list.shape[0] == 8:
            self.depth = self.depth[1:]

        while self.depth.shape[0] < 8:
            if self.depth.shape[0] == 0:
                self.depth = depth[None, ...].copy()
            self.depth = np.concatenate((self.depth, depth[None, ...].copy()), axis=0)

    def track_points(self, is_first_step=False, one_frame=True):
        """
        Track the key points in the current image using the CoTracker model.

        Parameters:
        -----------
        is_first_step : bool
            Whether or not this is the first step in the episode.
        """

        if is_first_step:
            self.cotracker(video_chunk=self.image_list[0, 0].unsqueeze(0).unsqueeze(0), 
                           is_first_step=True, 
                           add_support_grid=True, 
                           queries=self.semantic_similar_points[None].to(self.device))
            self.tracks = self.semantic_similar_points
        else:
            tracks, _ = self.cotracker(self.image_list, one_frame=one_frame)
            # Remove the support points
            tracks = tracks[:, :, 0:self.num_points, :]

            self.tracks = tracks

    def get_points(self, last_n_frames=1):
        """
        Get the list of points for the current frame.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        final_points : torch.Tensor
            The list of points for the current frame.
        """

        final_points = torch.zeros((last_n_frames, self.num_points, 3))
        width = self.image_list.shape[4]
        height = self.image_list.shape[3]

        for frame_num in range(last_n_frames):
            for point in range(self.num_points):
                frame_idx = -1 * (last_n_frames - frame_num)
                try:
                    depth = self.depth[frame_idx, int(self.tracks[0, frame_idx, point][1]), int(self.tracks[0, frame_idx, point][0])]
                except:
                    depth = 0

                x = (self.tracks[0, frame_idx, point][0] - width/2) * depth
                y = (self.tracks[0, frame_idx, point][1] - height/2) * depth
                x /= (width/2)
                y /= (height/2)

                final_points[frame_num, point] = torch.tensor([x, y, depth])

        return final_points.reshape(last_n_frames, -1)

    def plot_image(self, last_n_frames=1):
        """
        Plot the image with the key points overlaid on top of it. Running this will slow down your tracking, but it's good for debugging.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        img_list : list
            A list of images with the key points overlaid on top of them.
        """

        img_list = []

        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            curr_image = self.image_list[0, frame_idx].cpu().numpy().transpose(1, 2, 0) * 255

            fig, ax = plt.subplots(1)
            ax.imshow(curr_image.astype(np.uint8))

            rainbow = plt.get_cmap('rainbow')
            # Generate n evenly spaced colors from the colormap
            colors = [rainbow(i / self.tracks.shape[2]) for i in range(self.tracks.shape[2])]

            for idx, coord in enumerate(self.tracks[0, frame_idx]):
                ax.add_patch(patches.Circle((coord[0].cpu(), coord[1].cpu()), 5, facecolor=colors[idx], edgecolor="black"))
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_list.append(img.copy())
            plt.close()

        return img_list

import sys
sys.path.append("../")

import pickle
import cv2
import yaml
import imageio

from points_class import PointsClass

#TODO: Set if you want to read from a pickle or from mp4 files
# If you are reading from a pickle please make sure that the images are RGB not BGR
read_from_pickle = False
pickle_path = ""
pickle_image_key = ""

#Otherwise we need to add videos to a list
#TODO: A list of videos to read from if you are not loading data from a pickle
video_paths = []

#TODO: Set to true if you want to save a video of the points being tracked
write_videos = True

# If you want to subsample the frames, set the subsample rate here
subsample = 3

with open("../cfgs/config.yaml") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if read_from_pickle:
    examples = pickle.load(open(pickle_path, "rb"))
    num_demos = len(examples['observations'])
else:
    num_demos = len(video_paths)

# Initialize the PointsClass object
points_class = PointsClass(**cfg)
episode_list = []

for i in range(num_demos):
    # Read the frames from the pickle or video, these frames must be in RGB so if reading from a pickle make sure to convert if necessary
    if read_from_pickle:
        frames = examples['observations'][i][pickle_image_key]
    else:
        frames = []
        video = cv2.VideoCapture(video_paths[i])
        subsample_counter = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if subsample_counter % subsample == 0:
                # CV2 reads in BGR format, so we need to convert to RGB
                frames.append(frame[:, :, ::-1])
            subsample_counter += 1
        video.release()

    points_class.add_to_image_list(frames[0])
    points_class.find_semantic_similar_points()
    points_class.track_points(is_first_step=True)
    points_class.track_points(one_frame=False)
    points_class.get_depth()

    points_list = []
    points = points_class.get_points()
    points_list.append(points[0])

    if write_videos:
        video_list = []
        image = points_class.plot_image()
        video_list.append(image[0])

    mark_every = 1
    for idx,image in enumerate(frames[1:]):
        points_class.add_to_image_list(image)
        if (idx + 1) % mark_every == 0 or idx == (len(frames) - 1):
            to_add = mark_every - (idx + 1) % 8
            if to_add < mark_every:
                for j in range(to_add):
                    points_class.add_to_image_list(image)
            else:
                to_add = 0

            points_class.track_points(one_frame=False)
            points_class.get_depth(last_n_frames=8)

            points = points_class.get_points(last_n_frames=8)
            for j in range(mark_every - to_add):
                points_list.append(points[j])

            if write_videos:
                images = points_class.plot_image(last_n_frames=8)
                for j in range(mark_every - to_add):
                    video_list.append(images[j])

    if write_videos:
        imageio.mimsave(f"videos/{cfg['task_name']}_%d.mp4" % i, video_list, fps=30)
    
    episode_list.append(points_list)
    points_class.reset_episode()

pickle.dump(episode_list, open(f"{cfg['root_dir']}/processed_data/points/{cfg['task_name']}.pkl", "wb"))

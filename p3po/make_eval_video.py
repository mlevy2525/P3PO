import os
import imageio
import re
import pickle
import cv2

# Directory containing images
image_dir = '/home/ademi/eval_dump/plotted_images'
# Output path for the MP4 video
output_video = 'eval_rollout_video.mp4'
preprocessed_data_dir = '/home/ademi/hermes/data/pick_block_20241103a_preprocessed_30hz'
subsample = 3

# Regular expression to capture the number in filenames like 'image_x', 'image_xx', 'image_xxx'
def extract_number(filename):
    match = re.search(r'image_(\d+)', filename)
    return int(match.group(1)) if match else -1

# List all image file names in the directory and sort by the number in the filename
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))], key=extract_number)

# Load the first image to get the dimensions (assuming all images have the same dimensions)
first_image = imageio.imread(os.path.join(image_dir, image_files[0]))
height, width, _ = first_image.shape

# Create a writer object for the MP4 file with FFMPEG
writer = imageio.get_writer(output_video, fps=30, codec='libx264')  # Set fps (frames per second)

# Append each image to the writer
for frame_num, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    img = imageio.imread(image_path)
    img = cv2.putText(img, f'Frame {frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    writer.append_data(img)

# Close the writer to finalize the video
writer.close()

print(f"Video saved as {output_video}")

# # load pkl file called nearest_neighbor_info in eval_dump
# nearest_neighbor_info = pickle.load(open('eval_dump/nearest_neighbor_info.pkl', 'rb'))
# images = []
# for demo_num, idx in nearest_neighbor_info:
#     demo_images = sorted([f for f in os.listdir(f'{preprocessed_data_dir}/demonstration_{demo_num}/cam_3_rgb_images') if f.endswith(('.png', '.jpg', '.jpeg'))], key=extract_number)
#     image = imageio.imread(f'{preprocessed_data_dir}/demonstration_{demo_num}/cam_3_rgb_images/{demo_images[idx.item()]}')
#     images.append(image)

# # make a video of the images with number of frame overlayed in the top left corner
# output_video = 'eval_rollout_nearest_neighbor_video.mp4'
# writer = imageio.get_writer(output_video, fps=30, codec='libx264')
# for idx, img in enumerate(images):
#     img = cv2.putText(img, f'Frame {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     writer.append_data(img)
# writer.close()
# print(f"Video saved as {output_video}")




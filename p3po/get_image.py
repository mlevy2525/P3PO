import cv2
import numpy as np
from openteach.utils.network import ZMQCameraSubscriber
from xarm_env.envs.constants import HOST_ADDRESS, CAMERA_PORT_OFFSET, CAM_SERIAL_NUMS

class CameraCapture:
    def __init__(self):     
        # Camera subscribers
        self.image_subscribers = []
        for cam_idx in list(CAM_SERIAL_NUMS.keys()):
            port = CAMERA_PORT_OFFSET + cam_idx
            self.image_subscribers.append(
                ZMQCameraSubscriber(
                    host=HOST_ADDRESS,
                    port=port,
                    topic_type="RGB",
                )
            )
    
    def capture_images(self):
        """Capture images from all configured cameras."""
        image_list = []
        for idx, subscriber in enumerate(self.image_subscribers):
            print(f"Capturing image from camera {idx}")
            image = subscriber.recv_rgb_image()[0]
            resized_image = image
            image_list.append(resized_image)
        return image_list

if __name__ == "__main__":
    camera_capture = CameraCapture()
    images = camera_capture.capture_images()

    # save images
    for idx, image in enumerate(images):
        cv2.imwrite(f"camera_{idx}.jpg", image)
        print(f"Saved image {idx} to camera_{idx}.jpg")

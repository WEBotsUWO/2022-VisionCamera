#pyrealsense libraries
import cv2
import pyrealsense2 as rs
from realsense_depth import *

import numpy as np

#picture imports
from PIL import Image
from resizeimage import resizeimage

from resnet50_v2 import Resnet50
import time
from datetime import datetime


class ModelCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.r = Resnet50()
        self.r.setup()

    def takePicture(self, x=1280, y=720):
        point = (x,y)
        frames = self.pipeline.wait_for_frames()
        color_frame = np.asanyarray(frames.get_color_frame().get_data())
        print(type(color_frame))
        image = cv2.resize(color_frame, dsize=(x, y), interpolation=cv2.INTER_CUBIC)

        return image

    def predict(self):
        image = self.takePicture()
        predictions = self.r.detect(image)
        return predictions
        

if __name__ == '__main__':
    test = ModelCamera()
    im = Image.fromarray(test.takePicture(0,0))
    im.show()
    # test.predict() will be returning a dictionary of np arrays
    # the following fields can be accessed in the dictionary: 
    # - num_detections: an int np.array with only one value, the number of detections [N].
    # - detection_boxes: a float np.array of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
    # - detection_classes: a string np.array of shape [N] containing detection class index from the label file.
    # - detection_scores: a float np.array of shape [N] containing detection scores.
    output = test.predict()
    print(output)
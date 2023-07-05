from turtle import width

import matplotlib.pyplot as plt
from resnet50_v2 import Resnet50
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

if __name__ == '__main__':

    testPic = 'cityTest.jpg'


    testModel = Resnet50()
    img = Image.open(testPic)
    xLen = img.width
    yLen = img.height
    img_np = np.asarray(img)

    image_np_with_detections = img_np.copy()

    detections = testModel.detectRaw(img)

    category_index = dict()
    for i in range(len(Resnet50.classes_91)):
        category_index[i] = {'id': 1, 'name': Resnet50.classes_91[i]}

    boxes = []
    for i in detections['detection_boxes'][0]:
        boxes.append([])
        for k in range(len(i)):
            if k%2 == 0:
                boxes[-1].append(int(i[k]*yLen))
            else:
                boxes[-1].append(int(i[k]*xLen))
    
    boxes = np.asarray(boxes)
    print(boxes)

    #return this image for the object
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
        )

    plt.figure(figsize=(12,16))
    plt.imshow(image_np_with_detections)
    #plt.show() can;t get rid of the matplotlib is currently using agg issue
    plt.savefig('predictions.png')

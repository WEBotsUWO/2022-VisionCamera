
import tensorflow_hub as hub
from PIL import Image
from tensorflow import convert_to_tensor
import numpy as np
#TO-DO for v3: make it only use cv2 and np to avoid having to run tensorflow on raspberry pi
#please see https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1
class Resnet50:
    
    classes_91 = ["background", "person", "bicycle", "car", "motorcycle",
                "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"] 

    def __init__(self, modelPath='https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1'):
        self.detector = hub.load(modelPath)

    #assume img is a PIL object
    def detect(self, img):
        #converting shape to [1, height, width, 3], PIL opened as np array is usually shape of [height, widht, 3]
        img_tensor = convert_to_tensor(np.array([np.asarray(img)]))
        detector_output = self.detector(img_tensor)

        #replace tensors with np arrays
        for i in detector_output.keys():
            detector_output[i] = detector_output[i].numpy()
        #converting detection_classes from indices into string labels
        try:
            labels = []
            for i in range(len(detector_output['detection_classes'][0])):
                labels.append(Resnet50.classes_91[int(detector_output['detection_classes'][0][i])])
            detector_output['detection_classes'] = [labels]
        except Exception as e:
            print('unable to convert class indices into the corresponding string labels. loaded model is not trained on COCO17 dataset')
            print(e)
        
        return detector_output
    #similar to detect, but does not do any conversions and just as if it was the raw tf model output
    def detectRaw(self, img):
        #converting shape to [1, height, width, 3], PIL opened as np array is usually shape of [height, widht, 3]
        img_tensor = convert_to_tensor(np.array([np.asarray(img)]))
        detector_output = self.detector(img_tensor)
        return detector_output
        
if __name__ == '__main__':

    testModel = Resnet50()
    img = Image.open('cityTest.jpg')
    prediction = testModel.detect(img)
    class_ids = prediction["detection_classes"]
    print(class_ids)
    
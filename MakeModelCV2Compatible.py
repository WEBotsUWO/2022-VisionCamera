import cv2 
cv2.dnn.writeTextGraph('centernet_resnet50v2_512x512.pb', 'centernet_resnet50v2_512x512_graph.pbtxt')

cv2.dnn.readNetFromTensorflow('centernet_resnet50v2_512x512.pb', 'centernet_resnet50v2_512x512_graph.pbtxt')
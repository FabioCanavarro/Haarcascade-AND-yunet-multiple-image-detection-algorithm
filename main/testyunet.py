import cv2 as cv
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from deepface import DeepFace as df
import multiprocessing as mp
class YuNet:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return np.array([]) if faces[1] is None else faces[1]
    
def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]
    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)
    return output
def pad_image_to_target(image, target_shape, pad_value=0):
    height_diff = max(0, target_shape[0] - image.shape[0])
    width_diff = max(0, target_shape[1] - image.shape[1])

    top_pad = height_diff // 2
    bottom_pad = height_diff - top_pad
    left_pad = width_diff // 2
    right_pad = width_diff - left_pad

    
    padding = ((top_pad, bottom_pad), (left_pad, right_pad))

    if len(image.shape) == 3:
        padding += ((0, 0),)

    padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)

    return padded_image
def totaldetect(frames):
    amount = 0
    frame = frames.copy()
    face_rect = face_classifier.detectMultiScale(frame,1.15,minNeighbors=3)
    for (x,y,w,h) in face_rect:
        face_rect = face_classifier.detectMultiScale(frame,1.15,minNeighbors=3)
        frames = frame[y-5:y+h+5,x-5:x+w+5]
        h, w, _ = frames.shape
        model.setInputSize([w, h])
        results = model.infer(frames)
        xyy = visualize(frames,results)
        xyy = pad_image_to_target(xyy,(frame[y-5:y+h+5,x-5:x+w+5].shape[0],frame[y-5:y+h+5,x-5:x+w+5].shape[1]))
        if results.shape[0] != 0:
            frame[y-5:y+h+5,x-5:x+w+5] = xyy
        

        amount+=results.shape[0]

    return frame , amount


model = YuNet(modelPath=r"models\face_detection_yunet_2023mar.onnx",inputSize=[640, 480])
frame = cv2.imread(r"Results\Input.jpg")
face_classifier = cv2.CascadeClassifier(r"models\haarcascade_frontalface_default.xml")

# for testing in jupyter notebook
# def show(img,figsize=(20,10)):
#     """ Takes an image array as a parameter and shows the image but bigger, useful in jupyter"""
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111)
#     ax.imshow(img,cmap= 'gray')


detected = totaldetect(frame)




cv2.imshow("frame",detected[0])
print(detected[1])
while True:
    a = cv2.waitKey(1) & 0XFF
    if a == 27:
        cv2.destroyAllWindows()
        break

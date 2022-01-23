from keras.models import load_model
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CnnClassficationResult:
    confidence: float
    resultString: str
    img: object


class CnnClassifier:
    def __init__(self):
        self.model = load_model('handrecognition_cnn_model.h5')
        self.model.summary()

    def classify(self, tested_img):
        img_size = 50
        image = cv2.resize(tested_img, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        prediction = self.model.predict(np.array([image]))
        print(prediction)
        class_det = np.argmax(prediction, axis=1)
        cnn_class_names = ["PointingFinger", "Victory", "Hello", "Thumb", "Horns", "OK"]
        result = CnnClassficationResult(
            confidence=max(max(prediction)),
            resultString=cnn_class_names[class_det[0]],
            img=image
        )
        # cv2.imshow('test',image)
        return result
        # result.confidence = max(max(prediction))
        # result.resultString = cnn_class_names[class_det[0]]
        # result.img = image
        print('Detected ' + result.resultString + ' with confidence ' + str(result.confidence))

# classifier = CnnClassifier()
# img = cv2.imread('./Traindata/Gesture_0/gest1_1.jpg')
# img = cv2.imread('./FeatureBased/ToDetect/victory_test.png')
# classifier.classify(img)


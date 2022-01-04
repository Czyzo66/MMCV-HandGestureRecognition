from keras.models import load_model
import cv2
import numpy as np

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
        result = CnnClassficationResult()
        result.confidence = max(max(prediction))
        cnn_class_names = ["PointingFinger", "Victory", "Hello", "Thumb", "Horns", "OK"]
        pred = prediction.tolist()
        result.resultString = cnn_class_names[pred.index(max(pred))]
        result.img = image
        print('Detected ' + result.resultString + ' with confidence ' + str(result.confidence))

# classifier = CnnClassifier()
# img = cv2.imread('./Traindata/Gesture_0/gest1_1.jpg')
# img = cv2.imread('./FeatureBased/ToDetect/victory_test.png')
# classifier.classify(img)


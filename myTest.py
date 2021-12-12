import tensorflow as tf
import dataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

data_path = os.path.abspath('../gesty')
os.path.exists(data_path)

batch_size = 32
class_names = ["Gesture_0", "Gesture_1"]
num_classes = len(class_names)
img_size = 50
num_channels = 3
validation_size = 0.2

data = dataLoader.read_train_sets(data_path, img_size, class_names, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


##Creation of the model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data.train.images,
          data.train.labels,
          epochs=5,
          batch_size=batch_size,
          verbose=2,
          validation_data=(data.train.images, data.train.labels))
model.save('handrecognition_model.h5')

test_loss, test_acc = model.evaluate(data.valid.images, data.valid.labels)
print('Test acc: {:2.2f}%'.format(test_acc*100))

predictions = model.predict(data.valid.images)
test = np.argmax(predictions[0]), data.valid.labels[0]
print('hello')

def validate(predictions, label_array, img_array):
    class_names = ["Victory", "Hello"]
    plt.figure(figsize=(15,5))

    for i in range(1,10):
        prediction = predictions[i]
        label = label_array[i]
        img = img_array[i]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        plt.subplot(3,3,i)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img,cmap=plt.cm.binary)

        predicted_label = np.argmax(prediction)  # Get index of the predicted label from prediction

        # Change color of title based on good prediction or not
        if predicted_label == label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],
                                                              100 * np.max(prediction),
                                                              class_names[label]),
                   color=color)

    plt.show()

validate(predictions, data.valid.labels, data.valid.images)

import cnnDataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

data_path = os.path.abspath('./Traindata')
os.path.exists(data_path)

batch_size = 32
class_names = ["Gesture_0", "Gesture_1", "Gesture_2", "Gesture_3", "Gesture_4", "Gesture_5"]
global cnn_class_names
cnn_class_names = ["PointingFinger", "Victory", "Hello", "Thumb", "Horns", "OK"]
num_classes = len(class_names)
img_size = 50
num_channels = 3
validation_size = 0.2

data = cnnDataLoader.read_train_sets(data_path, img_size, class_names, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


##Creation of the model
model = Sequential()

# Convolutional layer + RELU
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(50, 50, 1)))
# Max Pooling
model.add(MaxPooling2D((2, 2)))
# Convolutional layer + RELU
model.add(Conv2D(64, (3, 3), activation='relu'))
# Max Pooling
model.add(MaxPooling2D((2, 2)))
# Convolutional layer + RELU
model.add(Conv2D(64, (3, 3), activation='relu'))
# Max Pooling
model.add(MaxPooling2D((2, 2)))
# Fully Connected
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
print('Saving model...')
model.summary()
model.save('handrecognition_cnn_model.h5')
print('Saved model')
print('Test model...')
test_loss, test_acc = model.evaluate(data.valid.images, data.valid.labels)
print('Accuracy: {:2.2f}%'.format(test_acc*100))

predictions = model.predict(data.valid.images)
test = np.argmax(predictions[0]), data.valid.labels[0]

def validate(predictions, label_array, img_array):
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

        plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(cnn_class_names[predicted_label],
                                                              100 * np.max(prediction),
                                                              cnn_class_names[label]),
                   color=color)

    plt.show()

validate(predictions, data.valid.labels, data.valid.images)

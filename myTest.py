import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

DATA_DIR = os.path.abspath('../gesty')
os.path.exists(DATA_DIR)

imagepaths = []
image_data = []
labels = []

dirlist = os.listdir(DATA_DIR)

def plot_image(path):
    img = cv2.imread(path)  # Reads the image into a numpy.array
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (RGB)
    print(img_cvt.shape)  # Prints the shape of the image just to check
    plt.grid(False)  # Without grid so we can see better
    plt.imshow(img_cvt)  # Shows the image
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image " + path)


for dir in os.listdir(DATA_DIR):
    if dir == '.DS_Store':
        continue

    inner_dir = os.path.join(DATA_DIR, dir)
    csv_file = pd.read_csv(os.path.join(inner_dir, "LIST-" + dir + '.csv'), sep=';')
    for row in csv_file.iterrows():
        img_path = os.path.join(inner_dir, row[1].Filename)
        imagepaths.append(img_path)
        labels.append(row[1].ClassId)

# plot_image(imagepaths[0])

for path in imagepaths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_data.append(img)
    plt.imshow(img)

image_data = np.array(image_data, dtype="uint8")
image_data = image_data.reshape(len(imagepaths),50,50,1)
labels = np.array(labels)

print("Images loaded: ", len(image_data))
print("Labels loaded: ", len(labels))

print(labels[0], imagepaths[0])

image_data_train, image_data_test, labels_train, labels_test = train_test_split(image_data, labels, test_size=0.3, random_state=42)

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

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(image_data_train,labels_train,epochs=5,batch_size=64,verbose=2,validation_data=(image_data_test,labels_test))
model.save('handrecognition_model.h5')

test_loss, test_acc = model.evaluate(image_data_test,labels_test)
print('Test acc: {:2.2f}%'.format(test_acc*100))

predictions = model.predict(image_data_test)
test = np.argmax(predictions[0]), labels_test[0]
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

validate(predictions, labels_test,image_data_test)
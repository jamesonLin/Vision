import json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf



with open('train_dataset/train.json') as f:
    data = json.load(f)

trafficLightFiles = np.array([])
trafficLightLabel = np.array([])

for i in data['annotations']:
    filename = i['filename']
    label = i['label']
    bndbox = i['bndbox']
    xmin = i['bndbox']['xmin']
    ymin = i['bndbox']['ymin']
    xmax = i['bndbox']['xmax']
    ymax = i['bndbox']['ymax']
    if len(i['inbox']) != 0:
        color = [j['color'] for j in i['inbox']]
        inboxBNDBOX = [j['bndbox'] for j in i['inbox']]
        inboxXMIN = [j['bndbox']['xmin'] for j in i['inbox']]
        inboxYMIN = [j['bndbox']['ymin'] for j in i['inbox']]
        inboxXMAX = [j['bndbox']['xmax'] for j in i['inbox']]
        inboxYMAX = [j['bndbox']['ymax'] for j in i['inbox']]
        trafficLightFiles = np.append(trafficLightFiles, filename)
        trafficLightLabel = np.append(trafficLightLabel, label)

# print(trafficLightFiles)
# print(trafficLightLabel)
# print(trafficLightColor)

# print(len(trafficLightFiles))
# print(len(trafficLightLabel))
# print(len(trafficLightColor))


# imgTrain = cv2.imread('train_dataset/' + filename)
# # print(len(imgTrain))
# imageTrain = cv2.resize(imgTrain, (200, 200))

# rgb = cv2.cvtColor(imageTrain, cv2.COLOR_BGR2RGB)

# rgbTensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

# rgbTensor = tf.expand_dims(rgbTensor, 0)

model = Sequential([
Conv2D(32, 3, input_shape=(64, 64, 3)),
MaxPooling2D(pool_size=2),
Conv2D(64, 3),
MaxPooling2D(pool_size=2),
Conv2D(128, 3),
MaxPooling2D(pool_size=2),
Flatten(),
Dense(256, activation='relu'),
Dense(1, activation='sigmoid')])

train = ImageDataGenerator(rescale = 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)
trainDataSet = train.flow_from_directory("train_dataset",
                                        target_size = (64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary')

validationData = train_test_split(trainDataSet, trafficLightLabel, test_size=0.1, random_state=42)

test = ImageDataGenerator(rescale = 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)
testDataSet = test.flow_from_directory("test_dataset",
                                        target_size = (64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary')


# plt.figure(figsize=(10, 10))
# for images, labels in trainDataSet:
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i])
#     plt.title(trafficLightLabel[i])
#     plt.axis("off")


model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x = trainDataSet, validation_data = validationData, epochs = 5)
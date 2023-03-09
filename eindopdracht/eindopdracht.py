import json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import cv2
import matplotlib.pyplot as plt



with open('train_dataset/train.json') as f:
    data = json.load(f)

filenameAnnotations = {}

for i in data['annotations']:
    filename = i['filename']
    bndbox = i['bndbox']
    xmin = i['bndbox']['xmin']
    ymin = i['bndbox']['ymin']
    xmax = i['bndbox']['xmax']
    ymax = i['bndbox']['ymax']
    if filename not in filenameAnnotations:
        filenameAnnotations[filename] = []
    filenameAnnotations[filename].append(i)
    if len(i['inbox']) != 0:
        color = [j['color'] for j in i['inbox']]
        inboxBNDBOX = [j['bndbox'] for j in i['inbox']]
        inboxXMIN = [j['bndbox']['xmin'] for j in i['inbox']]
        inboxYMIN = [j['bndbox']['ymin'] for j in i['inbox']]
        inboxXMAX = [j['bndbox']['xmax'] for j in i['inbox']]
        inboxYMAX = [j['bndbox']['ymax'] for j in i['inbox']]

validFilenames = np.array([])
invalidFilenames = np.array([])
validLabels = np.array([])
invalidLabels = np.array([])

for j, i in filenameAnnotations.items():
    hasNonEmptyInbox = any(len(x['inbox']) != 0 for x in i)
    # print(hasNonEmptyInbox)
    if hasNonEmptyInbox:
        validFilenames = np.append(validFilenames, j)
        validLabels = np.append(validLabels, 'traffic light')
    else:
        invalidFilenames = np.append(invalidFilenames, j)
        invalidLabels = np.append(invalidLabels, 'no traffic light')

# print("Valid labels:")
# print(validFilenames)
# print(len(validFilenames))
# print(len(validLabels))

# print("Invalid labels:")
# print(invalidFilenames)
# print(len(invalidFilenames))
# print(len(invalidLabels))

# imgTrain = cv2.imread('train_dataset/' + filename)
# # print(imgTrain)
# imageTrain = cv2.resize(imgTrain, (200, 200))

# rgb = cv2.cvtColor(imageTrain, cv2.COLOR_BGR2RGB)

# rgbTensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

# rgbTensor = tf.expand_dims(rgbTensor, 0)

model = Sequential([
Conv2D(32, 3, input_shape=(200, 200, 3)),
MaxPooling2D(pool_size=2),
Conv2D(64, 3),
MaxPooling2D(pool_size=2),
Conv2D(128, 3),
MaxPooling2D(pool_size=2),
Flatten(),
Dense(256, activation='relu'),
Dense(1, activation='sigmoid')])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# train = ImageDataGenerator(rescale = 1./255)
# trainDataSet = train.flow_from_directory("train_dataset",
#                                         target_size = (64, 64),
#                                         batch_size = 32,
#                                         class_mode = 'binary')

# train_generator = train.flow_from_dataframe(dataframe=pd.DataFrame({'filename': validFilenames, 'class': validLabels}),
#                                             directory='train_dataset',
#                                             x_col='filename',
#                                             y_col='class',
#                                             target_size=(200, 200),
#                                             batch_size=32,
#                                             class_mode='binary')

# test = ImageDataGenerator(rescale = 1./255,
#                             shear_range = 0.2,
#                             zoom_range = 0.2,
#                             horizontal_flip = True)
# testDataSet = test.flow_from_directory("test_dataset",
#                                         target_size = (64, 64),
#                                         batch_size = 32,
#                                         class_mode = 'binary')

# model = Sequential([
# Conv2D(32, 3, input_shape=(64, 64, 3)),
# MaxPooling2D(pool_size=2),
# Conv2D(32, 3),
# MaxPooling2D(pool_size=2),
# Flatten(),
# Dense(128, activation='relu'),
# Dense(1, activation='sigmoid')])

# model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(x = trainDataSet, validation_data = testDataSet, epochs = 5)
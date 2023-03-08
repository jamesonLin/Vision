import json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import tensorflow as tf
import cv2


train = ImageDataGenerator(rescale = 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)
trainDataSet = train.flow_from_directory("train_dataset",
                                        target_size = (64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary')

test = ImageDataGenerator(rescale = 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)
testDataSet = test.flow_from_directory("test_dataset",
                                        target_size = (64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary')



f = open('train_dataset/train.json')
data = json.load(f)

for i in data['annotations']:
    filename = i['filename']
    bndbox = i['bndbox']
    xmin = i['bndbox']['xmin']
    ymin = i['bndbox']['ymin']
    xmax = i['bndbox']['xmax']
    ymax = i['bndbox']['ymax']
    
    # print(xmin)
    
f.close()


img = cv2.imread(filename)

# bndboxTensor = tf.convert_to_tensor([ymin, xmin, ymax, xmax])

# model = Sequential([
# Conv2D(1, 1, input_shape=(64, 64, 3)),
# # Conv2D(10, 5),
# MaxPooling2D(pool_size=2),
# Flatten(),
# Dense(10, activation='relu')])

# model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x = trainDataSet, validation_data = testDataSet, epochs = 5)

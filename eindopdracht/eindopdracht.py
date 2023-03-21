import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import where, abs, square, reduce_sum, reduce_mean
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def customLoss(y_true, y_pred):
    # Split the true and predicted labels into classification and bounding box coordinates
    y_true_cls = y_true[:, :num_classes]
    y_pred_cls = y_pred[:, :num_classes]
    y_true_reg = y_true[:, num_classes:]
    y_pred_reg = y_pred[:, num_classes:]

    # Calculate the classification loss using binary crossentropy
    cls_loss = binary_crossentropy(y_true_cls, y_pred_cls)

    # Calculate the bounding box regression loss
    reg_loss = where(
        abs(y_true_reg - y_pred_reg) < 1,
        0.5 * square(y_true_reg - y_pred_reg),
        abs(y_true_reg - y_pred_reg) - 0.5,
    )
    reg_loss = reduce_sum(reg_loss, axis=1)

    # Calculate the total loss as a weighted sum of the classification and regression losses
    total_loss = reduce_mean(cls_loss + 5 * reg_loss)

    return total_loss

def scale(path):
    annotations = pd.read_csv(path + 'annotation.csv')
    for i, annotation in annotations.iterrows():
        height = annotation['height']
        width = annotation['width']
        return height, width
    
# Read the data from csv file
def readData(path):
    target_size = (224, 224)
    annotations = pd.read_csv(path + 'annotation.csv')

    X_img = []
    y_labels = []
    y_bboxes = []

    # Map label strings to integer labels
    label_map = {}
    unique_labels = set(annotations['class'])
    for i, label in enumerate(unique_labels):
        label_map[label] = i

    for i, annotation in annotations.iterrows():
        image_path = annotation['filename']
        labels = annotation['class']
        xmin = annotation['xmin']
        ymin = annotation['ymin']
        xmax = annotation['xmax']
        ymax = annotation['ymax']

        # Preprocess the image
        image = Image.open(path + image_path)
        image = image.resize(target_size)
        image_data = np.array(image, dtype=np.float32)
        image_data /= 255.0

        # Normalize bounding box coordinates
        height, width, channels = image_data.shape
        xmin_norm = xmin / width
        ymin_norm = ymin / height
        xmax_norm = xmax / width
        ymax_norm = ymax / height
        
        X_img.append(image_data)
        y_labels.append(label_map[labels])
        y_bboxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])

    X_img = np.array(X_img)
    y_labels = np.array(y_labels)
    y_bboxes = np.array(y_bboxes)

    return X_img, y_labels, y_bboxes, path

num_classes = 5

trainImg, trainLabels, trainbboxes, trainImgPath = readData('dataset2/train/')
validImg, validLabels, validbboxes, validImgPath = readData('dataset2/valid/')
testImg, testLabels, testbboxes, testImgPath = readData('dataset2/test/')

# Convert labels to one-hot encoding
trainLabels_cls = to_categorical(trainLabels, num_classes)
validLabels_cls = to_categorical(validLabels, num_classes)
testLabels_cls = to_categorical(testLabels, num_classes)

# Add bounding box coordinates to the classification labels
trainLabels_cls_reg = np.concatenate([trainLabels_cls, trainbboxes], axis=1)
validLabels_cls_reg = np.concatenate([validLabels_cls, validbboxes], axis=1)
testLabels_cls_reg = np.concatenate([testLabels_cls, testbboxes], axis=1)


# Define input tensor
inputs = Input(shape=(224, 224, 3))

# Convolutional layers
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# Fully connected layers
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)

# Output layers
bbox_output = Dense(4, name='bbox_output')(x) # output bounding box coordinates
class_output = Dense(num_classes, activation='softmax', name='class_output')(x) # output class label

# Define the model with input and output layers
model = Model(inputs=inputs, outputs=(bbox_output, class_output))

losses = {
	'bbox_output': 'mean_squared_error',
    'class_output': 'categorical_crossentropy'
}

metrics={
    'bbox_output': 'accuracy',
    'class_output': 'accuracy'
}

trainTargets = {
	"class_output": trainLabels_cls,
	"bbox_output": trainbboxes
}
validTargets = {
	"class_output": validLabels_cls,
	"bbox_output": validbboxes
}
testTargets = {
	"class_output": testLabels_cls,
	"bbox_output": testbboxes
}

# Compile the model with custom loss and Adam optimizer
# model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy'])
model.compile(loss=losses, optimizer='adam', metrics=metrics)


# Train the model on the training data
early_stop = EarlyStopping(monitor='val_bbox_output_accuracy', patience=5, verbose=1, mode='max', restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_bbox_output_accuracy', save_best_only=True, mode='max', verbose=1)
model.fit(trainImg, trainTargets, epochs=3, validation_data=(validImg, validTargets), callbacks=[early_stop, checkpoint])

# Evaluate the model on the test set
test_loss, test_bbox_loss, test_class_loss, test_bbox_acc, test_class_acc = model.evaluate(testImg, testTargets, verbose=0)

# Print the test set metrics
print('Test set loss: ', test_loss)
print('Test set bounding box loss: ', test_bbox_loss)
print('Test set class loss: ', test_class_loss)
print('Test set bounding box accuracy: ', test_bbox_acc)
print('Test set class accuracy: ', test_class_acc)


# # Load the new data
# newImg, newLabels, newBboxes, newImgPath = readData('dataset2/new/')

# # Predict bounding boxes on the new data
# preds = model.predict(newImg)

# label_map = {0: 'trafficLight-Red', 1: 'trafficLight', 2: 'car', 3: 'truck', 4: 'trafficLight-Green'}
# y, x = scale(newImgPath)
# size = 224
# y_as = y/size
# x_as = x/size

# print('size: ', y,' ',x)
# print('scale: ', y_as,' ',x_as)

# # Create empty arrays for the labels and boundingbox data
# pred_labels = np.zeros((len(newImg), 5))
# pred_bboxes = np.zeros((len(newImg), 4))
# lbl = np.zeros((len(newImg)))


# # plot the images with labels and bounding box
# for i in range(len(newImg)):
#     fig, ax = plt.subplots()
#     img = newImg[i]
#     pred_labels[i,:] = preds[i,:5]
#     pred_bboxes[i,:] = preds[i,5:]
#     lbl[i] = np.argmax(pred_bboxes[i,:], axis=-1)
#     # Plot the image
#     ax.imshow(img)
#     ax.axis('off')
#     # Get the label name from the label map
#     label_name = label_map[lbl[i]] 
#     # Plot the label name
#     ax.text(0, 0, label_name, fontsize=10, color='red')
#     # Get the bounding box coordinates and convert to pixel values
#     print('  Class', label_name, ':', pred_bboxes[i][0], pred_bboxes[i][1], pred_bboxes[i][2], pred_bboxes[i][3])
#     xmin = int(pred_bboxes[i][0] * x_as)
#     ymin = int(pred_bboxes[i][1] * y_as)
#     xmax = int(pred_bboxes[i][2] * x_as)
#     ymax = int(pred_bboxes[i][3] * y_as)
#     print('  Class', label_name, ':', xmin, ymin, xmax, ymax)
#     # Plot the bounding box
#     rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='r')
#     ax.add_patch(rect)
#     plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import where, abs, square, reduce_sum, reduce_mean
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

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
        # image_data /= 255.0

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

    return X_img, y_labels, y_bboxes

num_classes = 5
num_coords = 4

trainImg, trainLabels, trainbboxes = readData('dataset2/train/')
trainLabels_cat = np.concatenate([trainLabels[:, np.newaxis], trainbboxes], axis=1)
trainLabels_cls = to_categorical(trainLabels_cat[:, 0], num_classes)

validImg, validLabels, validbboxes = readData('dataset2/valid/')
validLabels_cat = np.concatenate([validLabels[:, np.newaxis], validbboxes], axis=1)
validLabels_cls = to_categorical(validLabels_cat[:, 0], num_classes)

testImg, testLabels, testbboxes = readData('dataset2/test/')
testLabels_cat = np.concatenate([testLabels[:, np.newaxis], testbboxes], axis=1)
testLabels_cls = to_categorical(testLabels_cat[:, 0], num_classes)


# Add bounding box coordinates to the classification labels
trainLabels_cls_reg = np.concatenate([trainLabels_cls, trainbboxes], axis=1)
validLabels_cls_reg = np.concatenate([validLabels_cls, validbboxes], axis=1)
testLabels_cls_reg = np.concatenate([testLabels_cls, testbboxes], axis=1)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3)),
    Conv2D(128, (3, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(num_classes + num_coords, activation='linear')
])

# Compile the model with custom loss and Adam optimizer
model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
model.fit(trainImg, trainLabels_cls_reg, epochs=3, validation_data=(validImg, validLabels_cls_reg), callbacks=[early_stop, checkpoint])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(testImg, testLabels_cls_reg, verbose=2)
print(test_loss)
print(test_acc)

# Predict bounding boxes on the test data
testPreds = model.predict(testImg)

# Extract predicted bounding box coordinates
testPreds_cls = testPreds[:, :num_classes]
testPreds_reg = testPreds[:, num_classes:]
print('rest')
print(testPreds_cls)
print(testPreds_cls)

# Convert bounding box coordinates from normalized to pixel coordinates
# testPreds_reg *= 224

# Extract pixel coordinates for each bounding box
testPreds_xmin = testPreds_reg[:, 0::4]
testPreds_ymin = testPreds_reg[:, 1::4]
testPreds_xmax = testPreds_reg[:, 2::4]
testPreds_ymax = testPreds_reg[:, 3::4]

# Convert pixel coordinates to int, this rounds it to only the int
testPreds_xmin = testPreds_xmin.astype(int)
testPreds_ymin = testPreds_ymin.astype(int) 
testPreds_xmax = testPreds_xmax.astype(int)
testPreds_ymax = testPreds_ymax.astype(int)


# for i in range(len(testImg)):
#     print('Image', i)
#     for j in range(num_classes):
#         if testPreds_cls[i, j] > 0.5:
#             xmin = int(testPreds_xmin[i, j])
#             ymin = int(testPreds_ymin[i, j])
#             xmax = int(testPreds_xmax[i, j])
#             ymax = int(testPreds_ymax[i, j])
#             print('  Class', j, ':', xmin, ymin, xmax, ymax)

# Print the predicted bounding boxes
# for i in range(len(testImg)):
#     image = testImg[i]
#     print('Image', i)
#     plt.imshow(image)
#     ax = plt.gca()
#     for j in range(num_classes):
#         if testPreds_cls[i, j] > 0.5:
#             xmin = int(testPreds_xmin[i])
#             ymin = int(testPreds_ymin[i])
#             xmax = int(testPreds_xmax[i])
#             ymax = int(testPreds_ymax[i])
#             print('  Class', j, ':', xmin, ymin, xmax, ymax)
#             print('witdh: ', xmax-xmin, 'height: ', ymax-ymin)
#             rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='r')
#             ax.add_patch(rect)
#     plt.show()
    
    
# Define a list of class names
class_names = ["class1", "class2", "class3", "class4", "class5"]

for i in range(len(testImg)):
    image = testImg[i]
    print('Image', i)
    plt.imshow(image)
    ax = plt.gca()
    for j in range(num_classes):
        if testPreds_cls[i, j] > 0.5:
            # Get the coordinates of the predicted bounding box
            xmin = testPreds_xmin[i]
            ymin = testPreds_ymin[i]
            xmax = testPreds_xmax[i]
            ymax = testPreds_ymax[i]

            # Draw the bounding box
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='r')
            ax.add_patch(rect)

            # Add the label above the bounding box
            label = class_names[j]
            plt.text(xmin, ymin-5, label, color='r', fontsize=12, fontweight='bold')

    plt.show()
            
# # Print the predicted bounding boxes
# for i in range(len(testImg)):
#     print('Image: ', i)
#     xmin = int(testPreds_xmin[i])
#     ymin = int(testPreds_ymin[i])
#     xmax = int(testPreds_xmax[i])
#     ymax = int(testPreds_ymax[i])
#     print('boundingBox: ', xmin, ymin, xmax, ymax)

# # Extract predicted classification labels and bounding boxes
# testPreds_cls = testPreds[:, :num_classes]
# testPreds_reg = testPreds[:, num_classes:]

# # Convert classification labels to integers
# testPreds_cls = np.argmax(testPreds_cls, axis=1)

# # Convert predicted bounding boxes to pixel coordinates
# testPreds_reg[:, 0] *= testImg.shape[2]
# testPreds_reg[:, 1] *= testImg.shape[1]
# testPreds_reg[:, 2] *= testImg.shape[2]
# testPreds_reg[:, 3] *= testImg.shape[1]

# # Plot images with predicted bounding boxes
# for i in range(len(testImg)):
#     image = testImg[i]
#     plt.imshow(image)
#     ax = plt.gca()
#     for j in range(num_classes):
#         if testPreds_cls[i] == j:
#             print("here: " + testPreds_reg.shape)
#             xmin = int(testPreds_reg[i, j*4])
#             ymin = int(testPreds_reg[i, j*4+1])
#             xmax = int(testPreds_reg[i, j*4+2])
#             ymax = int(testPreds_reg[i, j*4+3])
#             rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='r')
#             ax.add_patch(rect)
#     plt.show()





# from tensorflow.keras.callbacks import ModelCheckpoint

# # Define the checkpoint filepath and set up the callback
# checkpoint_filepath = 'model_checkpoint.h5'
# checkpoint_callback = ModelCheckpoint(
#     checkpoint_filepath,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True,
#     save_weights_only=False,
#     verbose=1
# )

# # Train the model on the training data with the checkpoint callback
# model.fit(trainImg, trainLabels_cls_reg, epochs=50, 
#           validation_data=(testImg, testLabels_cls_reg),
#           callbacks=[checkpoint_callback])

# # Load the best model
# model = tf.keras.models.load_model(checkpoint_filepath)

# # Evaluate the best model on the test data
# test_loss, test_acc = model.evaluate(testImg, testLabels_cls_reg, verbose=2)
# print(test_loss)
# print(test_acc)

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

NUM_CLASSES = 3
TRAINPATH = 'dataset3/train/'
VALIDPATH = 'dataset3/valid/'
TESTPATH = 'dataset3/test/'
NEWPATH = 'dataset3/new/'


def unnormalize_bbox(bbox, origSize, resSize):
    scale_x = origSize[0] / resSize
    scale_y = origSize[1] / resSize

    xNorm, yNorm, wNorm, hNorm = bbox
    xmin = xNorm * (origSize[0] / scale_x)
    ymin = yNorm * (origSize[1] / scale_y)
    xmax = wNorm * (origSize[0] / scale_x)
    ymax = hNorm * (origSize[1] / scale_y)

    return (xmin, ymin, xmax, ymax)

def scaleBbox(bbox, origSize, resSize):
    # Calculate scaling factor
    scaleX = resSize[0] / origSize[0]
    scaleY = resSize[1] / origSize[1]

    xmin, ymin, xmax, ymax = bbox
    xminScaled = xmin * scaleX
    yminScaled = ymin * scaleY
    xmaxScaled = xmax * scaleX
    ymaxScaled = ymax * scaleY

    return (xminScaled, yminScaled, xmaxScaled, ymaxScaled)

def CraeteLabelMap(path):
    annotations = pd.read_csv(path + 'annotation.csv')
    # Map label strings to integer labels
    labelMap = {}
    unique_labels = set(annotations['class'])
    for i, label in enumerate(unique_labels):
        labelMap[label] = i
    return labelMap
    
def readData(path, labelMap):
    target_size = (224, 224)
    annotations = pd.read_csv(path + 'annotation.csv')

    X_img = []
    y_labels = []
    y_bboxes = []

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
        y_labels.append(labelMap[labels])
        y_bboxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])

    X_img = np.array(X_img)
    y_labels = np.array(y_labels)
    y_bboxes = np.array(y_bboxes)
    
    # Convert labels to one-hot encoding
    Labels_cls = to_categorical(y_labels, NUM_CLASSES)

    return X_img, Labels_cls, y_bboxes

labelMap = CraeteLabelMap(TRAINPATH)

trainImg, trainLabels, trainbboxes = readData(TRAINPATH, labelMap)
validImg, validLabels, validbboxes = readData(VALIDPATH, labelMap)
testImg, testLabels, testbboxes = readData(TESTPATH, labelMap)


# Define input tensor
inputs = Input(shape=(224, 224, 3))

# # Convolutional layers
x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Define the classification and bounding box prediction heads
classification = Flatten()(x)
classification = Dense(256, activation="relu")(classification)
classification = Dropout(0.5)(classification)
class_output = Dense(NUM_CLASSES, activation="softmax", name="class_output")(classification)

bbox = Flatten()(x)
bbox = Dense(256, activation="relu")(bbox)
bbox = Dropout(0.5)(bbox)
bbox_output = Dense(4, activation="linear", name="bbox_output")(bbox)


# Define the model with input and output layers
model = Model(inputs=inputs, outputs=(class_output, bbox_output))

losses = {
    'class_output': 'categorical_crossentropy',
	'bbox_output': 'mean_squared_error'
}

metrics={
    'class_output': 'accuracy',
    'bbox_output': 'accuracy'
}

trainTargets = {
	"class_output": trainLabels,
	"bbox_output": trainbboxes
}
validTargets = {
	"class_output": validLabels,
	"bbox_output": validbboxes
}
testTargets = {
	"class_output": testLabels,
	"bbox_output": testbboxes
}


# Compile the model with categorical_crossentropy, mean_squared_error and Adam optimizer
model.compile(loss=losses, optimizer='adam', metrics=metrics)


# Train the model on the training data and validation on validation data
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_bbox_output_accuracy', save_best_only=True, mode='max', verbose=1)
model.fit(trainImg, trainTargets, epochs=150, batch_size=50, validation_data=(validImg, validTargets), callbacks=[checkpoint])

# Evaluate the model on the test set
test_loss, test_bbox_loss, test_class_loss, test_bbox_acc, test_class_acc = model.evaluate(testImg, testTargets, verbose=0)

# Print the test set metrics
print('Test set loss: ', test_loss)
print('Test set bounding box loss: ', test_bbox_loss)
print('Test set class loss: ', test_class_loss)
print('Test set bounding box accuracy: ', test_bbox_acc)
print('Test set class accuracy: ', test_class_acc)


# Load the new data
newImg, newLabels, newBboxes = readData(NEWPATH, labelMap)

# Predict bounding boxes on the new data
pred_labels, pred_bboxes = model.predict(newImg)

# np.set_printoptions(threshold=sys.maxsize)
# print(pred_labels)

orig_size = (1920, 1200)
res_size = 224
bboxes = [unnormalize_bbox(bbox, orig_size, res_size) for bbox in pred_bboxes]
img_size = (224, 224)
bboxScaled = [scaleBbox(bbox, orig_size, img_size) for bbox in bboxes]


labelMap = {v: k for k, v in labelMap.items()}
# plot the images with labels and bounding box
for i in range(len(newImg)):
    fig, ax = plt.subplots()
    img = newImg[i]
    lbl = np.argmax(pred_labels[i], axis=-1)
    ax.imshow(img)
    ax.axis('off')
    label_name = labelMap[lbl] 
    xmin, ymin, xmax, ymax = bboxScaled[i]
    ax.text(xmin, ymin, label_name, fontsize=10, color='red')
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='r')
    ax.add_patch(rect)
    plt.show()
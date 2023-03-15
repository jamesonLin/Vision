import os
import pandas as pd
import shutil
import random

# set up paths
image_dir = 'data/'
train_dir = 'dataset/train/'
test_dir = 'dataset/test/'
valid_dir = 'dataset/valid/'
csv_path = 'dataset/_annotations.csv'

# set up class whitelist
classes = {'trafficLight-Red', 'trafficLight-Green', 'trafficLight', 'car', 'truck'}

# create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# read CSV file into dataframe and filter by whitelist
df = pd.read_csv(csv_path, header=0, names=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
df = df[df['class'].isin(classes)]

# randomly select unique filenames
filenames = set(df['filename'])
selected_filenames = set(random.sample(filenames, min(len(filenames), 6000))) # size for the randomly select

# calculate number of images for each split
num_train = int(len(selected_filenames) * 0.6)
num_test = int(len(selected_filenames) * 0.3)
num_valid = len(selected_filenames) - num_train - num_test

# process each selected filename
train_csv_path = os.path.join(train_dir, 'annotation.csv')
test_csv_path = os.path.join(test_dir, 'annotation.csv')
valid_csv_path = os.path.join(valid_dir, 'annotation.csv')
train_filenames = set()
test_filenames = set()
valid_filenames = set()
with open(train_csv_path, 'w') as train_csv, \
        open(test_csv_path, 'w') as test_csv, \
        open(valid_csv_path, 'w') as valid_csv:
    train_csv.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
    test_csv.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
    valid_csv.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
    for i, filename in enumerate(selected_filenames):
        # determine which directory to copy image to
        if i < num_train:
            dest_dir = train_dir
            dest_csv = train_csv
            dest_filenames = train_filenames
        elif i < num_train + num_test:
            dest_dir = test_dir
            dest_csv = test_csv
            dest_filenames = test_filenames
        else:
            dest_dir = valid_dir
            dest_csv = valid_csv
            dest_filenames = valid_filenames

        # check if file exists and copy it to destination directory
        src_path = os.path.join(image_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.isfile(src_path) and filename not in dest_filenames:
            shutil.copy2(src_path, dest_path)
            dest_filenames.add(filename)

            # write corresponding annotations to destination CSV file
            rows = df[df['filename'] == filename].values.tolist()
            for row in rows:
                dest_csv.write(','.join(str(x) for x in row) + '\n')
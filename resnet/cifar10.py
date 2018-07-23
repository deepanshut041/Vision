import numpy as np
import matplotlib.pyplot as plt
import pickle
import download
import os
from keras.utils.np_utils import to_categorical

# This downloads cifar10 dataset if required 
def get_dataset():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = "dataset/CIFAR-10/"
    download.download(url=url, path=path, kind='tar.gz', progressbar=True, replace=False, verbose=True)



# Load data set from pickel file refrence toronto website
def unpickle(filename):
    path = "dataset/CIFAR-10/"
    file_path = os.path.join(path, "cifar-10-batches-py/", filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:

        data = pickle.load(file, encoding='bytes')

    return data

def convert_images(raw):
    img_size = 32
    num_channels = 3
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def load_data(filename):
    data = unpickle(filename)
    raw_images = data[b'data']
    _class = np.array(data[b'labels'])
    images = convert_images(raw_images)

    return images, _class

def load_training_data():
    t_images = 50000
    img_size = 32
    num_channels = 3
    num_classes = 10
    images = np.zeros(shape=[t_images, img_size, img_size, num_channels], dtype=float)
    _class = np.zeros(shape=[t_images], dtype=int)
    begin = 0

    # For each data-file total 5 batches are present.
    for i in range(5):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = load_data(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        _class[begin:end] = cls_batch
        begin = end

    return images, _class, to_categorical(_class, num_classes=num_classes)

def load_test_data():
    num_classes = 10
    images, _class = load_data(filename="test_batch")
    return images, _class, to_categorical(_class, num_classes=num_classes)

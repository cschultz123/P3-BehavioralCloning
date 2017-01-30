"""
This module contains all of the methods used to build and train the
neural network for the behavior cloning project.
"""

import cv2
import keras
from keras.models import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.misc as spm

data_source = 'driving_log.csv'

def read_input():
    """
    Read the driving_log.csv file into pandas data frame.

    :return: (DataFrame)
    """
    data = pd.read_csv(data_source)

    # remove spaces in files names
    data['center'] = data['center'].str.strip()
    data['right'] = data['right'].str.strip()
    data['left'] = data['left'].str.strip()

    return data

##############################################################################
# Image Augmentation Methods
##############################################################################

# borrowed from Vivek Yadav (https://github.com/vxy10/ImageAugmentation)
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def resize_image(x, size):
    """Down-sample image to size."""
    return spm.imresize(x, size=size)

def crop_out_car(x, y_range):
    rows, columns, channels = x.shape
    return x[y_range:rows-y_range,:,:]

def flip_axis(x, axis):
    """Flip image about specified axis."""
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    steering_adjustment = -1
    return x, steering_adjustment

# translate image horizontally.
def translate_image(image, x_range, y_range):
    dx = x_range*np.random.uniform()-x_range/2.
    dy = y_range*np.random.uniform()-y_range/2.
    t_matrix = np.array([[1,0,dx],[0,1,dy]], dtype=np.float64)
    # adjust steering angle by 0.2 radians per pixel
    steering_adjustment = dx/x_range*2*.2
    return cv2.warpAffine(image,t_matrix,(image.shape[1],image.shape[0])), steering_adjustment

##############################################################################
# Image Pre-processing and data generation methods
##############################################################################
def pre_process_image(image):
    """
    Pre-process image. This just involves cropping and resizing the image.

    :param image: rgb image
    :return:
    """
    # crop the car out of the image.
    img = crop_out_car(image, 25)

    # resize image
    return resize_image(img, size=(64, 64, 3))

def pre_process(log_entry, train=False):
    """
    Pre-process procedure.

    Training Routine:

    1. Randomly choose perspective (left, center, right)
    2. Read corresponding image and steering angle.
    3. Apply steering angle offset if left or right perspective.
    4. Augment brightness of image.
    5. Translate image in x and y.
    6. Flip image about y axis.
    7. Crop out car.
    8. Resize image.

    Non-Training Routine:

    1. Read center image and steering angle.
    2. Crop out car.
    3. Resize image.

    Args:
        log_entry (DataFrame): driving_log.csv row
        train (bool): turn on data augmentation

    Returns:
        (image, steering angle)
    """
    perspective = 'center'

    if train:
        # randomly choose which camera to use.
        perspective = ['center', 'right', 'left'][np.random.randint(3)]

    # read chosen image and steering angle.
    img, steer_angle = \
        mpimg.imread(log_entry[perspective]), log_entry['steering']

    # adjust steering angle to account for viepoint shift.
    if perspective == 'right':
        steer_angle -= 0.25
    elif perspective == 'left':
        steer_angle += 0.25

    if train:

        # augment the brightness of the image
        img = augment_brightness_camera_images(img)

        # translate the image in x and y
        img, steer_offset = translate_image(img, 20, 10)
        steer_angle += steer_offset

        # randomly flip image
        if np.random.randint(2):
            img, steer_sign = flip_axis(img, 1)
            steer_angle *= steer_sign

    # pre-process image
    img = pre_process_image(img)

    return img, steer_angle


def data_and_label_generator(batch_size=128, train=False):
    """
    This method generates a batch of pre-processed images with corresponding
    labels.

    :param batch_size:  (int) number of images and labels per iteration
    :param train: (bool) turns on image augmentation pipeline
    :return: array(batch_size, 64, 64, 3) batch of images, array(batch_size) batch of labels
    """
    data = read_input()

    while True:
        # containers for images and labels
        images = list()
        labels = list()

        # randomly select entries from drive_log.csv
        rnd_idx = np.random.randint(0, len(data), size=(batch_size,))
        random_selection = data.iloc[rnd_idx]

        for i in range(batch_size):
            image, angle = pre_process(random_selection.iloc[i], train)
            images.append(image)
            labels.append(angle)

        yield np.array(images, dtype=np.float32), np.array(labels, np.float32)

##############################################################################
# Model Creation and Training
##############################################################################

def create_model():
    """
    Creates keras model that is ready to be trained.

    :return: (Sequential) compiled keras model
    """
    # input image dimensions
    image_height = 64
    image_width = 64

    # NVIDIA network modified to use ELU and a 1x1 convolution layer.
    model = Sequential()
    model.add(layers.Lambda(lambda x: x / 255. - 0.5, input_shape=(image_height, image_width, 3)))
    model.add(layers.Conv2D(3, 1, 1, border_mode='valid', name='conv0'))
    model.add(layers.Conv2D(24, 3, 3, border_mode='valid', name='conv1'))
    model.add(layers.ELU())
    model.add(layers.Conv2D(36, 3, 3, border_mode='valid', name='conv2'))
    model.add(layers.ELU())
    model.add(layers.Conv2D(48, 3, 3, border_mode='valid', name='conv3'))
    model.add(layers.ELU())
    model.add(layers.Conv2D(64, 3, 3, border_mode='valid', name='conv4'))
    model.add(layers.ELU())
    model.add(layers.Conv2D(64, 3, 3, border_mode='valid', name='conv5'))
    model.add(layers.ELU())
    model.add(layers.Flatten())
    model.add(layers.Dense(100))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))
    model.add(layers.ELU())
    model.add(layers.Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def train(model):
    """
    Trains the model using driving_log.csv data set. This also saves the
    trained model architecture to "model.json" and weights to "model.h5".

    :param model: (Sequential) compiled keras model
    """
    t_gen = data_and_label_generator(batch_size=32, train=True)
    v_gen = data_and_label_generator(batch_size=32, train=False)

    model.fit_generator(generator=t_gen,
                        samples_per_epoch=40000,
                        nb_epoch=5,
                        validation_data=v_gen,
                        nb_val_samples=2000)

    model.save_weights('model.h5')
    with open("model.json", 'w') as f:
        f.write(model.to_json())
    print("Model Saved!")

if __name__ == "__main__":
    model = create_model()
    train(model)
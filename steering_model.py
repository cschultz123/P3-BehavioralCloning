import argparse
import cv2
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, ELU, Flatten, MaxPooling2D
import numpy as np
import os
import pandas as pd
import re


# <model number>_<previous model number>_<data src>_<epochs>_<steps>
MODEL_NAME_REG = re.compile("(?P<current>\d*)_(?P<prior>\d*)__(?P<data_src>.*)__(?P<epochs>\d*)_(?P<steps>\d*)\.h5")


##############################################################################
# Convenience Methods
##############################################################################


def load_driving_log(fp="original.csv"):
    """Load original.csv into Pandas Dataframe."""
    ds = pd.read_csv(fp, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    ds['left'] = ds['left'].str.strip()
    ds['center'] = ds['center'].str.strip()
    ds['right'] = ds['right'].str.strip()
    return ds


def imread(fp):
    """Read RGB image."""
    if not os.path.exists(fp):
        raise FileNotFoundError("{} not found!".format(fp))
    return cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)


def random_entry(random=False):
    """Randomly return left, center, and right image from entry in original.csv"""
    ds = load_driving_log()

    index = 0
    if random:
        index = np.random.randint(0, len(ds.index)-1)

    return imread(ds.left[index]), imread(ds.center[index]), imread(ds.right[index])


##############################################################################
# Image Augmentation Methods
##############################################################################


def crop(img, height=0):
    """Crop image vertically from bottom."""
    return img[:-height, :, :]


def shift(img, dx=0, dy=0):
    """Shift image horizontally and vertically."""
    rows, cols, channels = img.shape
    t_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, t_matrix, (cols, rows))


def horizontal_flip(img):
    """Flip image about y-axis."""
    return cv2.flip(img, 1)


def brightness(img, scale):
    """Increase brightness by percentage."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * scale
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


##############################################################################
# Image Pre-processing and data generation methods
##############################################################################


def randomize(x):
    return x * np.random.uniform() - x/2


def random_brightness(img):
    """Randomly apply brightness augmentation between 1 and 0.25."""
    scale = 0.75 * np.random.rand(1) + 0.25
    return brightness(img, scale)


def augment(img, angle, xr=20, yr=10, flip=False, light=False):
    """Augmentation for training images."""
    dx = randomize(xr)
    dy = randomize(yr)
    img = shift(img, dx, dy)
    angle += dx/xr * 2 * 0.2

    if flip:
        if np.random.choice([0, 1]):
            img = horizontal_flip(img)
            angle *= -1

    if light:
        img = random_brightness(img)

    return img, angle


def preprocess(img, normalize=True):
    """Prepare image to be input to model."""
    # crop out car
    img = crop(img, 25)

    # resize image
    img = cv2.resize(img, (64, 64))

    if normalize:
        # normalize mean and standard deviation
        img = np.float32(img)
        img -= np.mean(img, axis=2, keepdims=True)
        img /= (np.std(img, axis=2, keepdims=True) + 1e-7)

    return img


##############################################################################
# Training Generator
##############################################################################


def image_generator(fp, batch_size=32, train=False):

    # load original.csv as Pandas Dataframe
    ds = load_driving_log(fp)

    # 66% of data will be turn based driving
    high_selection, = np.where(ds.steering > 0.05)

    # 33% of data will be straight driving
    low_selection, = np.where(ds.steering <= 0.05)
    low_index = np.random.randint(0, len(low_selection), size=(int(len(high_selection) * 0.75),))
    low_selection = low_selection[low_index]

    selection = np.concatenate((high_selection, low_selection))

    ds = ds.iloc[selection]

    while True:
        # containers for images
        images = list()
        labels = list()

        # randomly select entries from driving log
        rnd_idx = np.random.randint(0, len(ds.index), size=(batch_size,))
        entries = ds.iloc[rnd_idx]

        for i in range(batch_size):

            # get current entry
            entry = entries.iloc[i]

            # select camera perspective
            camera = "center"
            if train:
                camera = np.random.choice(['left', 'center', 'right'])

            # read image and steering angle
            img, angle = imread(entry[camera]), entry['steering']

            # adjust angle for left and right perspectives
            if camera == 'right':
                angle -= 0.25
            elif camera == 'left':
                angle += 0.25

            if train:
                img, angle = augment(img, angle)

            img = preprocess(img)

            images.append(img)
            labels.append(angle)

        yield np.array(images, dtype=np.float32), np.array(labels, np.float32)


##############################################################################
# Model Creation and Training
##############################################################################


def create_model():
    """
    Keras model of NVIDIA network modified to use ELU and a 1x1 convolution layer.
    """
    model = Sequential()
    model.add(Conv2D(3,  (1, 1), padding='valid', name='conv0', input_shape=(64, 64, 3)))
    model.add(Conv2D(32, (3, 3), padding='valid', name='conv1'))
    model.add(ELU())
    model.add(Conv2D(32, (3, 3), padding='valid', name='conv2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='valid', name='conv3'))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid', name='conv4'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='valid', name='conv5'))
    model.add(ELU())
    model.add(Conv2D(128, (3, 3), padding='valid', name='conv6'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(ELU())
    model.add(Dense(1))

    return model


def train(fp, model_file, epochs=5, steps_per_epoch=400, validation_steps=100):
    """
    Trains the model using original.csv data set. This also saves the
    trained model to "model_name.h5".
    """
    if os.path.exists(model_file):
        print("Training with existing model, {}".format(model_file))
        model = load_model(model_file)
    else:
        print("Training with new model.")
        model = create_model()

    model.compile(loss='mse', optimizer='adam')

    training_generator = image_generator(fp, batch_size=32, train=True)
    validation_generator = image_generator(fp, batch_size=32)

    model.fit_generator(generator=training_generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps)

    # generate new model file name
    new_model_file = _generate_model_name(model_file, fp, epochs, steps_per_epoch)
    model.save(new_model_file)
    print("Model Saved! {}".format(new_model_file))


def _generate_model_name(model_file, fp, epochs, steps_per_epoch):
    """
    Generate model file names that make it easy to keep track of current
    state of model. The format is the following:
    
        <current version>_<previous version>_<data_src>_<epochs>_<steps>
        
    The data_src parameter is the csv filename used to train the model. The 
    current and previous version are used to identify which model was used
    prior to training.
    """
    match = MODEL_NAME_REG.match(model_file)
    current, previous = int(match.group("current")) + 1, match.group("current")
    data_src = os.path.splitext(os.path.basename(fp))[0]
    return "{}_{}__{}__{}_{}.h5".format(current, previous, data_src, epochs, steps_per_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='?', type=str, help="Path to original.csv file.")
    parser.add_argument('model', nargs='?', type=str, help="Model file name.")
    parser.add_argument('epochs', nargs='?', type=int, help="Number of epochs.")
    parser.add_argument('steps_per_epoch', nargs='?', type=int, help="Steps per epoch.")
    parser.add_argument('--new', action='store_true')
    args = parser.parse_args()

    # make reproducible
    np.random.seed(1)

    model = args.model
    if args.new:
        model = '0_0__none__0_0.h5'
    elif MODEL_NAME_REG.match(model) is None:
        raise NameError("Model name does not conform to standard.")

    train(fp=args.data, model_file=model, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)

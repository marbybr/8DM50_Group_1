import random
import gryds
import numpy as np
from unet_model import unet
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from unet_utils import extract_patches



def random_brightness(image,label):
    '''function that takes in a random image, and returns the image with a random brightness offset.
    NOTE: The brightness can not override a maximum value of 1. Any addition that would overflow is therefore
    capped to this maximum value.'''
    brightness_addition = random.random()
    return np.clip(image + brightness_addition, 0, 1), label

def bright_augmented_datagenerator(images, segmentations, patch_size, patches_per_im, batch_size):
    """
    Datagenerator function from unet_utils, but including the brightness augmentation
    """
    # Total number of patches generated per epoch
    total_patches = len(images) * patches_per_im
    # Amount of batches in one epoch
    nr_batches = int(np.ceil(total_patches / batch_size))

    while True:
        # Each epoch extract different patches from the training images
        x, y = extract_patches(images, segmentations, patch_size, patches_per_im, seed=np.random.randint(0, 500))

        # Feed data in batches to the network
        for idx in range(nr_batches):
            x_batch = x[idx * batch_size:(idx + 1) * batch_size]
            y_batch = y[idx * batch_size:(idx + 1) * batch_size]

            # Apply data augmentation to each patch
            augmented_x_batch = []
            augmented_y_batch = []

            for img, label in zip(x_batch, y_batch):
                augmented_image, augmented_label = random_brightness(img,label)
                augmented_x_batch.append(augmented_image)
                augmented_y_batch.append(augmented_label)

            # Convert to numpy arrays
            augmented_x_batch = np.array(augmented_x_batch)
            augmented_y_batch = np.array(augmented_y_batch)

            yield augmented_x_batch, augmented_y_batch

def bspline_geometric_augmentation(image):
    ''''Function that takes in an image and return the same image with a bspline geometric augmentation according
    to the example found on https://github.com/tueimage/gryds'''
    channels = image.shape[-1]  # Get the number of channels
    grid = image.shape[:2]  # Shape of the image (height, width)

    # Define random B-spline coefficients
    bspline_coefficients = np.random.randn(2, 3, 3)
    bspline_coefficients -= 0.5
    bspline_coefficients /= 20

    # Create the B-spline transformation
    bspline_transform = gryds.BSplineTransformation(bspline_coefficients)

    # Apply the transformation to each channel separately
    augmented_image = np.zeros_like(image)
    for channel in range(channels):
        image_interpolator = gryds.Interpolator(image[..., channel], order=1)
        augmented_image[..., channel] = image_interpolator.transform(bspline_transform)

    return augmented_image

def fully_augmented_datagenerator(images, segmentations, patch_size, patches_per_im, batch_size):
    """
    Datagenerator function from unet_utils that contains both the brightness augmentation AND the bspline
    geometric augmentation.
    """
    # Total number of patches generated per epoch
    total_patches = len(images) * patches_per_im
    # Amount of batches in one epoch
    nr_batches = int(np.ceil(total_patches / batch_size))

    while True:
        # Each epoch extract different patches from the training images
        x, y = extract_patches(images, segmentations, patch_size, patches_per_im, seed=np.random.randint(0, 500))

        # Feed data in batches to the network
        for idx in range(nr_batches):
            x_batch = x[idx * batch_size:(idx + 1) * batch_size]
            y_batch = y[idx * batch_size:(idx + 1) * batch_size]

            # Apply data augmentation to each patch
            augmented_x_batch = []
            augmented_y_batch = []

            for img, label in zip(x_batch, y_batch):
                augmented_image, augmented_label = random_brightness(img, label)
                sec_augmented_image = bspline_geometric_augmentation(augmented_image)
                sec_augmented_label = bspline_geometric_augmentation(augmented_label)
                augmented_x_batch.append(sec_augmented_image)
                augmented_y_batch.append(sec_augmented_label)

            # Convert to numpy arrays
            augmented_x_batch = np.array(augmented_x_batch)
            augmented_y_batch = np.array(augmented_y_batch)

            yield augmented_x_batch, augmented_y_batch
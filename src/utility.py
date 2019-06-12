import cv2
import os
import numpy as np
import tensorflow as tf
import keras.backend as backend
import matplotlib.pylab as plt
from random import randint

def normalize_image(image):
    return (image - image.min()) / np.ptp(image)

def read_image(folder, filename, size):
    image = plt.imread(os.path.join(folder, filename))

    if (not type(image) == np.ndarray):
        print("Failed to read image: '{}'".format(os.path.join(folder, filename)))

    if not (image.shape[0] == size and image.shape[1] == size):
        image = cv2.resize(image, (128, 128))
        print("Resizing image!")

    if randint(0, 1) == 1:
        return image / 127.5 - 1
    else:
        return np.fliplr(image / 127.5 - 1)

def red_image(imsize):
    image = np.zeros((imsize, imsize, 3))
    image[:] = (1.0, 0, 0)
    return image

def green_image(imsize):
    image = np.zeros((imsize, imsize, 3))
    image[:] = (0, 1.0, 0)
    return image

def label_image(labels, imsize):
    image = red_image(imsize) if labels[0] == 0 else green_image(imsize)

    for idx in range(1, len(labels)):
        bolt = red_image(imsize) if labels[idx] == 0 else green_image(imsize)
        image = np.concatenate((image, bolt), axis = 1)

    return image


def generate_batch_labels(labels, imsize):
    first = generate_label_layers(labels[0], imsize)
    batchlabels = np.reshape(first, (1, *first.shape))
    for idx in range(1, len(labels)):
        batchlabels = np.append(batchlabels, np.reshape(generate_label_layers(labels[idx], imsize), (1, *first.shape)), axis = 0)
    return batchlabels

def generate_label_layers(labels, imsize):
    layers = np.tile(np.array([[labels[0]]]), (imsize, imsize, 1))
    
    for idx in range(1, len(labels)):
        newlayer = np.tile(np.array([[labels[idx]]]), (imsize, imsize, 1))
        layers = np.append(layers, newlayer, axis = 2)

    return layers

def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
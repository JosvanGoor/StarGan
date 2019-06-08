import cv2
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as backend
from tensorflow.keras.layers import Layer
import matplotlib.pylab as plt

def normalize_image(image):
    return (image - image.min()) / np.ptp(image)

def read_image(folder, filename, size):
    image = plt.imread(os.path.join(folder, filename))

    if (not type(image) == np.ndarray):
        print("Failed to read image: '{}'".format(os.path.join(folder, filename)))

    if not (image.shape[0] == size and image.shape[1] == size):
        image = cv2.resize(image, (128, 128))
        print("Resizing image!")

    return image / 127.5 - 1
    # return image / 255.0

def label_image(labels, imsize):
    image = np.tile([[labels[0]]], (imsize, imsize))

    for idx in range(1, len(labels)):
        bolt = np.tile([[labels[idx]]], (imsize, imsize))
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

class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = backend.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        grad_norm = tf.norm(tf.layers.flatten(grad))
        return grad_norm

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)
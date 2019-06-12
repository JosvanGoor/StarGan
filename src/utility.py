import cv2
import keras.backend as backend
import matplotlib.pylab as plt
import numpy as np
import os
import tensorflow as tf

from    keras.engine.topology import Layer
from    random import randint

'''
    Gradient normalisation layer
'''
class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = tf.keras.backend.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        grad_norm = tf.norm(tf.layers.flatten(grad))
        return grad_norm

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

'''
    Returns image with values normalised between 0 and 1.
'''
def normalize_image(image):
    return (image - image.min()) / np.ptp(image)

'''
    Reads image and returns it with values between -1 and 1.
    Has a ~50% chance to flip the image over the horizontal axis (left-to-right)
    Automagically rescales the image to the expected image size.
'''
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

'''
    Generates and returns image of imsize x imsize filled with a certain color
'''
def color_image(imsize, color = (1.0, 1.0, 1.0)):
    image = np.zeros((imsize, imsize, 3))
    image[:] = color
    return image

'''
    Returns tensorboard label image
'''
def label_image(labels, imsize):
    image = color_image(imsize, (1.0, 0, 0)) if labels[0] == 0 else color_image(imsize, color(0.0, 1.0, 0.0))

    for idx in range(1, len(labels)):
        bolt = color_image(imsize, (1.0, 0, 0)) if labels[idx] == 0 else color_image(imsize, color(0.0, 1.0, 0.0))
        image = np.concatenate((image, bolt), axis = 1)

    return image

'''
    Generates label-layers for entire batch
'''
def generate_batch_label_layers(labels, imsize):
    first = generate_label_layers(labels[0], imsize)
    batchlabels = np.reshape(first, (1, *first.shape))
    for idx in range(1, len(labels)):
        batchlabels = np.append(batchlabels, np.reshape(generate_label_layers(labels[idx], imsize), (1, *first.shape)), axis = 0)
    return batchlabels

'''
    Generates label-layers for single input
'''
def generate_label_layers(labels, imsize):
    layers = np.tile(np.array([[labels[0]]]), (imsize, imsize, 1))
    
    for idx in range(1, len(labels)):
        newlayer = np.tile(np.array([[labels[idx]]]), (imsize, imsize, 1))
        layers = np.append(layers, newlayer, axis = 2)

    return layers

'''
    Writes log entry for tensorboard
'''
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
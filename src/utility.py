import cv2
import os
import numpy as np

def read_image(folder, filename, size):
    image = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)

    if not (image.shape[0] == size and image.shape[1] == size):
        image = cv2.resize(image, (128, 128))
        print("Resizing image!")

    # return image / 127.5 - 1
    return image / 255.0

def generate_label_layers(labels, imsize):
    layers = np.tile(np.array([[labels[0]]]), (imsize, imsize, 1))
    
    for idx in range(1, len(labels)):
        newlayer = np.tile(np.array([[labels[idx]]]), (imsize, imsize, 1))
        layers = np.append(layers, newlayer, axis = 2)

    return layers


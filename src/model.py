import numpy as np
from keras_contrib.layers import InstanceNormalization
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Add, Activation
import keras

def conv_layer(input, filters, kernelsize = 4, stride = 2, padding = "same", activation = "relu", normalization = True):
    layer = Conv2D(filters, kernel_size = kernelsize, strides = stride, padding = padding)(input)
    
    if normalization:
        layer = InstanceNormalization()(layer)
    
    if activation == "relu":
        return Activation("relu")(layer)
    elif activation == "leakyrelu":
        return LeakyReLU(0.01)(layer)
    elif activation == "none":
        return layer
    else:
        raise Exception("Unknown activation function: '{}'.".format(activation))

def residual_layer(input, filters = 256, kernelsize = 3, stride = 1, padding = "same"):
    layer = conv_layer(input, filters, kernelsize, stride, padding)
    return Add()([layer, input])

def deconv_layer(input, filters, kernelsize = 4, stride = 2, padding = "same"):
    layer = Conv2DTranspose(filters, kernel_size = kernelsize, strides = stride, padding = padding)(input)
    layer = InstanceNormalization()(layer)
    return Activation("relu")(layer)

def genout_layer(input):
    layer = Conv2D(3, kernel_size = 7, strides = 1, padding = "same")(input)
    return Activation('tanh', name = "generator_output")(layer)

def create_generator(imgsize, labelcount, bn_repeat = 6):
    input_layer = Input((imgsize, imgsize, labelcount + 3,), name = "generator_input")
    
    # Down sampling part
    layers = conv_layer(input_layer, 64, 7, 1)
    layers = conv_layer(layers, 128, 4, 2)
    layers = conv_layer(layers, 256, 4, 2)

    # Bottleneck
    for _ in range(bn_repeat):
        layers = residual_layer(layers, 256, 3, 1)

    # Up sampling layer
    layers = deconv_layer(layers, 128, 4, 2)
    layers = deconv_layer(layers, 64, 4, 2)
    output_layer = genout_layer(layers)

    return keras.models.Model(input_layer, output_layer, name = "generator")

def create_discriminator(imsize, labelcount, hl_repeat = 5):
    input_layer = Input((imsize, imsize, 3,), name = "discriminator_input")

    layers = conv_layer(input_layer, 64, 4, 2, "same", "leakyrelu", False)

    numfilters = 128
    for _ in range(hl_repeat):
        layers = conv_layer(layers, numfilters, 4, 2, "same", "leakyrelu", False)
        numfilters *= 2

    output_layer_isreal = Conv2D(1, kernel_size = 3, strides = 1, padding = "same", name = "output_isreal")(layers)
    output_layer_labels = Conv2D(labelcount, kernel_size = imsize // 64, strides = 1, padding = "valid", name = "output_labels")(layers)

    return keras.models.Model(input_layer, [output_layer_isreal, output_layer_labels], name = "discriminator")
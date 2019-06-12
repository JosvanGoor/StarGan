from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Add, Activation
from keras_contrib.layers import InstanceNormalization

def generator_conv2d_block(input, filters, kernel_size = 4, stride = 2, padding = "same"):
    layer = Conv2D \
    (
        filters,
        kernel_size = kernel_size,
        strides = stride,
        padding = padding
    )(input)

    layer = InstanceNormalization()(layer)
    return Activation("relu")(layer)

def discriminator_conv2d_block(input, filters, kernel_size = 4, stride = 2, padding = "same", leaky = 0.01):
    layer = Conv2D \
    (
        filters,
        kernel_size = kernel_size,
        strides = stride,
        padding = padding
    )(input)

    return LeakyReLU(leaky)(layer)

def residual_block(input, filters = 256, kernel_size = 3, stride = 1, padding = "same"):
    layer = generator_conv2d_block(input, filters, kernel_size, stride, padding = padding)
    return Add()([layer, input])

def deconv2d_block(input, filters, kernel_size = 4, stride = 2, padding = "same"):
    layer = Conv2DTranspose \
    (
        filters,
        kernel_size = kernel_size,
        strides = stride,
        padding = padding
    )(input)

    layer = InstanceNormalization()(layer)
    return Activation("relu")(layer)
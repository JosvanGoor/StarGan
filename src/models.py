from keras.layers import    Activation, \
                            Conv2D, \
                            Input

from keras.models import    Model

from src.layers import  discriminator_conv2d_block, \
                        deconv2d_block, \
                        generator_conv2d_block, \
                        residual_block

def create_generator(imsize, label_count, n_residual):
    input_layer = Input((imsize, imsize, label_count + 3,), name = "generator_in")

    # Downsampling
    layers = generator_conv2d_block(input_layer, 64, 7, 1)
    layers = generator_conv2d_block(layers, 128, 4, 2)
    layers = generator_conv2d_block(layers, 256, 4, 2)

    # Bottleneck
    for _ in range(n_residual):
        layers = residual_block(layers, 256)

    # Upsampling
    layers = deconv2d_block(layers, 128, 4, 2)
    layers = deconv2d_block(layers, 64, 4, 2)

    # output
    output_layer = Conv2D(3, kernel_size = 7, strides = 1, padding = "same")(layers)
    output_layer = Activation("tanh", name = "generator_out")(output_layer)

    return Model(input_layer, output_layer, name = "generator")

def create_discriminator(imsize, label_count, n_hidden, leakyness):
    # Input
    input_layer = Input((imsize, imsize, 3,), name = "discriminator_in")
    layers = discriminator_conv2d_block(input_layer, 64, 4, 2)

    # Hidden
    numfilters = 128
    for _ in range(n_hidden):
        layers = discriminator_conv2d_block(layers, numfilters)
        numfilters *= 2

    real = Conv2D \
    (
        1,
        kernel_size = 3,
        strides = 1,
        padding = "same",
        name = "discriminator_out_real"
    )(layers)

    labels = Conv2D \
    (
        label_count,
        kernel_size = imsize // 64,
        strides = 1,
        padding = "valid",
        name = "discriminator_out_labels"
    )(layers)

    return Model(input_layer, [real, labels], name = "discriminator")
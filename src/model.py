import numpy as np
from src.instancenormalization import InstanceNormalization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU

class Model:
    
    def __init__(self, model_dir, log_dir, image_data, bottleneck_depth = 6, disc_hidden_depth = 5):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.image_data = image_data
        self.bottleneck_depth = bottleneck_depth
        self.disc_hidden_depth = disc_hidden_depth

        self.generator = None
        self.generator_input = None
        self.generator_output = None
        self.discriminator = None
        self.discriminator_input = None
        self.discriminator_src_out = None
        self.discriminator_cls_out = None

        self.build_generator()
        self.build_discriminator()

    def build_generator(self):
        #input layer
        self.generator_input = Input(self.image_data.image_dim())

        # down-sampling layers
        layers = Conv2D(64, kernel_size = 7, strides = 1, activation = "relu", padding = "same")(InstanceNormalization()(self.generator_input))
        layers = Conv2D(128, kernel_size = 4, strides = 2, activation = "relu", padding = "same")(InstanceNormalization()(layers))
        layers = Conv2D(256, kernel_size = 4, strides = 2, activation = "relu", padding = "same")(InstanceNormalization()(layers))
        
        # bottleneck layers
        for _ in range(0, self.bottleneck_depth):
            layers = Conv2D(256, kernel_size = 3, strides = 1, activation = "relu", padding = "same")(InstanceNormalization()(layers))

        # upsampling layers
        layers = Conv2DTranspose(128, kernel_size = 4, strides = 2, activation = "relu", padding = "same")(InstanceNormalization()(layers))
        layers = Conv2DTranspose(64, kernel_size = 4, strides = 2, activation = "relu", padding = "same")(InstanceNormalization()(layers))
        self.generator_output = Conv2D(3, kernel_size = 7, strides = 1, activation = "tanh", padding = "same")(InstanceNormalization()(layers))

        self.generator = keras.models.Model \
        (
            inputs = self.generator_input,
            outputs = self.generator_output,
            name = "Generator"
        )

        self.generator.compile \
        (
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
        )
        print("\nGenerator:")
        self.generator.summary()

    def build_discriminator(self):
        #input layer
        self.discriminator_input = Input(self.image_data.disc_dim())
        layers = Conv2D(64, kernel_size = 4, strides = 2, padding = "same")(self.discriminator_input)
        layers = LeakyReLU(0.01)(layers)

        # hidden layers
        n_filters = 128
        for _ in range(0, self.disc_hidden_depth):
            layers = Conv2D(n_filters, kernel_size = 4, strides = 2, padding = "same")(layers)
            layers = LeakyReLU(0.01)(layers)
            n_filters *= 2

        # output layers
        self.discriminator_src_out = Conv2D(1, kernel_size = 3, strides = 1, padding = "same")(layers)
        self.discriminator_cls_out = Conv2D \
        (
            self.image_data.num_classes(),
            kernel_size = 
            (
                self.image_data.disc_dim()[0] // 64,
                self.image_data.disc_dim()[1] // 64
            ),
            strides = 1,
            padding = "valid"
        )(layers)

        self.discriminator = keras.models.Model \
        (
            inputs = self.discriminator_input,
            outputs = [self.discriminator_src_out, self.discriminator_cls_out],
            name = "Discriminator"
        )

        print("\nDiscriminator:")
        self.discriminator.summary()


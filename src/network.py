import json
import keras
import os
import pickle
import src.loss as loss
import src.utility as util

from keras.layers import Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from src.imagedata import ImageData
from src.models import create_discriminator, create_generator
from utility import GradNorm

class Network:

    def __init__(self, arguments):
        self.arguments = json.dumps(vars(arguments), indent = 4)
        self.image_data = ImageData \
        (
            os.path.join(arguments.data_dir, "images/"),
            os.path.join(arguments.data_dir, "list_attr_celeba.csv"),
            arguments.image_size,
            arguments.batch_size,
            arguments.selected_labels
        )

        # general arguments
        self.image_size = arguments.image_size
        self.selected_labels = arguments.selected_labels
        self.num_labels = len(arguments.selected_labels)

        #training / network parameters
        self.adv_weight = arguments.adv_weight
        self.batch_size = arguments.batch_size
        self.cls_weight = arguments.cls_weight
        self.beta_1 = arguments.beta_1
        self.beta_2 = arguments.beta_2
        self.dis_hidden = arguments.dis_hidden
        self.dis_lr = arguments.dis_lr
        self.gen_lr = arguments.gen_lr
        self.gen_residuals = arguments.gen_residuals
        self.grad_pen = arguments.grad_pen
        self.leakyness = arguments.leakyness
        self.rec_weight = arguments.rec_weight

        # folders
        self.checkpoint_dir = arguments.checkpoint_dir
        self.data_dir = arguments.data_dir
        self.log_dir = arguments.log_dir
        self.model_dir = arguments.model_dir

        # generated variables
        self.generator = None # base
        self.discriminator = None # base
        self.combined_model = None # trainable
        self.discriminator_model = None # trainable

        # const ?!?
        self.checkpoint_name = "checkpoint_epoch_{}"
        self.weights_file = "weights.h5"
        self.combined_optimizer_file = "combined_optimizer.pkl"
        self.discriminator_optimizer_file = "discriminator_optimizer.pkl"

    '''
        Store / load the network. Store always goes to checkpoint.
        Load function is non-member
    '''

    def store_optimizer(self, model, filename):
        with open(filename, "wb") as file:
            values = getattr(model.optimizer, "weights")
            values = keras.backend.batch_get_value(values)
            pickle.dump(values, file)

    def load_optimizer(self, model, filename):
        with open(filename, "rb") as file:
            values = pickle.load(file)
            model._make_train_function()
            model.optimizer.set_weights(values)

    def store_network(self, epoch):
        folder = os.path.join(self.checkpoint_dir, self.checkpoint_name.format(epoch))
        os.makedirs(folder, exist_ok = True)

        # Store weights and optimizer
        self.combined_model.save_weights(os.path.join(folder, self.weights_file))
        self.store_optimizer(self.combined_model, os.path.join(folder, self.combined_optimizer_file))
        self.store_optimizer(self.discriminator_model, os.path.join(folder, self.discriminator_optimizer_file))

        # Store arguments
        with open("arguments.json", "w") as file:
            file.write(self.arguments)
    
    '''
        Builds the model
    '''
    def build(self):
        # Create base models
        self.generator = create_generator \
        (
            self.image_size,
            self.num_labels,
            self.gen_residuals
        )

        self.discriminator = create_discriminator \
        (
            self.image_size,
            self.num_labels,
            self.dis_hidden,
            self.leakyness
        )
        
        ##
        # Create combined model, discriminator does NOT get trained here.
        ##
        self.discriminator.trainable = False

        real_image = Input((self.image_size, self.image_size, 3,))
        real_labels = Input((self.image_size, self.image_size, self.num_labels,))
        fake_labels = Input((self.image_size, self.image_size, self.num_labels))

        generator_input = Concatenate(axis = 3)([real_image, fake_labels])
        # output of generator after image + fake_labels
        generator_fake_output = self.generator(generator_input)
        # output of discriminator pass on generated fake image
        fakeimg_real, fakeimg_labels = self.discriminator(generator_fake_output)
        # input layer for generated fake image, and origninal labels
        cycle_input = Concatenate(axis = 3)([generator_fake_output, real_labels])
        reconstructed_image = self.generator(cycle_input)

        self.combined_model = Model \
        (
            inputs = [real_image, real_labels, fake_labels],
            outputs = [reconstructed_image, fakeimg_real, fakeimg_labels],
            name = "Combined Model"
        )

        self.combined_model.compile \
        (
            loss = ["mae", loss.negative_mean_loss, loss.classification_loss],
            loss_weights = [self.rec_weight, self.cls_weight, self.adv_weight],
            optimizer = Adam(lr = self.gen_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        )

        # Create discriminator train model
        self.discriminator.trainable = True
        real_image = Input((self.image_size, self.image_size, 3,))
        fake_image = Input((self.image_size, self.image_size, 3,))
        # Gradient normalization pass
        interpolation = Input((self.image_size, self.image_size, 3,))
        normalized = GradNorm()([self.discriminator(interpolation)[0], interpolation])
        # Fake image pass
        fakeimg_real, _ = self.discriminator(fake_image)
        # real image pass
        realimg_real, realimg_labels = self.discriminator(real_image)

        self.discriminator_model = Model \
        (
            inputs = [real_image, fake_image, interpolation],
            outputs = [realimg_real, realimg_labels, fakeimg_real, normalized],
            name = "Trainable discriminator"
        )

        self.discriminator_model.compile \
        (
            loss = [loss.negative_mean_loss, loss.mean_loss, loss.classification_loss, "mse"],
            loss_weights = [1, 1, self.cls_weight, self.grad_pen],
            optimizer = Adam(lr = self.dis_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        )

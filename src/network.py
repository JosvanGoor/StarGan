import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import src.loss as loss
import src.utility as util
import tensorflow as tf

from io import BytesIO
from keras.layers import Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from src.imagedata import ImageData
from src.models import create_discriminator, create_generator
from tqdm import trange

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
        self.start_epoch = arguments.start_epoch

        #training / network parameters
        self.adv_weight = arguments.adv_weight
        self.batch_size = arguments.batch_size
        self.beta_1 = arguments.beta_1
        self.beta_2 = arguments.beta_2
        self.cls_weight = arguments.cls_weight
        self.critics = arguments.critics
        self.dis_hidden = arguments.dis_hidden
        self.dis_lr = arguments.dis_lr
        self.epochs = arguments.epochs
        self.gen_lr = arguments.gen_lr
        self.gen_residuals = arguments.gen_residuals
        self.grad_pen = arguments.grad_pen
        self.iterations = arguments.iterations
        self.leakyness = arguments.leakyness
        self.log_delay = arguments.log_delay
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
        with open(os.path.join(folder, "arguments.json"), "w") as file:
            file.write(self.arguments)
    
    '''
        Builds the model, this is required even when restoring the model
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
            loss_weights = [self.rec_weight, 1, self.adv_weight],
            optimizer = Adam(lr = self.gen_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        )

        ##
        # Create discriminator train model
        ##
        self.discriminator.trainable = True
        real_image = Input((self.image_size, self.image_size, 3,))
        fake_image = Input((self.image_size, self.image_size, 3,))
        # Gradient normalization pass
        interpolation = Input((self.image_size, self.image_size, 3,))
        normalized = util.GradNorm()([self.discriminator(interpolation)[0], interpolation])
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
            loss_weights = [1, self.cls_weight, self.rec_weight, self.grad_pen],
            optimizer = Adam(lr = self.dis_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        )

    '''
        Train model
    '''
    def train(self):
        tbcallback = keras.callbacks.TensorBoard(log_dir = self.log_dir, write_graph = False)
        tbcallback.set_model(self.combined_model)

        dis_names = ["Discriminator Real Loss", "Discriminator Classification Loss", "Discriminator Adverserial Loss", "Gradient Penalty"]
        gen_names = ["Generator Reconstruction Loss", "Generator Adverserial Loss", "Generator Classification Loss"]

        batch_iterator = iter(self.image_data)

        with keras.backend.get_session().as_default() as session:
            session.run(keras.backend.tf.global_variables_initializer())

            for epoch in trange(0, self.epochs, initial = self.start_epoch, desc = "Epochs"):
                # Store model every epoch (~10GB required for 200 epochs)
                if not self.start_epoch == epoch:
                    self.store_network(epoch)

                # Run iterations within epoch
                for step in trange(self.iterations, desc = "Iterations"):
                    current_step = (self.iterations * epoch) + step

                    # With log_delay intervals write log
                    if current_step % self.log_delay == 0:
                        self.write_tensorboard_image(tbcallback, current_step)

                    for _ in range(0, self.critics):
                        try:
                            train_images, train_labels, fake_labels = next(batch_iterator)
                        except:
                            batch_iterator = iter(self.image_data)
                            train_images, train_labels, fake_labels = next(batch_iterator)
                        
                        # Train the discriminator
                        fake_label_layers = util.generate_batch_label_layers(fake_labels, self.image_size)
                        fake_in_combined = np.append(train_images, fake_label_layers, axis = 3)
                        fake_images = self.generator.predict(fake_in_combined)

                        fake_src = np.zeros((self.batch_size, 2, 2, 1))
                        real_src = np.ones((self.batch_size, 2, 2, 1))

                        interpolation = (self.dis_lr * train_images) + 1 - self.dis_lr * fake_images

                        dis_logs = self.discriminator_model.train_on_batch \
                        (
                            [train_images, fake_images, interpolation],
                            [real_src, train_labels, fake_src, np.ones(self.batch_size)]
                        )
                    
                    # Train the combined model (generator)
                    original_label_layers = util.generate_batch_label_layers(train_labels, self.image_size)
                    gen_logs = self.combined_model.train_on_batch \
                    (
                        [train_images, original_label_layers, fake_label_layers],
                        [train_images, fake_src, fake_label_layers]
                    )

                    util.write_log(tbcallback, gen_names, gen_logs[1:4], current_step)
                    util.write_log(tbcallback, dis_names, [dis_logs[1]+dis_logs[2]] +dis_logs[3:5], current_step)

    def write_tensorboard_image(self, callback, step):
        image, train_label, fake_labels = self.image_data.get_validation_image()

        labels = util.generate_label_layers(fake_labels, self.image_size)
        combined = np.append(image, labels, axis = 2)
        generator_out = self.generator.predict(np.reshape(combined, (1, self.image_size, self.image_size, 3 + self.num_labels)))

        cycle_in = np.reshape(generator_out, (128, 128, 3))
        cycle_in = np.append(cycle_in, util.generate_label_layers(train_label, self.image_size), axis = 2)
        cycle_in = np.reshape(cycle_in, (1, self.image_size, self.image_size, 3 + self.num_labels))
        cycled = self.generator.predict(cycle_in)

        image_out = np.concatenate \
        (
            (
                image,
                generator_out.reshape((self.image_size, self.image_size, 3,)),
                cycled.reshape((self.image_size, self.image_size, 3,)),
            ),
            axis = 1
        )

        image_out = util.normalize_image(image_out)
        buf = BytesIO()
        plt.imsave(buf, image_out)
        tb_image = tf.Summary.Image(encoded_image_string = buf.getvalue())

        label_image = util.label_image(train_label, self.image_size)
        fakelbl_image = util.label_image(fake_labels, self.image_size)
        labels_image = np.concatenate((label_image, fakelbl_image), axis = 0)
        
        buf = BytesIO()
        plt.imsave(buf, labels_image)
        tb_labels = tf.Summary.Image(encoded_image_string = buf.getvalue())

        summary = tf.Summary \
        (
            value = \
            [
                tf.Summary.Value(tag = "in -> out -> cycled", image = tb_image),
                tf.Summary.Value(tag = "labels, (top: real): {}".format(self.selected_labels), image = tb_labels)
            ]
        )
        
        callback.writer.add_summary(summary, step)
        callback.writer.flush()

def restore_network(arguments):
    restore_epoch = arguments.start_epoch

    if arguments.restore_arguments: # overwrite arguments with old arguments
        argfile = os.path.join(arguments.checkpoint_dir, "checkpoint_epoch_{}".format(restore_epoch), "arguments.json")
        args = None
        with open(argfile) as file: args = json.load(file)
        arguments.__dict__.update(args) # this is so fucking ugly I cant even.

    # Generate network stuffs
    network = Network(arguments)
    network.start_epoch = restore_epoch
    network.build()

    # now restore the data
    folder = os.path.join(network.checkpoint_dir, network.checkpoint_name.format(network.start_epoch))

    network.combined_model.load_weights(os.path.join(folder, network.weights_file))
    network.load_optimizer(network.combined_model, os.path.join(folder, network.combined_optimizer_file))
    network.load_optimizer(network.discriminator_model, os.path.join(folder, network.discriminator_optimizer_file))

    return network


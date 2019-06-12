import json
import os
import src.utility as utility
from src.imagedata import ImageData
from src.model import create_generator, create_discriminator
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Input, Concatenate
from keras.models import Model
import keras
from tqdm import trange
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import pickle
import src.loss as loss

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

class Network:

    def __init__(self, arguments):
        self.image_data = ImageData \
        (
            os.path.join(arguments.dataset, "images/"),
            os.path.join(arguments.dataset, "list_attr_celeba.csv"),
            arguments.img_size,
            arguments.batch_size,
            arguments.selected_attrs,
            0.1
        )

        self.epochs = arguments.epoch
        self.iterations = arguments.iteration
        self.print_frequency = arguments.print_freq
        self.save_frequency = arguments.save_freq
        self.decay_flag = arguments.decay_flag
        self.decay_epoch = arguments.decay_epoch
        self.learning_rate = arguments.lr
        self.gradient_penalty = arguments.ld
        self.adverserial_weight = arguments.adv_weight
        self.reconstruction_weight = arguments.rec_weight
        self.classification_weight = arguments.cls_weight
        self.resblocks = arguments.n_res
        self.discriminatorblocks = arguments.n_dis
        self.augment_flag = arguments.augment_flag
        self.image_size = arguments.img_size
        self.selected_attributes = arguments.selected_attrs
        self.batch_size = arguments.batch_size

        self.checkpoint_dir = arguments.checkpoint_dir
        self.result_dir = arguments.result_dir
        self.log_dir = arguments.log_dir
        self.sample_dir = arguments.sample_dir
        self.starter_epoch = 0
        self.log_delay = 100

        self.model_foldername = "checkpoint_epoch_{}"
        self.weights_file = "weights.h5"
        self.combined_sym_file = "combined_sym.pkl"
        self.discriminator_sym_file = "discriminator_sym.pkl"

        print("batch_size: {}".format(self.batch_size))

    '''
        Store / restore the model
    '''
    def store_network(self, epoch):
        folder = os.path.join(self.checkpoint_dir, self.model_foldername.format(epoch))
        os.makedirs(folder, exist_ok = True)
        
        # store model weights
        self.combined_model.save_weights(os.path.join(folder, self.weights_file))

        # store combined optimizer
        with open(os.path.join(folder, self.combined_sym_file), "wb") as file:
            symw = getattr(self.combined_model.optimizer, "weights")
            values = tf.keras.backend.batch_get_value(symw)
            pickle.dump(values, file)

        # store combined optimizer
        with open(os.path.join(folder, self.discriminator_sym_file), "wb") as file:
            symw = getattr(self.discriminator_model.optimizer, "weights")
            values = tf.keras.backend.batch_get_value(symw)
            pickle.dump(values, file)

    def restore_network(self, epoch):
        folder = os.path.join(self.checkpoint_dir, self.model_foldername.format(epoch))

        self.starter_epoch = epoch
        self.combined_model.load_weights(os.path.join(folder, self.weights_file))

        # load combined sym weights
        with open(os.path.join(folder, self.combined_sym_file), "rb") as file:
            values = pickle.load(file)
            self.combined_model._make_train_function()
            self.combined_model.optimizer.set_weights(values)

        # load discriminator sym weights
        with open(os.path.join(folder, self.discriminator_sym_file), "rb") as file:
            values = pickle.load(file)
            self.discriminator_model._make_train_function()
            self.discriminator_model.optimizer.set_weights(values)

    '''
        Build the model
    '''
    def build(self):
        self.generator = create_generator \
        (
            self.image_size,
            len(self.selected_attributes),
            self.resblocks
        )
        
        self.discriminator = create_discriminator \
        (
            self.image_size,
            len(self.selected_attributes),
            self.discriminatorblocks
        )

        self.gen_optimizer = Adam(lr = self.learning_rate, beta_1 = 0.5, beta_2 = 0.99)
        self.dis_optimizer = Adam(lr = self.learning_rate, beta_1 = 0.5, beta_2 = 0.99)

        self.discriminator.trainable = False

        real_img = Input((self.image_size, self.image_size, 3))
        original_labels = Input((self.image_size, self.image_size, len(self.selected_attributes)))
        target_labels = Input((self.image_size, self.image_size, len(self.selected_attributes)))

        gen_input = Concatenate(axis = 3)([real_img, target_labels])
        combined_input = self.generator(gen_input)
        output_src, output_cls = self.discriminator(combined_input)
        middel_layer = Concatenate(axis = 3)([combined_input, original_labels])
        reconstructed_image = self.generator(middel_layer)

        self.combined_model = Model \
        (
            inputs = [real_img, original_labels, target_labels],
            outputs = [reconstructed_image, output_src, output_cls],
            name = "combined model"
        )

        print("Combined model:")
        self.combined_model.summary()

        self.combined_model.compile \
        (
            loss = ["mae", loss.negative_mean_loss, loss.classification_loss],
            loss_weights = [self.reconstruction_weight, self.classification_weight + 5, self.adverserial_weight],
            optimizer = self.gen_optimizer
        )

        # Train discriminator bit
        self.discriminator.trainable = True
        shape = (self.image_size, self.image_size, 3)
        fake_input, real_input, interpolation = Input(shape), Input(shape), Input(shape)
        norm = utility.GradNorm()([self.discriminator(interpolation)[0], interpolation])
        fake_output_src, fake_output_cls = self.discriminator(fake_input)
        real_output_src, real_output_cls = self.discriminator(real_input)
        self.discriminator_model = Model \
        (
            [real_input, fake_input, interpolation],
            [fake_output_src, real_output_src, real_output_cls, norm],
            name = "connected discriminator"
        )
        
        print("Discriminator model:")
        self.discriminator_model.summary()
        
        self.discriminator_model.compile \
        (
            loss = [loss.mean_loss, loss.negative_mean_loss, loss.classification_loss, "mse"],
            loss_weights = [1, 1, self.classification_weight, self.gradient_penalty],
            optimizer = self.dis_optimizer
        )
    
    '''
        Train the model
    '''
    def train(self):
        tbcallback = tf.keras.callbacks.TensorBoard(log_dir = self.log_dir, write_graph = False)
        tbcallback.set_model(self.combined_model)

        dis_names = ['Discriminator Adversarial loss', 'Discriminator Classification loss', 'Gradient Penalty']
        gen_names = ['Cycle loss', 'Generator Adversarial loss', 'Generator Classification loss']
        
        batch_iterator = iter(self.image_data)
        batch_idx = 0

        with tf.keras.backend.get_session().as_default() as session:
            # this helps with restoring contrib layers using tensorflow layers.
            session.run(keras.backend.tf.global_variables_initializer())

            for epoch in trange(0, self.epochs, initial = self.starter_epoch, desc = "Epochs"):
                # Store once per epoch
                if not self.starter_epoch == epoch:
                    self.store_network(epoch)

                for step in trange(self.iterations, desc = "Iterations"):
                    curstep = (self.iterations * epoch) + step

                    # Output tensorflow image
                    if curstep % self.log_delay == 0:
                        train_images, train_labels, fake_labels = self.image_data.get_validation_image()

                        image = train_images
                        labels = utility.generate_label_layers(fake_labels, self.image_size)
                        combined = np.append(image, labels, axis = 2)
                        gen_out = self.generator.predict(np.reshape(combined, (1, 128, 128, 8)))

                        cycle_in = np.reshape(gen_out, (128, 128, 3))
                        cycle_in = np.append(cycle_in, utility.generate_label_layers(train_labels, self.image_size), axis = 2)
                        cycle_in = np.reshape(cycle_in, (1, 128, 128, 8))
                        cycled = self.generator.predict(cycle_in)

                        buf = BytesIO()
                        image_out = np.concatenate \
                        (
                            ( image, gen_out.reshape((128, 128, 3)), cycled.reshape((128, 128, 3))),
                            axis = 1
                        )
                        image_out = utility.normalize_image(image_out)
                        plt.imsave(buf, image_out)
                        images = tf.Summary.Image(encoded_image_string = buf.getvalue())

                        lbl_img = utility.label_image(train_labels, self.image_size)
                        fake_img = utility.label_image(fake_labels, self.image_size)
                        labels_image = np.concatenate((lbl_img, fake_img), axis = 0)
                        
                        buf = BytesIO()
                        plt.imsave(buf, labels_image)
                        labels_image = tf.Summary.Image(encoded_image_string = buf.getvalue())

                        summary = tf.Summary(value = [tf.Summary.Value(tag = "in -> out -> cycled", image = images),
                                                    tf.Summary.Value(tag = "labels (top = real, bottom = fake): {}".format(self.selected_attributes), image = labels_image)])
                        tbcallback.writer.add_summary(summary, curstep)
                        tbcallback.writer.flush()
                        
                    # 0 - critics
                    for _ in range(0, 5):
                        try:
                            train_images, train_labels, fake_labels = next(batch_iterator)
                        except:
                            batch_iterator = iter(self.image_data)
                            train_images, train_labels, fake_labels = next(batch_iterator)
                        batch_idx += 1

                        fake_labels = utility.generate_batch_labels(fake_labels, self.image_size)
                        in_combined = np.append(train_images, fake_labels, axis = 3)
                        fake = self.generator.predict(in_combined)
                        
                        fake_src = np.zeros((self.batch_size, 2, 2, 1))
                        real_src = np.ones((self.batch_size, 2, 2, 1))

                        interpolation = self.learning_rate * (train_images) + (1 - self.learning_rate) * fake

                        d_logs = self.discriminator_model.train_on_batch \
                        (
                            [train_images, fake, interpolation],
                            [fake_src, real_src, train_labels, np.ones(self.batch_size)]
                        )

                    tiled_original_labels = utility.generate_batch_labels(train_labels, self.image_size)
                    tiled_target_labels = fake_labels
                    g_logs = self.combined_model.train_on_batch \
                    (
                        [train_images, tiled_original_labels, tiled_target_labels],
                        [train_images, real_src, fake_labels]
                    )

                    write_log(tbcallback, gen_names, g_logs[1:4], batch_idx)
                    write_log(tbcallback, dis_names, [d_logs[1]+d_logs[2]] +d_logs[3:5], batch_idx)




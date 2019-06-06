import os
from src.imagedata import ImageData
from src.model import create_generator, create_discriminator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

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

        self.checkpoint_dir = arguments.checkpoint_dir
        self.result_dir = arguments.result_dir
        self.log_dir = arguments.log_dir
        self.sample_dir = arguments.sample_dir

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

        self.gen_optimizer = Adam(lr = self.learning_rate, beta_1 = 0.99, beta_2 = 0.99)
        self.dis_optimizer = Adam(lr = self.learning_rate, beta_1 = 0.9, beta_2 = 0.99)

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
        
        self.combined_model.summary()
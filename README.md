# celeba_stargan
Stargan for CelebA. A Deep Learning practical

Keras implementation based on 
https://github.com/taki0112/StarGAN-Tensorflow

# Notes:

## Defaults:
epochs: 20
iterations: 1000
batch_size: 16
decay_flag: True
decay_epoch: 10
learn_rate: 0.0001
gradient_penalty_lambda: 10.0
adv_weigt: 1 // weight about gan
rec_weight: 10 //reconstruction weight
cls_weight: 10 // classification weight

base_channel_per_layer: 64
number_resblock: 6
number_discriminator_layer: 6
number_critics: 5
img_size: 128
img_channels: 3
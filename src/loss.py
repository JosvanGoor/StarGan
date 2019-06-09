import tensorflow as tf
import tensorflow.keras as keras

def wasserstein_loss(y_true, y_pred):
        """Calculates the Wasserstein loss for a sample batch.
        The Wasserstein loss function is very simple to calculate. In a standard GAN, the
        discriminator has a sigmoid output, representing the probability that samples are
        real or generated. In Wasserstein GANs, however, the output is linear with no
        activation function! Instead of being constrained to [0, 1], the discriminator wants
        to make the distance between its output for real and generated samples as
        large as possible.
        The most natural way to achieve this is to label generated samples -1 and real
        samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
        outputs by the labels will give you the loss immediately.
        Note that the nature of this loss means that it can be (and frequently will be)
        less than 0."""
        return keras.backend.mean(y_true * y_pred)

def classification_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = y_pred))

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mean_loss(y_true, y_pred):
    return keras.backend.mean(y_pred)

def negative_mean_loss(y_true, y_pred):
    return tf.math.scalar_mul(-1, keras.backend.mean(y_pred))
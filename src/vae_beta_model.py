from tensorflow import shape, exp, reduce_mean, reduce_sum, square, GradientTape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    ZeroPadding1D,
    Cropping1D,
    Reshape,
    Conv1D,
    Conv1DTranspose,
    BatchNormalization,
    Activation,
    Input,
    Flatten,
    Dense,
)
from tensorflow.keras.losses import (
    binary_crossentropy,
    mean_absolute_error,
)
from tensorflow.keras.backend import random_normal
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Normalization


normalizer = Normalization()

# parameters of the VAE model
input_shape = 2807
latent_dim = 8
channels = (16, 32, 64, 128)
strides = (4, 4, 4, 5)


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding an input sample."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = shape(z_mean)[0]
        dim = shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, encoder, decoder, beta=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.loss_tracker = Mean(name="mae")
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = reduce_mean(
                reduce_sum(binary_crossentropy(data, reconstruction), axis=1)
            )
            kl_loss = -0.5 * (1 + z_log_var - square(z_mean) - exp(z_log_var))
            kl_loss = reduce_mean(reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
            loss = reduce_mean(
                reduce_sum(mean_absolute_error(data, reconstruction), axis=1)
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.loss_tracker.update_state(loss)
        return {
            "mae": self.loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = reduce_mean(binary_crossentropy(inputs, reconstruction))
        reconstruction_loss *= 2807
        kl_loss = -0.5 * (1 + z_log_var - square(z_mean) - exp(z_log_var))
        kl_loss = reduce_mean(kl_loss)
        total_loss = reconstruction_loss + self.beta * kl_loss
        self.add_metric(kl_loss, name="kl_loss", aggregation="mean")
        self.add_metric(total_loss, name="total_loss", aggregation="mean")
        self.add_metric(
            reconstruction_loss, name="reconstruction_loss", aggregation="mean"
        )
        return reconstruction


# parameters of the VAE model
input_shape = 2807
latent_dim = 8
channels = (16, 32, 64, 128)
strides = (4, 4, 4, 5)
layers = 1
kernel_size = (8,)


def encoder_model(
    input_shape=(2807, 1),
    channels=channels,
    strides=strides,
    kernel_size=3,
    latent_dim=latent_dim,
):
    # input & padding
    encoder_inputs = Input(shape=input_shape)
    x = ZeroPadding1D(padding=((0, 393)))(encoder_inputs)
    # x = normalizer(encoder_inputs)
    for ch_n, str_n in zip(channels, strides):
        x = Conv1D(ch_n, kernel_size, padding="same", strides=1)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Conv1D(ch_n, kernel_size, padding="same", strides=str_n)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    # final fully-conneted layers
    x = Flatten()(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    return encoder_inputs, z_mean, z_log_var, z


def decoder_model(
    latent_dim=latent_dim, channels=channels[::-1], strides=strides[::-1], kernel_size=3
):

    # input & first fully connected layer
    latent_inputs = Input(shape=(latent_dim,))

    x = Dense(1280, activation="elu")(latent_inputs)
    x = Reshape((10, 128))(x)

    # convolutional layers
    for ch_n, str_n in zip(channels, strides):
        x = Conv1DTranspose(ch_n, kernel_size, padding="same", strides=str_n)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Conv1D(ch_n, kernel_size, padding="same", strides=1)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    # final layer & output
    x = Conv1DTranspose(1, 1, activation="relu", padding="same")(x)
    x = Cropping1D(cropping=(0, 393))(x)
    decoded = x

    return latent_inputs, decoded


def normalizer_adapt(X):
    normalizer.adapt(X)


def vae_model(beta=1):

    encoder_inputs, z_mean, z_log_var, z = encoder_model()
    encoder = Model(encoder_inputs, (z_mean, z_log_var, z), name="VAE_encoder")
    encoder.summary()

    decoder_inputs, decoder_outputs = decoder_model()
    decoder = Model(decoder_inputs, decoder_outputs, name="VAE_decoder")
    decoder.summary()

    return VAE(encoder, decoder, beta)

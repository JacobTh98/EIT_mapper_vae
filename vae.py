import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datagenerator import DataGenerator_ref
from extra_fcts import delta_perms
#from models import vae_model
#import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow import shape, exp, reduce_mean, reduce_sum, square, GradientTape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *#Layer, ZeroPadding1D, Cropping1D, Dropout, Reshape, Conv1D, Conv1DTranspose, BatchNormalization, Activation, MaxPooling1D, Input, Concatenate, Flatten, Dense, Add
from tensorflow.keras.losses import Huber, binary_crossentropy
from tensorflow.keras.backend import random_normal
from tensorflow.keras.metrics import Mean
from usflfcts import *
import pandas as pd
import seaborn as sns
import requests
#plt.rcParams['text.usetex'] = True
#plt.rcParams.update({'font.family': 'Serif'})
#telegram_KI_bot('Fertig Initialisiert')
#token = '5389645416:AAHaA6OQO2WZh5Snm-fp0mP4aVDg1T0vYW8'
#chat_id = '5140274444'

filepath = 'fin_data/samples_01/'
spath = 'fin_model/model_01/'

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding an input sample."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = shape(z_mean)[0]
        dim = shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = reduce_mean(reduce_sum(binary_crossentropy(data, reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - square(z_mean) - exp(z_log_var))
            kl_loss = reduce_mean(reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = reduce_mean(binary_crossentropy(inputs, reconstruction))
        reconstruction_loss *= 2807
        kl_loss = 1 + z_log_var - square(z_mean) - exp(z_log_var)
        kl_loss = reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        return reconstruction

input_shape = 2807
latent_dim = 8
# parameters of the VAE model
channels = (16,32,64,128)
strides = (4,4,4,5)

##Encoder
def encoder_model(input_shape=(2807,1), channels=channels, strides=strides, kernel_size=3, latent_dim=latent_dim):
    # input & padding
    encoder_inputs = Input(shape=input_shape)
    x = ZeroPadding1D(padding=((0,393)))(encoder_inputs)
    for ch_n, str_n in zip(channels, strides):
        x = Conv1D(ch_n, kernel_size, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(ch_n, kernel_size, padding='same', strides=str_n)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))
    return encoder_inputs, z_mean, z_log_var, z
##Decoder
def decoder_model(latent_dim=latent_dim, channels=channels[::-1], strides=strides[::-1], kernel_size=9):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(1280, activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5))(latent_inputs)###Neu
    x = Reshape((10, 128))(x)
    for ch_n, str_n in zip(channels, strides):
        x = Conv1DTranspose(ch_n, kernel_size, padding='same', strides=str_n)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(ch_n, kernel_size, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    # final layer
    x = Conv1DTranspose(1, 1, activation='relu', padding='same')(x)
    x = Cropping1D(cropping=(0, 393))(x)
    decoded = x
    return latent_inputs, decoded

def vae_model():
    encoder_inputs, z_mean, z_log_var, z = encoder_model()
    encoder = Model(encoder_inputs, (z_mean, z_log_var, z), name='VAE_encoder')
    encoder.summary()
    
    decoder_inputs, decoder_outputs = decoder_model()
    decoder = Model(decoder_inputs, decoder_outputs, name='VAE_decoder')
    decoder.summary()
    
    return VAE(encoder, decoder)

vae = vae_model()
vae.compile(optimizer=Adam())


# initialize data generators
print(filepath)
params = {'path' : filepath,
          'dim': 2807,#3122
          'batch_size': 32,
          'shuffle': True}
p = len(os.listdir(params['path']))-2000# Für Validierung der Daten
idx = np.arange(p)
np.random.shuffle(idx)

#np.savez(spath+'params_vae.npz', params = params)
training_generator = DataGenerator_ref(idx, **params)

print("len(training_generator.indexes)\t",len(training_generator.indexes))
print("delta_perm(params['path'])\t",delta_perm(params['path']))

# train model
# https://www.tensorflow.org/tutorials/keras/save_and_load

history = vae.fit(training_generator, epochs=2)
telegram_KI_bot("VAE-Fertig trainiert")

vae.save("model_"+len(os.listdir("vae_models")), save_format="tf")
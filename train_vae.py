from select import epoll
from datagenerator import DataGenerator
from extra_fcts import *
from vae_model import vae_model
import os
import numpy as np

from tensorflow.keras.optimizers import Adam

vae = vae_model()
vae.compile(optimizer=Adam())

fpath = read_dirs('data/')

# initialize data generators
params = {"path": fpath, "dim": 2807, "batch_size": 32, "shuffle": True}
check_train_data(params)

range = read_integer("How many samples do you want to train?\t")
idx = np.arange(0, range)
np.random.shuffle(idx)

print("\tPreperate samples for training...")
training_generator = DataGenerator(idx, **params)

epochs = read_integer("How many epochs do you want to train?\t")
print("\tStart vae training...")
history = vae.fit(training_generator, epochs=epochs)

sname = "vae_models/model_" + str(len(os.listdir("vae_models")))
vae.save(sname, save_format="tf")
print("Saved VAE model:\t", sname)

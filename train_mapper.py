import numpy as np
from mapper_model import mapper_model
from datagenerator import DataGenerator_elsig_ref
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import os
from tqdm import tqdm

from extra_fcts import read_dirs, read_integer

vae_path = read_dirs("vae_models/", "Which vae model do you wan to use?")
vae = keras.models.load_model(vae_path)
print("\tLoaded vae successfully.")

fpath = read_dirs("data/", "On which data-set base was this vae trained on?")

print("FPATH:", fpath)

params = {"path": fpath, "dim": 2807, "batch_size": 100, "shuffle": True}

range_ = read_integer("How many samples do you want to train?\t")

idx = np.arange(0, range_)
np.random.shuffle(idx)

training_generator = DataGenerator_elsig_ref(idx, **params)
print("#############", len(training_generator))

encodings = np.empty((range_, 8, 1))
el_signals = np.empty((range_, 192, 1))
reference = np.empty((range_, 2807, 1))

print("\tGenerate encodings, el_sign and reference.")
for n in tqdm(range(len(training_generator))):
    X, y = next(training_generator)
    _, _, z = vae.encoder.predict(X)
    encodings[
        n * params["batch_size"] : (n + 1) * params["batch_size"], :, :
    ] = np.expand_dims(z, axis=2)
    el_signals[n * params["batch_size"] : (n + 1) * params["batch_size"], :, :] = y
    reference[n * params["batch_size"] : (n + 1) * params["batch_size"], :, :] = X

mapper = mapper_model()
mapper.summary()

mapper.compile(Adam(), loss="mse")

epochs = read_integer("How many epochs do you want to train?\t")
print("\tStart mapper training...")

history = mapper.fit(
    el_signals, encodings, epochs=epochs, batch_size=params["batch_size"]
)

sname = "mapper_models/model_" + str(len(os.listdir("mapper_models")))
mapper.save(sname, save_format="tf")
print("Saved mapper model:\t", sname)

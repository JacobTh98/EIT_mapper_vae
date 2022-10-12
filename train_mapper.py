import numpy as np
from mapper_model import mapper_model
from datagenerator import DataGenerator_elsig_ref
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import os

vae = keras.models.load_model("vae_models/model_0")
print("\tLoaded vae successfully.")

params = {"path": "data/samples_01", "dim": 2807, "batch_size": 100, "shuffle": True}
idx = np.arange(0, 10000)
np.random.shuffle(idx)

training_generator = DataGenerator_elsig_ref(idx, **params)

encodings = np.empty((10000, 8, 1))
el_signals = np.empty((10000, 192, 1))
reference = np.empty((10000, 2807, 1))

for n in range(len(training_generator)):
    X, y = next(training_generator)
    _, _, z = vae.encoder.predict(X)
    encodings[n * 100 : (n + 1) * 100, :, :] = np.expand_dims(z, axis=2)
    el_signals[n * 100 : (n + 1) * 100, :, :] = y
    reference[n * 100 : (n + 1) * 100, :, :] = X

mapper = mapper_model()
mapper.summary()

mapper.compile(Adam(), loss="mse")

history = mapper.fit(el_signals, encodings, epochs=50, batch_size=128)

sname = "mapper_models/model_" + str(len(os.listdir("mapper_models")))
mapper.save(sname, save_format="tf")
print("Saved mapper model:\t", sname)

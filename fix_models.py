"""
Extracts weights from the broken .keras files and re-saves as .weights.h5
which load reliably regardless of Keras version.
Run once on the server: python fix_models.py
"""
import zipfile, os, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_NN(Nactions, Nobservations):
    return Sequential([
        Input(shape=(Nobservations,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(Nactions),
    ])

for name in ['DQN_offline_true', 'DQN_offline_false']:
    src = f'{name}.keras'
    tmp = f'{name}_extracted'

    with zipfile.ZipFile(src, 'r') as z:
        z.extractall(tmp)

    model = build_NN(3, 2)
    model.load_weights(os.path.join(tmp, 'model.weights.h5'))
    shutil.rmtree(tmp)

    # Save as plain weights file — no serialization issues
    model.save_weights(f'{name}.weights.h5')

    sample = np.zeros((1, 2), dtype=np.float32)
    print(f'{name}: Q-values = {model(sample).numpy()}')

print('Done. Saved DQN_offline_true.weights.h5 and DQN_offline_false.weights.h5')

"""
Re-saves the trained .keras models using the correct Input-layer architecture
so that weights load properly regardless of Keras version.
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

    # .keras files are zip archives — extract the weight file inside
    with zipfile.ZipFile(src, 'r') as z:
        z.extractall(tmp)

    weight_file = os.path.join(tmp, 'model.weights.h5')

    model = build_NN(3, 2)
    model.load_weights(weight_file)

    # Re-save with the clean architecture
    model.save(src)
    shutil.rmtree(tmp)

    # Quick sanity check — print a sample Q-value
    sample = np.zeros((1, 2), dtype=np.float32)
    print(f'{name}: Q-values for zero state = {model(sample).numpy()}')

print('Done. Both models re-saved.')

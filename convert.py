import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

m = Sequential([Input(shape=(2,)), Dense(64, 'relu'), Dense(64, 'relu'), Dense(3)])
m.load_weights('DQN_offline_true.weights.h5')
m.save('DQN_offline_true.keras')
print('Done')

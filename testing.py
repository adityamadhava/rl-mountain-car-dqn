import numpy as np
import gymnasium as gym
import pygame
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

_POS_MIN, _POS_MAX = -1.2,  0.6
_VEL_MIN, _VEL_MAX = -0.07, 0.07

def normalize_obs(obs):
    pos = (obs[0] - _POS_MIN) / (_POS_MAX - _POS_MIN) * 2.0 - 1.0
    vel = (obs[1] - _VEL_MIN) / (_VEL_MAX - _VEL_MIN) * 2.0 - 1.0
    return np.array([pos, vel], dtype=np.float32)

def build_NN(Nactions, Nobservations):
    model = Sequential([
        Input(shape=(Nobservations,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(Nactions),
    ])
    return model

def choose_action(x, model, Nactions):
    # Choose action using the trained DQN model; no exploration at test time.
    Q_val = model(np.expand_dims(x, axis=0), training=False).numpy()[0]
    action = np.argmax(Q_val)
    return action


# The following lines load the DQN model.
# Path for model trained without offline data: 'DQN_offline_false.weights.h5'
# Path for model trained with offline data:    'DQN_offline_true.weights.h5'
Nactions = 3
model = build_NN(Nactions, 2)
model.load_weights('DQN_offline_true.weights.h5')

# Initialize the Mountain Car environment with render_mode='human' for animation.
env = gym.make('MountainCar-v0', render_mode='human')

# Reset the environment to get the initial state.
x_raw, _ = env.reset()
x = normalize_obs(x_raw)

end_episode = False
total_reward = 0
while not(end_episode):
    # Pick an action using choose_action() function.
    a = choose_action(x, model, Nactions)

    # Take the picked action; get next state, reward, and episode flags.
    x_dash_raw, r, terminated, truncated, _ = env.step(a)

    # Update the total reward.
    total_reward += r

    # Update the state for the next time slot.
    x = normalize_obs(x_dash_raw)

    # Update end_episode for the next time slot.
    end_episode = terminated or truncated


# Print the total reward.
print('Total reward = {}'.format(np.round(total_reward, 2)))

# Close the environment.
env.close()

pygame.display.quit()

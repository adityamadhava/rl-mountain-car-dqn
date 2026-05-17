import numpy as np
import gymnasium as gym
import pygame
from tensorflow.keras.models import load_model

def choose_action(x, model, Nactions):
    # Choose action using the trained DQN model; no exploration at test time.
    Q_val = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    action = np.argmax(Q_val)
    return action


# The following lines load the DQN model.
# Path for model trained without offline data: 'DQN_offline_false.h5'
# Path for model trained with offline data:    'DQN_offline_true.h5'
model = load_model('DQN_offline_true.h5')
Nactions = 3

# Initialize the Mountain Car environment with render_mode='human' for animation.
env = gym.make('MountainCar-v0', render_mode='human')

# Reset the environment to get the initial state.
x, _ = env.reset()

end_episode = False
total_reward = 0
while not(end_episode):
    # Pick an action using choose_action() function.
    a = choose_action(x, model, Nactions)

    # Take the picked action; get next state, reward, and episode flags.
    x_dash, r, terminated, truncated, _ = env.step(a)

    # Update the total reward.
    total_reward += r

    # Update the state for the next time slot.
    x = np.copy(x_dash)

    # Update end_episode for the next time slot.
    end_episode = terminated or truncated


# Print the total reward.
print('Total reward = {}'.format(np.round(total_reward, 2)))

# Close the environment.
env.close()

pygame.display.quit()

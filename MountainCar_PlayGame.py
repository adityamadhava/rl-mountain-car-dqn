import numpy as np
import gymnasium as gym
from gymnasium.utils.play import play
import pygame

global total_reward, k
total_reward = 0
k = 0

def total_reward_func(state, next_state, action, reward, terminated, truncated, info):
    global total_reward
    global k
    
    # This if-else statement is used to calculate and display the total reward
    # of an episode.
    if terminated:
        total_reward+=reward
        print('Total reward in this episode is {}, Iteration = {}'.format(total_reward, k))
        total_reward = 0
        k = 0
    else:
        total_reward+=reward
        k+=1



# Initiate the lunar lander environment
env = gym.make('MountainCar-v0', render_mode='rgb_array')

# Play the game
play(env, keys_to_action={'a': 0, 's': 1, 'd': 2}, noop=1, fps=50, callback=total_reward_func)

# Close the environment
env.close()

# The following line needs to be executed to close the display incase the game
# was abruptly stopped.
# pygame.display.quit()
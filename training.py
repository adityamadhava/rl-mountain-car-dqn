import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- GPU setup: predict network on GPU 0, target network on GPU 1 ---
_gpus = tf.config.list_physical_devices('GPU')
_PREDICT_DEV = '/GPU:0' if _gpus else '/CPU:0'
_TARGET_DEV  = '/GPU:1' if len(_gpus) >= 2 else _PREDICT_DEV
print(f'Predict → {_PREDICT_DEV} | Target → {_TARGET_DEV}')


def load_offline_data(path, min_score):
    state_data = []
    action_data = []
    reward_data = []
    next_state_data = []
    terminated_data = []

    dataset = pd.read_csv(path)
    dataset_group = dataset.groupby('Episode #')
    for play_no, df in dataset_group:
        start_idx = 0
        if isinstance(df.iloc[0, 1], str) and '{}' in df.iloc[0, 1]:
            start_idx = 1
        df = df[start_idx:]

        state = []
        for s in df.iloc[:, 1]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                state.append([float(val.strip(',')) for val in s])
            else:
                state.append(s)
        state = np.array(state)

        action = np.array(df.iloc[:, 2]).astype(int)
        reward = np.array(df.iloc[:, 3]).astype(np.float32)

        next_state = []
        for s in df.iloc[:, 4]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                next_state.append([float(val.strip(',')) for val in s])
            else:
                next_state.append(s)
        next_state = np.array(next_state)

        terminated = np.array(df.iloc[:, 5]).astype(int)

        total_reward = np.sum(reward)
        if total_reward >= min_score:
            state_data.append(state)
            action_data.append(action)
            reward_data.append(reward)
            next_state_data.append(next_state)
            terminated_data.append(terminated)

    if not state_data:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    state_data      = np.concatenate(state_data)
    action_data     = np.concatenate(action_data)
    reward_data     = np.concatenate(reward_data)
    next_state_data = np.concatenate(next_state_data)
    terminated_data = np.concatenate(terminated_data)

    return state_data, action_data, reward_data, next_state_data, terminated_data


def build_NN(Nactions, Nobservations):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(Nobservations,)),
        Dense(64, activation='relu'),
        Dense(Nactions),
    ])
    return model


def exploration_prob_scheduler(episode):
    # Slower decay so the car explores long enough to accidentally reach the goal.
    # 0.999^episode hits 0.1 around episode 2300, which suits a 3000-episode run.
    return max(0.1, 1.0 * (0.999 ** episode))


# MountainCar observation bounds for normalisation
_POS_MIN, _POS_MAX = -1.2,  0.6
_VEL_MIN, _VEL_MAX = -0.07, 0.07

def normalize_obs(obs):
    """Scale position to [-1,1] and velocity to [-1,1] before feeding the network."""
    pos = (obs[..., 0:1] - _POS_MIN) / (_POS_MAX - _POS_MIN) * 2.0 - 1.0
    vel = (obs[..., 1:2] - _VEL_MIN) / (_VEL_MAX - _VEL_MIN) * 2.0 - 1.0
    return np.concatenate([pos, vel], axis=-1).astype(np.float32)


def choose_actions_batch(obs_batch, model, Nactions, epsilon):
    """Per-env epsilon-greedy with a single batched forward pass."""
    explore  = np.random.uniform(size=len(obs_batch)) < epsilon
    Q_vals   = model(tf.constant(obs_batch, dtype=tf.float32), training=False).numpy()
    greedy   = np.argmax(Q_vals, axis=-1)
    random_a = np.random.randint(0, Nactions, size=len(obs_batch))
    return np.where(explore, random_a, greedy)


def add_batch_to_buffer(obs, actions, rewards, next_obs, terminateds,
                         s_buf, a_buf, r_buf, ns_buf, t_buf,
                         buffer_size, buffer_counter, buffer_ix):
    """Vectorised (loop-free) FIFO buffer insert for a batch of N transitions."""
    N = len(obs)
    idx = (np.arange(N) + buffer_ix) % buffer_size
    s_buf[idx]  = obs
    a_buf[idx]  = actions
    r_buf[idx]  = rewards
    ns_buf[idx] = next_obs
    t_buf[idx]  = terminateds
    buffer_counter = min(buffer_counter + N, buffer_size)
    buffer_ix      = (buffer_ix + N) % buffer_size
    return buffer_counter, buffer_ix


def make_train_step(model_predict, model_target, optimizer, beta):
    """
    Returns a tf.function-compiled Double DQN train step.
    model_predict lives on _PREDICT_DEV, model_target on _TARGET_DEV.
    Both forward passes run concurrently on separate GPUs when they have
    no data dependency between them.
    """
    @tf.function
    def train_step(states, actions, rewards, next_states, terminateds):
        with tf.GradientTape() as tape:
            # These two calls have no mutual dependency → TF schedules them
            # concurrently on GPU:0 and GPU:1 respectively.
            q_next_predict = model_predict(next_states, training=False)
            q_next_target  = model_target(next_states,  training=False)

            # Double DQN: predict picks action, target evaluates its value
            best_next_a = tf.argmax(q_next_predict, axis=1, output_type=tf.int32)
            batch_sz    = tf.shape(states)[0]
            gather_idx  = tf.stack([tf.range(batch_sz), best_next_a], axis=1)
            Q_next      = tf.gather_nd(q_next_target, gather_idx)

            td_targets = rewards + beta * Q_next * (1.0 - terminateds)

            q_current  = model_predict(states, training=True)
            action_idx = tf.stack([tf.range(batch_sz), actions], axis=1)
            q_taken    = tf.gather_nd(q_current, action_idx)

            loss = tf.reduce_mean(tf.square(td_targets - q_taken))

        grads = tape.gradient(loss, model_predict.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_predict.trainable_variables))
        return loss

    return train_step


def plot_reward(total_reward_per_episode, window_length):
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(total_reward_per_episode) + 1)

    plt.plot(episodes, total_reward_per_episode,
             alpha=0.4, color='steelblue', label='Total Reward per Episode')

    if len(total_reward_per_episode) >= window_length:
        moving_avg = np.convolve(
            total_reward_per_episode,
            np.ones(window_length) / window_length,
            mode='valid'
        )
        avg_ep = np.arange(window_length, len(total_reward_per_episode) + 1)
        plt.plot(avg_ep, moving_avg, color='red', linewidth=2,
                 label=f'Moving Average (window={window_length})')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Double DQN Training: Mountain Car')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def DQN_training(env, offline_data, use_offline_data):
    # The function should return the final trained predict DQN model and
    # total reward of every episode.

    # ---------- Hyperparameters ----------
    N_ENVS      = 16       # Parallel async environments (uses all CPU cores)
    Nu          = 4        # Env steps between predict-network updates
    Nb          = 512      # Batch size (GPU-efficient; split across 2 GPUs)
    Nt          = 20       # Predict updates between target-network syncs
    beta        = 0.99     # Discount factor
    Nepisodes   = 3000     # Total episodes to train (counted across all envs)
    alpha       = 0.001    # Learning rate
    Nsave       = 200      # Predict updates between periodic saves
    buffer_size = 100_000  # Replay buffer size
    Nactions    = 3        # MountainCar: push-left / no-push / push-right
    Nobservations = 2      # MountainCar: position, velocity
    E           = 50       # (offline mode) episodes before adding env data to buffer

    # --- Build models: predict on GPU:0, target on GPU:1 ---
    with tf.device(_PREDICT_DEV):
        model_predict = build_NN(Nactions, Nobservations)
        optimizer     = Adam(learning_rate=alpha)
    with tf.device(_TARGET_DEV):
        model_target  = build_NN(Nactions, Nobservations)

    model_target.set_weights(model_predict.get_weights())

    # JIT-compiled Double DQN train step
    train_step = make_train_step(model_predict, model_target, optimizer, beta)

    # --- Replay buffer (float32 throughout to avoid repeated casts) ---
    s_buf  = np.zeros((buffer_size, Nobservations), dtype=np.float32)
    a_buf  = np.zeros(buffer_size,                  dtype=np.int32)
    r_buf  = np.zeros(buffer_size,                  dtype=np.float32)
    ns_buf = np.zeros((buffer_size, Nobservations), dtype=np.float32)
    t_buf  = np.zeros(buffer_size,                  dtype=np.float32)

    buffer_counter = 0
    buffer_ix      = 0

    # Pre-fill replay buffer with human-collected offline data (normalised)
    if use_offline_data:
        s_d, a_d, r_d, ns_d, t_d = offline_data
        if len(s_d) > 0:
            buffer_counter, buffer_ix = add_batch_to_buffer(
                normalize_obs(s_d), a_d.astype(np.int32),
                r_d.astype(np.float32), normalize_obs(ns_d),
                t_d.astype(np.float32),
                s_buf, a_buf, r_buf, ns_buf, t_buf,
                buffer_size, buffer_counter, buffer_ix
            )

    counter_target     = 0
    counter_save       = 0
    step_counter       = 0
    completed_episodes = 0
    episode_rewards    = np.zeros(N_ENVS, dtype=np.float32)
    total_reward_per_episode = []

    model_name = 'DQN_offline_true' if use_offline_data else 'DQN_offline_false'

    # 16 async environments — each runs in its own subprocess
    vec_env = gym.vector.AsyncVectorEnv(
        [lambda: gym.make('MountainCar-v0')] * N_ENVS
    )
    obs, _ = vec_env.reset()
    obs = normalize_obs(obs)

    while completed_episodes < Nepisodes:
        epsilon = exploration_prob_scheduler(completed_episodes)

        # One batched forward pass covers all N_ENVS environments
        actions = choose_actions_batch(obs, model_predict, Nactions, epsilon)

        obs_next_raw, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        obs_next = normalize_obs(obs_next_raw)
        dones    = terminateds | truncateds
        episode_rewards += rewards

        # gymnasium vector envs auto-reset terminated envs; the true terminal
        # observation (needed for the Bellman target) is in infos['final_observation']
        next_obs_buf = obs_next.copy()
        final_obs  = infos.get('final_observation', None)
        final_mask = infos.get('_final_observation', dones)
        if final_obs is not None:
            for i in range(N_ENVS):
                if final_mask[i] and final_obs[i] is not None:
                    next_obs_buf[i] = normalize_obs(final_obs[i][np.newaxis])[0]

        # Hold off adding env data for the first E episodes in offline mode
        if not use_offline_data or completed_episodes >= E:
            buffer_counter, buffer_ix = add_batch_to_buffer(
                obs, actions.astype(np.int32), rewards.astype(np.float32),
                next_obs_buf, terminateds.astype(np.float32),
                s_buf, a_buf, r_buf, ns_buf, t_buf,
                buffer_size, buffer_counter, buffer_ix
            )

        # Log and count completed episodes
        for i in range(N_ENVS):
            if dones[i]:
                total_reward_per_episode.append(float(episode_rewards[i]))
                print('Episode = {}, Total reward = {:.2f}, Epsilon = {:.4f}'.format(
                    completed_episodes + 1, episode_rewards[i], epsilon))
                episode_rewards[i] = 0.0
                completed_episodes += 1
                if completed_episodes >= Nepisodes:
                    break

        obs = obs_next
        step_counter += 1

        # Train every Nu env-steps
        if step_counter % Nu == 0 and buffer_counter >= Nb:
            ix = np.random.choice(buffer_counter, size=Nb, replace=False)

            # Cast once here; tf.function avoids retracing for same dtypes
            train_step(
                tf.constant(s_buf[ix]),
                tf.constant(a_buf[ix]),
                tf.constant(r_buf[ix]),
                tf.constant(ns_buf[ix]),
                tf.constant(t_buf[ix]),
            )

            counter_target += 1
            if counter_target == Nt:
                model_target.set_weights(model_predict.get_weights())
                counter_target = 0

            counter_save += 1
            if counter_save == Nsave:
                model_predict.save(model_name + '.keras')
                counter_save = 0

    vec_env.close()
    return model_predict, np.array(total_reward_per_episode)


# Initiate the mountain car environment.
# NO RENDERING. It will slow the training process.
env = gym.make('MountainCar-v0')

# Load the offline data collected in step 3. Also, process the dataset.
path = 'car_dataset.csv' # This should contain the path to the collected dataset.
min_score = -np.inf # The minimum total reward of an episode that should be used for training.
offline_data = load_offline_data(path, min_score)

# Train DQN model
use_offline_data = True # If True then the offline data will be used. Else, offline data will not be used.
final_model, total_reward_per_episode = DQN_training(env, offline_data, use_offline_data)

# Save the final model
model_name = 'DQN_offline_true' if use_offline_data else 'DQN_offline_false'
final_model.save(model_name + '.h5')

# Plot reward per episode and moving average reward
window_length = 50    # Window length for moving average reward.
plot_reward(total_reward_per_episode, window_length)

env.close()

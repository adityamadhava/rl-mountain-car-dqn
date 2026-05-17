"""
Fine-tunes the existing trained model with epsilon decaying all the way to 0.
Loads saved weights, trains for 3000 more episodes, saves updated model.
Run: python continue_training.py
"""
import zipfile, os, shutil
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

_gpus = tf.config.list_physical_devices('GPU')
_PREDICT_DEV = '/GPU:0' if _gpus else '/CPU:0'
_TARGET_DEV  = '/GPU:1' if len(_gpus) >= 2 else _PREDICT_DEV

_POS_MIN, _POS_MAX = -1.2,  0.6
_VEL_MIN, _VEL_MAX = -0.07, 0.07

def normalize_obs(obs):
    pos = (obs[..., 0:1] - _POS_MIN) / (_POS_MAX - _POS_MIN) * 2.0 - 1.0
    vel = (obs[..., 1:2] - _VEL_MIN) / (_VEL_MAX - _VEL_MIN) * 2.0 - 1.0
    return np.concatenate([pos, vel], axis=-1).astype(np.float32)

def build_NN(Nactions, Nobservations):
    model = Sequential([
        Input(shape=(Nobservations,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(Nactions),
    ])
    return model

def load_from_keras_zip(model, keras_path):
    tmp = keras_path + '_tmp'
    with zipfile.ZipFile(keras_path, 'r') as z:
        z.extractall(tmp)
    model.load_weights(os.path.join(tmp, 'model.weights.h5'))
    shutil.rmtree(tmp)

def exploration_prob_scheduler(episode, total_episodes):
    # Decays from 0.1 all the way to 0.0 — forces the model to develop a true greedy policy.
    return max(0.0, 0.1 * (1.0 - episode / total_episodes))

def choose_actions_batch(obs_batch, model, Nactions, epsilon):
    explore  = np.random.uniform(size=len(obs_batch)) < epsilon
    Q_vals   = model(tf.constant(obs_batch, dtype=tf.float32), training=False).numpy()
    greedy   = np.argmax(Q_vals, axis=-1)
    random_a = np.random.randint(0, Nactions, size=len(obs_batch))
    return np.where(explore, random_a, greedy)

def add_batch_to_buffer(obs, actions, rewards, next_obs, terminateds,
                         s_buf, a_buf, r_buf, ns_buf, t_buf,
                         buffer_size, buffer_counter, buffer_ix):
    N   = len(obs)
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
    @tf.function
    def train_step(states, actions, rewards, next_states, terminateds):
        with tf.GradientTape() as tape:
            q_next_predict = model_predict(next_states, training=False)
            q_next_target  = model_target(next_states,  training=False)
            best_next_a    = tf.argmax(q_next_predict, axis=1, output_type=tf.int32)
            batch_sz       = tf.shape(states)[0]
            gather_idx     = tf.stack([tf.range(batch_sz), best_next_a], axis=1)
            Q_next         = tf.gather_nd(q_next_target, gather_idx)
            td_targets     = rewards + beta * Q_next * (1.0 - terminateds)
            q_current      = model_predict(states, training=True)
            action_idx     = tf.stack([tf.range(batch_sz), actions], axis=1)
            q_taken        = tf.gather_nd(q_current, action_idx)
            loss           = tf.reduce_mean(tf.square(td_targets - q_taken))
        grads = tape.gradient(loss, model_predict.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_predict.trainable_variables))
        return loss
    return train_step

# ---------- Config ----------
MODEL_IN  = 'DQN_offline_true.keras'   # model to fine-tune
MODEL_OUT = 'DQN_offline_true.keras'   # overwrite with improved model
Nactions      = 3
Nobservations = 2
N_ENVS        = 16
Nu, Nb, Nt    = 4, 512, 20
beta          = 0.99
alpha         = 0.0005        # lower LR for fine-tuning
Nepisodes     = 3000
buffer_size   = 100_000

# ---------- Build & load ----------
with tf.device(_PREDICT_DEV):
    model_predict = build_NN(Nactions, Nobservations)
    optimizer     = Adam(learning_rate=alpha)
with tf.device(_TARGET_DEV):
    model_target  = build_NN(Nactions, Nobservations)

load_from_keras_zip(model_predict, MODEL_IN)
model_target.set_weights(model_predict.get_weights())
train_step = make_train_step(model_predict, model_target, optimizer, beta)

# ---------- Replay buffer ----------
s_buf  = np.zeros((buffer_size, Nobservations), dtype=np.float32)
a_buf  = np.zeros(buffer_size, dtype=np.int32)
r_buf  = np.zeros(buffer_size, dtype=np.float32)
ns_buf = np.zeros((buffer_size, Nobservations), dtype=np.float32)
t_buf  = np.zeros(buffer_size, dtype=np.float32)
buffer_counter = 0
buffer_ix      = 0

# ---------- Training loop ----------
counter_target = counter_save = step_counter = completed_episodes = 0
episode_rewards = np.zeros(N_ENVS, dtype=np.float32)
total_reward_per_episode = []

vec_env = gym.vector.AsyncVectorEnv([lambda: gym.make('MountainCar-v0')] * N_ENVS)
obs, _ = vec_env.reset()
obs = normalize_obs(obs)

while completed_episodes < Nepisodes:
    epsilon = exploration_prob_scheduler(completed_episodes, Nepisodes)
    actions = choose_actions_batch(obs, model_predict, Nactions, epsilon)

    obs_next_raw, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    obs_next = normalize_obs(obs_next_raw)
    dones    = terminateds | truncateds
    episode_rewards += rewards

    next_obs_buf = obs_next.copy()
    final_obs    = infos.get('final_observation', None)
    final_mask   = infos.get('_final_observation', dones)
    if final_obs is not None:
        for i in range(N_ENVS):
            if final_mask[i] and final_obs[i] is not None:
                next_obs_buf[i] = normalize_obs(final_obs[i][np.newaxis])[0]

    buffer_counter, buffer_ix = add_batch_to_buffer(
        obs, actions.astype(np.int32), rewards.astype(np.float32),
        next_obs_buf, terminateds.astype(np.float32),
        s_buf, a_buf, r_buf, ns_buf, t_buf,
        buffer_size, buffer_counter, buffer_ix
    )

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

    if step_counter % Nu == 0 and buffer_counter >= Nb:
        ix = np.random.choice(buffer_counter, size=Nb, replace=False)
        train_step(tf.constant(s_buf[ix]), tf.constant(a_buf[ix]),
                   tf.constant(r_buf[ix]), tf.constant(ns_buf[ix]),
                   tf.constant(t_buf[ix]))
        counter_target += 1
        if counter_target == Nt:
            model_target.set_weights(model_predict.get_weights())
            counter_target = 0
        counter_save += 1
        if counter_save == 200:
            model_predict.save(MODEL_OUT)
            counter_save = 0

vec_env.close()
model_predict.save(MODEL_OUT)
print('Saved to', MODEL_OUT)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(total_reward_per_episode, alpha=0.4, label='Reward')
if len(total_reward_per_episode) >= 50:
    ma = np.convolve(total_reward_per_episode, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(total_reward_per_episode)), ma, color='red', label='MA-50')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Fine-tuning: epsilon 0.1 → 0.0')
plt.legend(); plt.tight_layout()
plt.savefig('finetune_plot.png', dpi=150)
plt.show()

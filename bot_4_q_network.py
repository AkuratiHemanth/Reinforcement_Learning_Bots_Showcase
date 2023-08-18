from typing import List
import gym
import numpy as np
import random
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def one_hot(i: int, n: int) -> np.array:
    return np.identity(n)[i].reshape((1, -1))

def print_report(rewards: List, episode: int):
    print(report % (
        np.mean(rewards[-100:]),
        max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
        np.mean(rewards),
        episode
    ))

def main():
    env = gym.make('FrozenLake-v1')
    rewards = []

    n_obs, n_actions = env.observation_space.n, env.action_space.n
    obs_t_ph = tf.compat.v1.placeholder(shape=[1, n_obs], dtype=tf.float32)
    obs_tpl_ph = tf.compat.v1.placeholder(shape=[1, n_obs], dtype=tf.float32)
    act_ph = tf.compat.v1.placeholder(tf.int32, shape=())
    rew_ph = tf.compat.v1.placeholder(shape=(), dtype=tf.float32)
    q_target_ph = tf.compat.v1.placeholder(shape=[1, n_actions], dtype=tf.float32)

    W = tf.Variable(tf.random.uniform([n_obs, n_actions], 0, 0.01))
    q_current = tf.matmul(obs_t_ph, W)
    q_target = tf.matmul(obs_tpl_ph, W)

    q_target_max = tf.reduce_max(q_target, axis=1)
    q_target_sa = rew_ph + discount_factor * q_target_max
    q_current_sa = q_current[0, act_ph]
    error = tf.reduce_sum(tf.square(q_target_sa - q_current_sa))
    pred_act_ph = tf.argmax(q_current, 1)

    trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update_model = trainer.minimize(error)

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        for episode in range(1, num_episodes + 1):
            obs_t = env.reset()
            episode_reward = 0

            while True:
                obs_t_oh = one_hot(obs_t[0], n_obs)
                action = session.run(pred_act_ph, feed_dict={obs_t_ph: obs_t_oh})[0]

                if np.random.rand(1) < exploration_probability(episode):
                    action = env.action_space.sample()

                obs_tpl, reward, done, _, info = env.step(action)
                episode_reward += reward

                obs_tpl_oh = one_hot(obs_tpl, n_obs)
                q_target_val = session.run(q_target, feed_dict={obs_tpl_ph: obs_tpl_oh})
                session.run(update_model, feed_dict={
                    q_target_ph: q_target_val,
                    rew_ph: reward,
                    act_ph: action,
                    obs_t_ph: obs_t_oh
                })

                episode_reward += reward
                obs_t = obs_tpl

                if done:
                    rewards.append(episode_reward)
                    if episode % report_interval == 0:
                        print_report(rewards, episode)
                    break
    print_report(rewards, -1)

if __name__ == '__main__':
    num_episodes = 4000
    discount_factor = 0.99
    learning_rate = 0.15
    report_interval = 500
    exploration_probability = lambda episode: 50.0 / (episode + 10)
    report = '100-ep Average: %.2f. Best 100-ep Average: %.2f. Average: %.2f (Episode %d)' \
             '(Episode %d)'
    main()

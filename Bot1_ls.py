from typing import Tuple
from typing import Callable
from typing import List

import gym
import numpy as np
import random

num_episodes = 5000
discount_factor = 0.85
learning_rate = 0.9
v_lr = 0.5
report_interval = 500

report = '100-ep Average: %.2f. Best 100-ep Average: %.2f. Average: %.2f (Episode %d)' \
             '(Episode %d)'

def makeQ(model: np.array) -> Callable[[np.array], np.array]:
    return lambda X: X.dot(model)

def initialize(shape: Tuple):
    W = np.random.normal(0.0, 0.1, shape)
    Q = makeQ(W)
    return W, Q

def train(X: np.array, y: np.array, W: np.array) -> Tuple[np.array, Callable]:
    I = np.eye(X.shape[1])
    newW = np.linalg.inv(X.T.dot(X) + 1e-4 * I).dot(X.T.dot(y))
    W = learning_rate * newW + (1 - learning_rate) * W
    Q = makeQ(W)
    return W, Q

def one_hot(i: int, n: int) -> np.array:
    return np.identity(n)[i]

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
    W, Q = initialize((n_obs, n_actions))
    states, labels = [], []

    for episode in range(1, num_episodes + 1):
        if len(states) >= 10000:
            states, labels = [], []
        state = one_hot(env.reset(), n_obs)
        episode_reward = 0

        while True:
            states.append(state)
            noise = np.random.random((1, env.action_space.n)) / (episode ** 2.)
            action = np.argmax(Q(state) + noise)
            state2, reward, done, _, info = env.step(action)

            state2 = one_hot(state2, n_obs)
            Qtarget = reward + discount_factor * np.max(Q(state2))  # Fixed line
            label = Q(state)
            label[action] = (1 - learning_rate) * label[action] + learning_rate * Qtarget
            labels.append(label)

            episode_reward += reward
            state = state2

            if len(states) % 10 == 0:
                W, Q = train(np.array(states), np.array(labels), W)
            if done:
                rewards.append(episode_reward)
                if episode % report_interval == 0:
                    print_report(rewards, episode)
                break
    print_report(rewards, -1)

if __name__ == '__main__':
    main()

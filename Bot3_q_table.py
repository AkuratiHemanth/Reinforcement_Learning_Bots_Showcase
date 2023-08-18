import gym
import numpy as np
import random

random.seed(0)
np.random.seed(0)

def main():
    env = gym.make('FrozenLake-v1')

    num_episodes = 10000
    rewards = []
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(1, num_episodes + 1):
        state = env.reset()[0]
        episode_reward = 0
        
        while True:
            noise = np.random.random((1, env.action_space.n)) / (episode ** 2.)
            action = np.argmax(Q[state, :] + noise)
            state2, reward, done, _, info = env.step(action)
            episode_reward += reward
            Q[state, action] = Q[state, action] + 0.8 * (reward + 0.99 * np.max(Q[state2, :]) - Q[state, action])
            state = state2
            
            if done:
                rewards.append(episode_reward)
                break

    average_reward = sum(rewards) / num_episodes
    print(f'Average reward over {num_episodes} episodes: {average_reward:.2f}')

if __name__ == '__main__':
    main()

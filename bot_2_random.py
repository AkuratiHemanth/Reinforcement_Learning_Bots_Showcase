import gym
import random

random.seed(0)
num_episodes = 10

def main():
    env = gym.make('SpaceInvaders-v0')
    env.seed(0)

    rewards = []
    

    for _ in range(num_episodes):
        env.reset()
        episode_reward = 0

        while True:
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action) 
            episode_reward += reward

            if done:
                print('Reward: %s' % episode_reward)
                rewards.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(rewards)/len(rewards)))

if __name__ == '__main__':
    main()

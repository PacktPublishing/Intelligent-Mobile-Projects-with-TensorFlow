import gym
import numpy as np

env = gym.make("CartPole-v0")

total_rewards = []
for _ in range(1000):
	rewards = 0
	obs = env.reset()
	while True:
		action = 1 if obs[2] > 0 else 0
 		obs, reward, done, info = env.step(action) 
		rewards += reward
		if done:
			break
	total_rewards.append(rewards)

print(np.mean(total_rewards))

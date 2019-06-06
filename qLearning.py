import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

SMALLER_NUMBER = [20] * len(env.observation_space.high)
SMALLER_NUMBER_WINDOW = (env.observation_space.high - env.observation_space.low) / SMALLER_NUMBER

print(SMALLER_NUMBER_WINDOW)

q_table = np.random.uniform(low=-2, high=0, size=(SMALLER_NUMBER + [env.action_space.n]))

print(q_table.shape)
print(q_table)

done = False

while not done:
	action = 2
	new_state, reward, done, _ = env.step(action)
	print(new_state, reward)
	env.render()
	#env.step(action)

env.close()
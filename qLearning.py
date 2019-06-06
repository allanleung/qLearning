import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

UPDATES_EPISODE = 2000


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#print(DISCRETE_OS_WIN_SIZE)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
	## current state - enviroment / space size  
	# return numpy int 
	discrete_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
	return tuple(discrete_state.astype(np.int))


# return initial state 
discrete_state = get_discrete_state(env.reset())

print(discrete_state)


### Printing test
#index by tuple and starting Q value
print(q_table[discrete_state])
print(np.argmax(q_table[discrete_state]))

#print(q_table.shape)
#print(q_table)


## Iterate over many many times 

for episode in range (EPISODES):
	if episode % UPDATES_EPISODE == 0:
		render = True
	else:
		render = False
	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:
		action = np.argmax(q_table[discrete_state])
		new_state, reward, done, _ = env.step(action)
		#use for new formulation of q values
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()
		#print(new_state, reward)
		if not done: 
			# max future will get * by the discount, slowly over time future action get back into the q value
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]

			# Q Learning from wikipedia formula 
			# Back properation depends on DISCOUNT * max_future_q
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE + (reward + DISCOUNT * max_future_q)

			# Update the action 
			q_table[discrete_state+(action, )] = new_q

		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action, )] = 0


		discrete_state = new_discrete_state

env.close()


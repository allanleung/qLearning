import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000

UPDATES_EPISODE = 500
EPSILON = 0.5

START_EPS_DECAY = 1
# // so there is no float 
END_EPS_DECAY = EPISODES // 2

eps_decay_value = EPSILON / (END_EPS_DECAY - START_EPS_DECAY)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#print(DISCRETE_OS_WIN_SIZE)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []

aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

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
	episode_reward = 0
	if episode % UPDATES_EPISODE == 0:
		render = True
	else:
		render = False
	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:

		if np.random.random() > EPSILON:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		episode_reward += reward
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
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

			# Update the action 

			q_table[discrete_state+(action, )] = new_q

		elif new_state[0] >= env.goal_position:

			#print(f"We made it on try {episode}")
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state
	if END_EPS_DECAY >= episode >= START_EPS_DECAY:
		EPSILON -= eps_decay_value

	# total rewards at the end
	ep_rewards.append(episode_reward)

	# if epside % UPDATES_EPISODE == 0	
	if not episode % UPDATES_EPISODE:
		average_reward = sum(ep_rewards[-UPDATES_EPISODE:])/len(ep_rewards[-UPDATES_EPISODE:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-UPDATES_EPISODE:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-UPDATES_EPISODE:]))
		print(f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()
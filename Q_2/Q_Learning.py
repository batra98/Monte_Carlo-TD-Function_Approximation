import numpy as np
from collections import defaultdict

def test(env,agent,num_episodes = 1000):
	R = 0.0
	for episode in range(1,num_episodes+1):

	    if episode%100 == 0:
	        print("\rEpisode {}/{}".format(episode,num_episodes),end = "")
	    # R = 0.0
	    done = False
	    
	    state = env.reset()
	    action = np.argmax(agent.Q[state])

	    while not done:
	        next_state,reward,done,_ = env.step(action)
	        next_action = agent.policy(next_state)
	        # next_action = np.random.choice(np.arange(len(next_probs)),p = next_probs)
	        # next_action = np.argmax(agent.Q[next_state])
	# 			action = np.random.choice(np.arange(len(probs)),p = probs)
	        # print("State = {},Action = {},Q[s][a] = {}".format(state,action,agent.Q[state]))
	        
	        
	        R += reward
	        state = next_state
	        action = next_action
	        # env.render()
	    # print("-----------")

	
	return float(R)/num_episodes


ava_actions = [0,1,2,3]


class Q_Learning(object):

	def __init__(self,env,discount_factor = 0.9,alpha = 0.5,epsilon = 0.9):
		self.discount_factor = discount_factor
		self.alpha = alpha
		self.epsilon = epsilon
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = self.make_epsilon_greedy_policy(self.Q,self.epsilon,env.action_space.n)
		self.returns = []
		self.R = []

	def make_epsilon_greedy_policy(self,Q,epsilon,nA):

		def fn(state):
			e = np.random.random()

			if e < self.epsilon:
				A = np.random.choice(ava_actions)
			else:
				A = np.argmax(self.Q[state])

			return A

		return fn

	def learn(self,env,num_episodes = 10000):

		for episode in range(1,num_episodes+1):

			if (episode%1000) == 0:
				print("\rEpisode {}/{}".format(episode,num_episodes),end = "")

			state = env.reset()
			done = False

			self.R = []

			# self.epsilon = 0.1 + (0.9) * np.exp(-0.01 * episode)
			self.epsilon = 0.05 + 1*np.exp((-5*episode)/float(num_episodes))
			self.alpha = min(self.alpha,float(1000)/(episode))



			while not done:

				action = self.policy(state)
				# action = np.random.choice(np.arange(len(probs)),p = probs)

				next_state,reward,done,_ = env.step(action)

				self.R.append(reward)


				best_next_action = np.argmax(self.Q[next_state])
				self.Q[state][action] += self.alpha*(reward+self.discount_factor*(self.Q[next_state][best_next_action]) - self.Q[state][action])

				state = next_state

			if episode%10 == 0:

				# T = len(self.R)
				# G = 0

				# t = T-2

				# while t>=0:
				# 	G = self.R[t+1]+self.discount_factor*G
				# 	t = t-1

				# self.returns.append(G)
				self.returns.append(test(env,self,10))

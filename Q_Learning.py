import numpy as np
from collections import defaultdict

class Q_Learning(object):

	def __init__(self,env,discount_factor = 0.9,alpha = 0.81,epsilon = 0.9):
		self.discount_factor = discount_factor
		self.alpha = alpha
		self.epsilon = epsilon
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = self.make_epsilon_greedy_policy(self.Q,self.epsilon,env.action_space.n)
		self.returns = []
		self.R = []

	def make_epsilon_greedy_policy(self,Q,epsilon,nA):

		def fn(state):
			A = np.ones(nA,dtype = float)*epsilon/(nA)
			best_action = np.argmax(Q[state])
			A[best_action] += (1.0 - epsilon)
			return A

		return fn

	def learn(self,env,num_episodes = 10000):

		for episode in range(1,num_episodes+1):

			if (episode%1000) == 0:
				print("\rEpisode {}/{}".format(episode,num_episodes),end = "")

			state = env.reset()
			done = False

			self.R = []

			self.epsilon = 0.1 + (0.9) * np.exp(-0.01 * episode)


			while not done:

				probs = self.policy(state)
				action = np.random.choice(np.arange(len(probs)),p = probs)

				next_state,reward,done,_ = env.step(action)

				self.R.append(reward)


				best_next_action = np.argmax(self.Q[next_state])
				self.Q[state][action] += self.alpha*(reward+self.discount_factor*(self.Q[next_state][best_next_action]) - self.Q[state][action])

				state = next_state

			T = len(self.R)
			G = 0

			t = T-2

			while t>=0:
				G = self.R[t+1]+self.discount_factor*G
				t = t-1

			self.returns.append(G)

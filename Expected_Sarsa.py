import numpy as np
from collections import defaultdict

class Expected_Sarsa(object):

	def __init__(self,env,discount_factor = 0.9,alpha = 0.81,epsilon = 0.9):
		self.discount_factor = discount_factor
		self.alpha = alpha
		self.epsilon = epsilon
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = self.make_epsilon_greedy_policy(self.Q,self.epsilon,env.action_space.n)
		self.R = []
		self.returns = []


	def make_epsilon_greedy_policy(self,Q,epsilon,nA):

		def fn(state):
			A = np.ones(nA,dtype = float)*epsilon/(nA)
			best_action = np.argmax(Q[state])
			A[best_action] += (1.0 - epsilon)
			return A

		return fn


	def learn(self,env,num_episodes = 10000):

		for episode in range(1,num_episodes+1):

			if(episode%100) == 0:
				print("\rEpisode {}/{}".format(episode,num_episodes),end = "")

			state = env.reset()
			probs = self.policy(state)
			action = np.random.choice(np.arange(len(probs)),p = probs)


			self.epsilon = 0.1 + (0.9) * np.exp(-0.01 * episode)

			done = False

			self.R = []

			while not done:
				next_state,reward,done,_ = env.step(action)

				self.R.append(reward)
				next_probs = self.policy(next_state)
				next_action = np.random.choice(np.arange(len(next_probs)),p = next_probs)

				expected_Q = 0

				for i in range(env.action_space.n):
					expected_Q += next_probs[i]*self.Q[next_state][i]

				self.Q[state][action] += self.alpha*(reward + self.discount_factor*expected_Q - self.Q[state][action])

				action = next_action
				state = next_state

			T = len(self.R)
			G = 0

			t = T-2

			while t>=0:
				G = self.R[t+1]+self.discount_factor*G
				t = t-1

			self.returns.append(G)


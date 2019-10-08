import numpy as np
import sys
from collections import defaultdict
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD



class Mc_OnPolicy(object):
	def __init__(self,mark,epsilon,env,discount_factor):
		self.mark = mark
		self.epsilon = epsilon
		self.discount_factor = discount_factor
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = self.make_epsilon_greedy_policy(self.Q,self.epsilon,env.action_space.n)


	def make_epsilon_greedy_policy(self,Q,epsilon,nA):

		def fn(state,available_actions):
			assert len(available_actions) > 0
			e = np.random.random()
			if e < epsilon:
				A = np.random.choice(available_actions)
			else:
				if state[1] == 'X':
					best_action = available_actions[0]
					for actions in available_actions:
						if Q[state][actions] <= Q[state][best_action]:
							best_action = actions
				else:
					best_action = available_actions[0]
					for actions in available_actions:
						if Q[state][actions] >= Q[state][best_action]:
							best_action = actions
				A = best_action



			return A

		return fn


	def generate_episode(self,env,policy):
		episodes = []
		start_mark = 'O'
		env.set_start_mark(start_mark)
		state = env.reset()

		done = False
		iteration = 0

		while not done:
			available_actions = env.available_actions()
			action = policy(state,available_actions)
			nstate,reward,done,_ = env.step(action)

			episodes.append((state,action,reward))
			state = nstate
			iteration += 1

			start_mark = next_mark(start_mark)

		return episodes


	def learn(self,env,num_episodes):
		returns_sum = defaultdict(float)
		returns_count = defaultdict(float)

		for episode in range(1,num_episodes+1):
			if episode%1000 == 0:
				print("\rEpisode {}/{}.".format(episode,num_episodes),end="")
				sys.stdout.flush()

			episodes = self.generate_episode(env,self.policy)

			sa_in_episode = set([(tuple(x[0]),x[1]) for x in episodes])
		
			# print(Q)

			for state,action in sa_in_episode:
				sa_pair = (state,action)
				first_occurence_idx = next(i for i,x in enumerate(episodes) if x[0] == state and x[1] == action)
				# print(sa_pair)
				# print(first_occurence_idx)
				G = sum([x[2]*(self.discount_factor**i) for i,x in enumerate(episodes[first_occurence_idx:])])
				# print(G)
				returns_sum[sa_pair] += G
				returns_count[sa_pair] += 1.0
				self.Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

		return self.Q,self.policy



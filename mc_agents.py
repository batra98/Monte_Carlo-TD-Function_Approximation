import numpy as np
import sys
from collections import defaultdict
import td_agent
from tqdm import tqdm as _tqdm
tqdm = _tqdm
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD



class Mc_OnPolicy(object):
	def __init__(self,mark,epsilon,env,discount_factor):
		self.mark = mark
		self.epsilon = epsilon
		self.discount_factor = discount_factor
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = self.make_epsilon_greedy_policy(self.Q,self.epsilon,env.action_space.n)

	def best_val_indices(self,values,fn):
		best = fn(values)
		return [i for i, v in enumerate(values) if v == best]




	def make_epsilon_greedy_policy(self,Q,epsilon,nA):

		def fn(state,available_actions,bench = True):
			assert len(available_actions) > 0

			if bench == False:
				e = np.random.random()
			else:
				e = 1
			if e < epsilon:
				A = np.random.choice(available_actions)
			else:
				if state[1] == 'X':
					# best_action = available_actions[0]
					available_values = []
					for actions in available_actions:
						available_values.append(Q[state][actions])
					# 	if Q[state][actions] <= Q[state][best_action]:
					# 		best_action = actions
					indices = self.best_val_indices(available_values,min)
				else:
					available_values = []
					for actions in available_actions:
						available_values.append(Q[state][actions])
					# best_action = available_actions[0]
					# for actions in available_actions:
					# 	if Q[state][actions] >= Q[state][best_action]:
					# 		best_action = actions
					indices = self.best_val_indices(available_values,max)

				indices = np.array(indices)
				idx = np.random.choice(indices)
				A = available_actions[idx]



			return A

		return fn


	def generate_episode(self,env,policy,start_mark):
		episodes = []
		
		env.set_start_mark(start_mark)
		state = env.reset()

		done = False
		iteration = 0

		while not done:
			available_actions = env.available_actions()
			action = policy(state,available_actions,False)
			nstate,reward,done,_ = env.step(action)

			episodes.append((state,action,reward))
			state = nstate
			iteration += 1

			start_mark = next_mark(start_mark)

		return episodes


	def learn(self,env,num_episodes):
		returns_sum = defaultdict(float)
		returns_count = defaultdict(float)
		start_mark = 'X'


		for episode in range(1,num_episodes+1):
			if episode%1000 == 0:
				print("\rEpisode {}/{}.".format(episode,num_episodes),end="")
				sys.stdout.flush()


			episodes = self.generate_episode(env,self.policy,start_mark)


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

			start_mark = next_mark(start_mark)

			play_against_td(self,10)

		return self.Q,self.policy


	def act(self,state,available_actions):
		return self.policy(state,available_actions)


def play_against_td(agent_mc,max_episode = 10):
	td_agent.load_model(td_agent.MODEL_FILE)
	start_mark = 'O'

	env = TicTacToeEnv()
	env.set_start_mark(start_mark)
	agents = [agent_mc,td_agent.TDAgent('X',0,0)]

	episode = 0
	results = []

	for i in range(max_episode):
		env.set_start_mark(start_mark)
		state = env.reset()
		_,mark = state

		done = False

		while not done:
			agent = agent_by_mark(agents,mark)

			ava_actions = env.available_actions()

			# print(agent.mark)

			# if agent.mark == 'O':
				# print(agent.Q[state])

			action = agent.act(state,ava_actions)

			

			state,reward,done,_ = env.step(action)

			# env.render()

			if done:
				results.append(reward)
				break
			else:
				_,mark = state
		start_mark = next_mark(start_mark)
		episode += 1

	o_win = results.count(1)
	x_win = results.count(-1)
	draw = len(results) - o_win - x_win
	# print("O_WINS = {},X_WINS = {},DRAW = {}".format(o_win,x_win,draw))


	



import numpy as np
import sys
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm as _tqdm
tqdm = _tqdm
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD


class MC_OffPolicy_Weighted_Importance(object):
	def __init__(self,mark,env,discount_factor):
		self.mark = mark
		self.discount_factor = discount_factor
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.target_policy = self.greedy_policy(self.Q,env.action_space.n)
		self.behaviour_policy = self.random_policy(env.action_space.n)



	def random_policy(self,nA):

		def fn(state,available_actions):
			return np.random.choice(available_actions)
		return fn





	def best_val_indices(self,values,fn):
		best = fn(values)
		return [i for i, v in enumerate(values) if v == best]

	def greedy_policy(self,Q,nA):
		def fn(state,available_actions):
			assert len(available_actions) > 0

			available_values = []
			for actions in available_actions:
				available_values.append(Q[state][actions])

			if state[1] == 'X':
				indices = self.best_val_indices(available_values,min)
			else:
				indices = self.best_val_indices(available_values,max)

			indices = np.array(indices)
			idx = np.random.choice(indices)
			A = available_actions[idx]

			return A

		# def fn(state,available_actions):

		# 	if state[1] == 'X':
		# 		best_action = np.argmin(Q[state])
		# 	else:
		# 		best_action = np.argmax(Q[state])

		# 	return best_action

		return fn

	def generate_episode(self,env,policy,start_mark):
		episodes = []

		env.set_start_mark(start_mark)
		state = env.reset()

		done = False
		iteration = 0

		while not done:
			available_actions = env.available_actions()
			action = policy(state,available_actions)
			# action = np.random.choice(np.arange(len(prob)),p = prob)
			# action = int(action)
			nstate,reward,done,_ = env.step(action)

			episodes.append((state,action,reward))
			state = nstate
			iteration += 1

			start_mark = next_mark(start_mark)

		return episodes

	def learn(self,env,num_episodes):
		C = defaultdict(lambda: np.zeros(env.action_space.n))

		start_mark = 'X'


		for episode in range(1,num_episodes+1):
			if episode%1000 == 0:
				print("\rEpisode {}/{}".format(episode,num_episodes),end="")
				sys.stdout.flush()

			episodes = self.generate_episode(env,self.behaviour_policy,start_mark)

			G = 0.0
			W = 1.0

			t_initial = len(episodes)
			for t in range(len(episodes))[::-1]:
				state,action,reward = episodes[t]
				# print("State = {}, Action = {}, reward = {},W = {}".format(state,action,reward,W))
				G = self.discount_factor*G + reward
				C[state][action] += W

				# if (W/C[state][action]) != 1.0:
					# print(W/C[state][action])

				# print(self.Q[state])

				# if t_initial-t > 4:
					# print(t_initial-t)

				self.Q[state][action] += (W/C[state][action]) * (G - self.Q[state][action])



				# print(self.Q[state])


				x = np.nonzero(state[0])
				y = []



				for i in range(9):
					if i in x[0]:
						continue
					else:
						y.append(i)

				# print(state)
				# print(y)
				y = np.array(y) 

				# print(self.target_policy(state,y))

				if action != self.target_policy(state,y):
					break


				W = W*(len(y))

			start_mark = next_mark(start_mark)

		# print(self.Q)

	def act(self,state,available_actions):
		return self.target_policy(state,available_actions)






class Mc_OnPolicy(object):
	def __init__(self,mark,epsilon,env,discount_factor):
		self.mark = mark
		self.epsilon = epsilon
		self.discount_factor = discount_factor
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = self.make_epsilon_greedy_policy(self.Q,self.epsilon,env.action_space.n)
		self.backup = []

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


	def learn(self,env,num_episodes,agent_2):
		returns_sum = defaultdict(float)
		returns_count = defaultdict(float)
		unique_states = set()
		number_unique = []
		mean_returns = []
		low_returns = []
		high_returns = []


		start_mark = 'X'



		for episode in range(1,num_episodes+1):
			if episode%1000 == 0:
				print("\rEpisode {}/{}.".format(episode,num_episodes),end="")
				sys.stdout.flush()


			episodes = self.generate_episode(env,self.policy,start_mark)


			sa_in_episode = set([(tuple(x[0]),x[1]) for x in episodes])
		
			# print(Q)

			for x in episodes:
				unique_states.add(x[0])


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

			# print(self.Q)

			self.backup.append(deepcopy(self.Q[((0,0,0,0,0,0,0,0,0),'X')][4]))

			start_mark = next_mark(start_mark)

			# mu,low,high = play_against(self,agent_2,10)
			# number_unique.append(len(unique_states))
			# mean_returns.append(mu)
			# low_returns.append(low)
			# high_returns.append(high)


	def act(self,state,available_actions):
		return self.policy(state,available_actions)


def play_against(agent_mc,agent_2,max_episode = 10):
	start_mark = 'O'

	env = TicTacToeEnv()
	env.set_start_mark(start_mark)
	agents = [agent_mc,agent_2]

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

			if agent.mark == 'O':
				print(agent.Q[state])

			action = agent.act(state,ava_actions)

			

			state,reward,done,_ = env.step(action)

			env.render()

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
	print("O_WINS = {},X_WINS = {},DRAW = {}".format(o_win,x_win,draw))


	if o_win == 0 and x_win == 0:
		return float(o_win-x_win)/10.0,0,0


	if o_win > 0 and x_win > 0:
		return float(o_win-x_win)/10.0,-1,1
	else:
		if o_win == 0:
			if draw > 0:
				return float(o_win-x_win)/10.0,-1,0
			else:
				return float(o_win-x_win)/10.0,-1,-1
		else:
			if draw > 0:
				return float(o_win-x_win)/10.0,0,0
			else:
				return float(o_win-x_win)/10.0,1,1



	# print("O_WINS = {},X_WINS = {},DRAW = {}".format(o_win,x_win,draw))


	



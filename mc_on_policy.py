import numpy as np
import sys
from collections import defaultdict
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD



def make_epsilon_greedy_policy(Q,epsilon,nA):

	def fn(obs):
		non_zero = np.nonzero(obs)
		
		A = np.zeros(nA,dtype = float)
		
		# print(non_zero)

		for i in range(nA):
			if i not in non_zero[0]:
				A[i] = 1.0

		# print(A)
		A = A*epsilon/(nA-len(non_zero[0]))

		if len(non_zero[0]) != 0:
			best_action = non_zero[0][0]
			for i in range(nA):
				if i not in non_zero[0]:
					if Q[obs][i] >= Q[obs][best_action]:
						best_action = i
		else:

			best_action = np.argmax(Q[obs])
		A[best_action] += (1 - epsilon)
		# print(A)
		return A

	return fn 

def generate_episode(env,policy):
	episodes = []

	start_mark = 'O'
	env.set_start_mark(start_mark)
	state = env.reset()
	# _,mark = state


	done = False
	iteration = 0
	
	while not done:
		available_actions = env.available_actions()
		# print(available_actions)
		# env.show_turn(True,mark)
		# if iteration%2 == 0:
			# probability = policy(state[0])
			# print(state[0])
			# action = np.random.choice(np.arange(len(probability)),p = probability)
			

			
		# else:

		action = np.random.choice(available_actions)

		# probability = policy(state[0])
		# action = np.random.choice(np.arange(len(probability)),p = probability)
		# print(action)

		nstate,reward,done,_ = env.step(int(action))
		# print((state,action,reward))
		episodes.append((state[0],action,reward))
		state = nstate
		iteration += 1

		# _,mark = state
		start_mark = next_mark(start_mark)

	return episodes




def mc_control_on_policy(env,num_episodes,discount_factor = 0.1,epsilon = 0.1):
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	
	policy = make_epsilon_greedy_policy(Q,epsilon,env.action_space.n)

	# for actions in range(1,env.action_space.n+1):
	# 	print(policy(actions))

	for episode in range(1,num_episodes+1):
		if episode%1000 == 0:
			print("\rEpisode {}/{}.".format(episode,num_episodes),end="")
			sys.stdout.flush()

		episodes = generate_episode(env,policy)
		# print(episodes)
		sa_in_episode = set([(tuple(x[0]),x[1]) for x in episodes])
		
		# print(Q)

		for state,action in sa_in_episode:
			sa_pair = (state,action)
			first_occurence_idx = next(i for i,x in enumerate(episodes) if x[0] == state and x[1] == action)
			# print(sa_pair)
			# print(first_occurence_idx)
			G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episodes[first_occurence_idx:])])
			# print(G)
			returns_sum[sa_pair] += G
			returns_count[sa_pair] += 1.0
			Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
		
	return Q,policy






env = TicTacToeEnv(show_number = True)

Q,policy = mc_control_on_policy(env,num_episodes = 50000,epsilon = 0.1)











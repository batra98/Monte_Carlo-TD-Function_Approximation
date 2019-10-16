import Sarsa
import Q_Learning
import Expected_Sarsa
import frozen_lake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)


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
	        next_action = np.argmax(agent.Q[next_state])
	# 			action = np.random.choice(np.arange(len(probs)),p = probs)
	        # print("State = {},Action = {},Q[s][a] = {}".format(state,action,agent.Q[state]))
	        
	        
	        R += reward
	        state = next_state
	        action = next_action
	        # env.render()
	    # print("-----------")

	print("")
	print(R)





env = frozen_lake.FrozenLakeEnv(None,"4x4",True)
env.render()

sarsa_agent = Sarsa.Sarsa(env)
Q_Learning_agent = Q_Learning.Q_Learning(env)
Expected_Sarsa_agent = Expected_Sarsa.Expected_Sarsa(env)

sarsa_agent.learn(env,50000)
# Q_Learning_agent.learn(env,50000)
# Expected_Sarsa_agent.learn(env,50000)

print(sarsa_agent.Q)
# print(Q_Learning_agent.Q)
# print(Expected_Sarsa_agent.Q)
# print(sarsa_agent.epsilon)


# window_size = 100
# averaged_returns = np.zeros(len(sarsa_agent.returns)-window_size+1)
# max_returns = np.zeros(len(sarsa_agent.returns)-window_size+1)
# min_returns = np.zeros(len(sarsa_agent.returns)-window_size+1)



# for i in range(len(averaged_returns)):
#   averaged_returns[i] = np.mean(sarsa_agent.returns[i:i+window_size])
#   max_returns[i] = np.max(sarsa_agent.returns[i:i+window_size])
#   min_returns[i] = np.min(sarsa_agent.returns[i:i+window_size])

# plt.plot(averaged_returns)
# plt.ylabel("Moving average of first returns (window_size={})".format(window_size))
# plt.xlabel("Episode")

# print(sarsa_agent.returns)


# window_size = 100
# averaged_returns = np.zeros(len(Q_Learning_agent.returns)-window_size+1)
# max_returns = np.zeros(len(Q_Learning_agent.returns)-window_size+1)
# min_returns = np.zeros(len(Q_Learning_agent.returns)-window_size+1)



# for i in range(len(averaged_returns)):
#   averaged_returns[i] = np.mean(Q_Learning_agent.returns[i:i+window_size])
#   max_returns[i] = np.max(Q_Learning_agent.returns[i:i+window_size])
#   min_returns[i] = np.min(Q_Learning_agent.returns[i:i+window_size])

# plt.plot(averaged_returns)
# plt.xlabel("Episode")
# plt.ylabel("Moving average of first returns (window_size={})".format(window_size))
# # plot_mean_and_CI(averaged_returns,min_returns,max_returns,'g--','g')

# window_size = 100
# averaged_returns = np.zeros(len(Expected_Sarsa_agent.returns)-window_size+1)
# max_returns = np.zeros(len(Expected_Sarsa_agent.returns)-window_size+1)
# min_returns = np.zeros(len(Expected_Sarsa_agent.returns)-window_size+1)



# for i in range(len(averaged_returns)):
#   averaged_returns[i] = np.mean(Expected_Sarsa_agent.returns[i:i+window_size])
#   max_returns[i] = np.max(Expected_Sarsa_agent.returns[i:i+window_size])
#   min_returns[i] = np.min(Expected_Sarsa_agent.returns[i:i+window_size])

# plt.plot(averaged_returns)
# plt.xlabel("Episode")
# plt.ylabel("Moving average of first returns (window_size={})".format(window_size))
plt.show()

test(env,sarsa_agent,1000)
# test(env,Q_Learning_agent,1000)
# test(env,Expected_Sarsa_agent,1000)


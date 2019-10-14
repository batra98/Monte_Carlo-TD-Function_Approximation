import mc_agents
from collections import defaultdict
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD

import base_agent
import td_agent

import numpy as np
import matplotlib.pyplot as plt

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

def plotting(returns,window_size = 100):
    averaged_returns = np.zeros(len(returns)-window_size+1)
    max_returns = np.zeros(len(returns)-window_size+1)
    min_returns = np.zeros(len(returns)-window_size+1)
    
    
    for i in range(len(averaged_returns)):
      averaged_returns[i] = np.mean(returns[i:i+window_size])
      max_returns[i] = np.max(returns[i:i+window_size])
      min_returns[i] = np.min(returns[i:i+window_size])
    
#     plt.plot(averaged_returns)
    
#     plot_mean_and_CI(averaged_returns,min_returns,max_returns,'g--','g')
    
    return (averaged_returns,max_returns,min_returns)
rndm_state_action = [[((0,0,0,0,0,0,0,0,0),'X'),4],[((0,1,0,0,2,0,0,0,0),'O'),0],[((1,1,2,0,2,2,0,0,1),'X'),3],[((1,1,2,0,2,2,0,0,1),'O'),3]]

env = TicTacToeEnv(show_number = True)
mc_onpolicy = mc_agents.Mc_OnPolicy('O',0.1,env,0.9)
td_agent.load_model(td_agent.MODEL_FILE)
mu = mc_onpolicy.learn(env,50000,td_agent.TDAgent('X',0,0),rndm_state_action)

# fig = plt.figure(figsize = (10,5))
# smoothing_window = 100
# a = plotting(mu,smoothing_window)

# plot_mean_and_CI(a[0],a[1],a[2],'b--','b')
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward (Smoothed)")
# plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
# plt.show()


# mc_agents.load_model('Mc_OnPolicy_agent.dat',mc_onpolicy)
# print(mc_onpolicy.Q)

# mc_agents.play_against(mc_onpolicy,base_agent.BaseAgent('X'),3000,False)
# fig = plt.figure(figsize = (10,5))
# plt.plot(mc_onpolicy.unique_states,mu)
# plt.show()
print(mc_agents.play_against(mc_onpolicy,td_agent.TDAgent('X',0,0),3000,False))


def play(max_episode = 10):
    episode = 0
    bs = base_agent.BaseAgent('X')
    env = TicTacToeEnv()
    
    accuracy = 0
    
    while episode < max_episode:
        start_mark = 'O'
        env.set_start_mark(start_mark)
        state = env.reset()
        done = False
        
        iterat = 0
        while not done:
            ava_actions = env.available_actions()
#             print(state[0])
            if iterat%2 == 0:
                
                action = bs.act(state,ava_actions)
                state, reward, done, _ = env.step(action)
            else:
#                 probability = mc.policy(state[0])
                # print(Q[state])
#                 print(probability)
#                 action = np.random.choice(np.arange(len(probability)),p = probability)
                action = mc_offpolicy.act(state,ava_actions)
#                 print(action)
                state,reward,done,_ = env.step(int(action))
                accuracy += reward
                
            iterat += 1
            env.render()
            
            start_mark = next_mark(start_mark)

        
        episode += 1
        
    print(accuracy)

# play()

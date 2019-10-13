import mc_agents
import base_agent
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD

import td_agent
import matplotlib.pyplot as plt
import base_agent



def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

env = TicTacToeEnv(show_number = True)
td_agent.load_model(td_agent.MODEL_FILE)
mc_onpolicy = mc_agents.Mc_OnPolicy('O',0.1,env,1.0)

# mc_onpolicy.learn(env,5000,base_agent.BaseAgent('X'))
rndm_state_action = [[((0,0,0,0,0,0,0,0,0),'X'),4],[((0,1,0,0,2,0,0,0,0),'O'),0],[((0,0,1,0,2,0,0,0,0),'X'),8]]
mc_onpolicy.learn(env,10000,td_agent.TDAgent('X',0,0),rndm_state_action)

# print(mc_onpolicy.backup)

# rndm_state_action = [((0,1,0,0,2,0,0,0,0),'O'),0]
# rndm_state_action = [((0,0,1,0,2,0,0,0,0),'X'),8]


# Y = [items[rndm_state_action[0]][rndm_state_action[1]] for items in mc_onpolicy.backup]
for i in range(len(rndm_state_action)):
    X = [i for i in range(len(mc_onpolicy.backup))]
    Y = mc_onpolicy.backup[rndm_state_action[i]]
    print(np.var(np.array(Y)))
    plt.plot(X,Y)
    plt.show()

# print(mc_onpolicy.backup)



# fig2 = plt.figure(figsize=(10,5))
# plot_mean_and_CI(a,b,c,'k')
# a,b,c = mc_onpolicy.learn(env,10000,td_agent.TDAgent('X',0,0))

# a = np.array(a)
# b = np.array(b)
# c = np.array(c)
# a1 = np.array(a1)
# b1 = np.array(b1)
# c1 = np.array(c1)

# # print(a.shape)
# # print(b.shape)


# # plt.plot(a,b)
# plot_mean_and_CI(a,b,c,'g--','g')
# plot_mean_and_CI(a1,b1,c1,'b--','b')

# plt.show()





def play(max_episode = 10):
    episode = 0
    bs = base_agent.BaseAgent('O')
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
            if iterat%2 == 1:
                
                action = bs.act(state,ava_actions)
                state, reward, done, _ = env.step(action)
            else:
#                 probability = mc.policy(state[0])
                print(Q[state])
#                 print(probability)
#                 action = np.random.choice(np.arange(len(probability)),p = probability)
                action = policy(state,ava_actions)
#                 print(action)
                state,reward,done,_ = env.step(int(action))
                accuracy += reward
                
            iterat += 1
            env.render()
            
            start_mark = next_mark(start_mark)

        
        episode += 1
        
    print(accuracy)


# play(10)

# for state in Q.keys():
# 	print("{}, {}".format(state,Q[state]))











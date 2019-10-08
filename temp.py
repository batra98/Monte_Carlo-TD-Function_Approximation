import mc_agents
import base_agent
import random
import numpy as np
from collections import defaultdict
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD

import td_agent

env = TicTacToeEnv(show_number = True)

mc_onpolicy = mc_agents.Mc_OnPolicy('O',0.1,env,0.1)
Q,policy = mc_onpolicy.learn(env,50000)
print(len(Q.keys()))

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











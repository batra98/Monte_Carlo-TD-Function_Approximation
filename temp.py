import mc_agents
import base_agent
import random
import numpy as np
from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD


env = TicTacToeEnv(show_number = True)

mc_onpolicy = mc_agents.Mc_OnPolicy('O',0.1,env,0.1)
Q,policy = mc_onpolicy.learn(env,500000)

def play(max_episode = 10):
    episode = 0
    bs = base_agent.BaseAgent('O')
    env = TicTacToeEnv()
    
    accuracy = 0
    
    while episode < max_episode:
        start_mark = 'O'
        env.set_start_mark('O')
        state = env.reset()
        done = False
        
        iterat = 0
        while not done:
            ava_actions = env.available_actions()
#             print(state[0])
            if iterat%2 == 1:
                
                action = bs.act(state,ava_actions)
                state, reward, done, info = env.step(action)
            else:
#                 probability = mc.policy(state[0])
                # print(Q[state[0]])
#                 print(probability)
#                 action = np.random.choice(np.arange(len(probability)),p = probability)
                action = policy(state[0],ava_actions)
#                 print(action)
                state,reward,done,_ = env.step(int(action))
            iterat += 1
            # env.render()
            if reward == 1:
            	accuracy += reward
            start_mark = next_mark(start_mark)
        episode += 1
        
    print(accuracy)

play(10000)








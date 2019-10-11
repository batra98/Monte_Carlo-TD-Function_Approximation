import Sarsa
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


env = frozen_lake.FrozenLakeEnv(None,"8x8",True)
env.render()

sarsa_agent = Sarsa.Sarsa(env)

sarsa_agent.learn(env,10000)

print(sarsa_agent.Q)
print(sarsa_agent.epsilon)


window_size = 100
averaged_returns = np.zeros(len(sarsa_agent.returns)-window_size+1)
max_returns = np.zeros(len(sarsa_agent.returns)-window_size+1)
min_returns = np.zeros(len(sarsa_agent.returns)-window_size+1)



for i in range(len(averaged_returns)):
  averaged_returns[i] = np.mean(sarsa_agent.returns[i:i+window_size])
  max_returns[i] = np.max(sarsa_agent.returns[i:i+window_size])
  min_returns[i] = np.min(sarsa_agent.returns[i:i+window_size])

plt.plot(averaged_returns)
plt.xlabel("Episode")
plt.ylabel("Moving average of first returns (window_size={})".format(window_size))
# plot_mean_and_CI(averaged_returns,min_returns,max_returns,'g--','g')
plt.show()
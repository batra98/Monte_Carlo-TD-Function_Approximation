# Monte-Carlo, TD Methods and Functional Approximation

## Introduction
In this assignment, we will use Monte-Carlo (MC) Methods and Temporal Difference (TD) Learning on couple of games and toy problems.
The problems as given below:
1. Train an agent that plays the **Tic-Tac-Toe using Monte-Carlo Methods**.
2. Train an agent that generates the optimal policy through **TD-Methods in the Frozen-Lake Environment**.
3. Build **a Deep Q-Learning Network (DQN) which can play Atari Breakout** and get the best scores. I was not able to implement this component of the assignment, so instead I build **a DQN which can play the cart-pole game**.

Details of the problems are included in the respective folders.


## :file_folder: File Structure
```bash
.
├── Q_1
│   ├── Mc_OffPolicy_agent.dat
│   ├── Mc_OnPolicy_agent
│   ├── Monte-Carlo_Methods(3).html
│   ├── Monte-Carlo_Methods.ipynb
│   ├── __pycache__
│   ├── base_agent.py
│   ├── best_td_agent.dat
│   ├── gym-tictactoe
│   ├── human_agent.py
│   ├── mc_agents.py
│   └── td_agent.py
├── Q_2
│   ├── Expected_Sarsa.py
│   ├── Frozen_Lake_Through_TD_Methods.html
│   ├── Frozen_Lake_Through_TD_Methods.ipynb
│   ├── Q_Learning.py
│   ├── Sarsa.py
│   ├── __pycache__
│   └── frozen_lake.py
├── Q_3
│   ├── DQN_Agent.py
│   ├── Function_Approximation_DQN.html
│   ├── Function_Approximation_DQN.ipynb
│   ├── __pycache__
│   └── cartpole-dqn.h5
├── README.md
└── assignment.pdf

7 directories, 21 files
```

- **Q_\*** - Contains files for respective problems along with trained models.
- **assignment.pdf** - contains the all the problems statements of the assignment.

## Future Work
At the time of doing the assignment, I did't have sufficient knowledge of DL to implement the last part of the assignment.
I would like to complete this part of the assignment now.
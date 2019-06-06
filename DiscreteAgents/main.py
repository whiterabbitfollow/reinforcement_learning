# -*- coding: utf-8 -*-
import gym
import pandas as pd 
from utils import create_trajectory_from_history
import matplotlib.pyplot as plt 
import numpy as np
import copy
from MonteCarloAgents import MonteCarloFirstVisitAgent, MonteCarloEveryVisitAgent
from TDAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, ValueIterationAgent
from mpl_toolkits.axes_grid1 import make_axes_locatable

run_stochastic_env = True

env = gym.make("CliffWalking-v0")

if run_stochastic_env:

    for s in env.P:

        all_actions = []

        for a in env.P[s]:
            all_actions.append(env.P[s][a][0])

        for a in env.P[s]:

            env.P[s][a] = list(all_actions)

            for aa in env.P[s]:

                p, s_nxt, r, is_done = env.P[s][a][aa]

                if a == aa:
                    env.P[s][a][aa] = (0.85, s_nxt, r, is_done)
                else:
                    env.P[s][a][aa] = (0.05, s_nxt, r, is_done)

data = dict()

agent_types = [QLearningAgent,
               SARSAAgent,
               ExpectedSARSAAgent,
               MonteCarloFirstVisitAgent,
               MonteCarloEveryVisitAgent,
               ValueIterationAgent
               ]

for agent_class in agent_types:

    agent = agent_class(env)

    learning_curve_training, learning_curve_eval, delta_history, has_agent_converged = agent.train(delta_thr=-1, max_no_iterations = 20000, verbosity=2)

    history, r = agent.evaluate(render=False)

    data[str(agent)] = {"lc_training":copy.deepcopy(learning_curve_training),
         "lc_eval":copy.deepcopy(learning_curve_eval),
         "converged":has_agent_converged,
         "eval_hist":copy.deepcopy(history),
         "no_steps":agent.no_learning_steps,
         "agent":copy.deepcopy(agent),
         "delta_hist":copy.deepcopy(delta_history)
         }


# In[]

plt.figure(1)
plt.clf()

for agent_str in data:
    if not agent_str.startswith("Value"):
        plt.plot(pd.Series(data[agent_str]["lc_training"]).ewm(alpha=0.01).mean().values,label=agent_str)

plt.title("Accumulated reward [-]")
plt.xlabel("Number episodes [-]")
plt.ylabel("")
plt.xlim(0,1000)
plt.ylim(-4000,0)
plt.legend(loc="best")
plt.show()



fig, ax = plt.subplots(1,2,figsize=(10,5))
ax_no = 0

for agent_str in data:

    delta = data[agent_str]["delta_hist"]
    x = []
    y = []

    if "Monte" in agent_str:
        ax_no = 1
    else:
        ax_no = 0

    for step_no, delta_step in delta:
        x.append(step_no)
        y.append(delta_step)

    df = pd.DataFrame({"delta":y,"index":x}).set_index("index")
    ax[ax_no].plot(df.ewm(alpha=0.1).mean(),label=agent_str)


ax[0].set_xlabel("Nr environment steps [-]")
ax[1].set_xlabel("Nr environment steps [-]")

ax[0].set_ylabel("TD error [-]")
ax[1].set_ylabel("TD error [-]")

ax[0].legend(loc="best")
ax[1].legend(loc="best")
ax[0].set_ylim(0,50)
ax[1].set_ylim(0,1000)
ax[0].set_xlim(0,50000)
ax[1].set_xlim(0,50000)

plt.suptitle("Convergence properties of different agents")
plt.show()

fig, axs = plt.subplots(6,1)
N_COLS = 1

for cell_no, agent_str in enumerate(data):

    agent = data[agent_str]["agent"]
    history = data[agent_str]["eval_hist"]

    row = cell_no//N_COLS
    col = cell_no-row*N_COLS

    V = agent.get_state_value_function()
    grid = np.array(V).reshape(4,12)
    rows, cols = create_trajectory_from_history(history)

#    im = axs[row][col].imshow(grid)
#    axs[row][col].plot(cols,rows,c="black",label="greedy policy trajectory")
#    axs[row][col].set_title(str(agent))
#    divider = make_axes_locatable(axs[row][col])

    im = axs[cell_no].imshow(grid)
    axs[cell_no].plot(cols,rows,c="black",label="greedy policy trajectory")
    axs[cell_no].set_title(str(agent))

    axs[cell_no].set_xticklabels([])
    axs[cell_no].set_yticklabels([])

    divider = make_axes_locatable(axs[cell_no])

    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
#    axs[row][col].legend()
#
#plt.suptitle("Path taken with greedy policy")

plt.show()

import warnings ; warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook as tqdm
import gym, gym_walk, gym_aima
from itertools import cycle
from pprint import pprint
import numpy as np
import random

import policies as pol
import policyUtils as pu

'''
SETTINGS
'''
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)

'''
GLOBALS
'''
LEFT, RIGHT = range(2)

###############################
########### MAIN    ###########

# pi = lambda s: {0:LEFT, 1:RIGHT, 2:LEFT}[s]
# P = pol.banditWalk
# pu.print_policy(pi, pol.banditWalk, action_symbols=("<",">"), n_cols=3)

# V = pu.policy_evaluation(pi, P)
# print(V)

env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6

LEFT, RIGHT = range(2)
pi = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
pu.print_policy(pi, P, action_symbols=('<', '>'), n_cols=7)
V = pu.policy_evaluation(pi, P)
pu.print_state_value_function(V, P, n_cols=7)
pu.print_action_value_function(


successProbPerGoal = np.zeros(len(P), dtype=np.float64)
for i in range(len(P)):
    successProbPerGoal = pu.probability_success(env, pi, i)

pprint(successProbPerGoal)

#   itterate on the agent
for i in range(0, 1):
    V = pu.policy_evaluation(pi, P)
    pi = pu.policy_improvement(V, P)

pu.print_policy(pi, P, action_symbols=('<', '>'), n_cols=7)
V = pu.policy_evaluation(pi, P)
pu.print_state_value_function(V, P, n_cols=7)

successProbPerGoal = np.zeros(len(P), dtype=np.float64)
for i in range(len(P)):
    successProbPerGoal = pu.probability_success(env, pi, i)

pprint(successProbPerGoal)

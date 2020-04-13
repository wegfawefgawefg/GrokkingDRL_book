import warnings ; warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook as tqdm
import gym, gym_walk, gym_aima
from itertools import cycle
from pprint import pprint
import numpy as np
import random

import policies as pol
import policyUtils as pu
import termBar as tb

'''
tackle the 8 by 8 frozen lake
tackle gridworld
'''

#######################################
#   settings
#######################################
np.set_printoptions(suppress=True)
# random.seed(123); np.random.seed(123)

#######################################
#   helpers
#######################################
def speculatePolicy(pi, P, env, goalState):
    pu.print_policy(pi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
    V = pu.policy_evaluation(pi, P)
    pu.print_state_value_function(V, P)
    
    successProbability = pu.probability_success(env, pi, goalState) * 100.0
    meanReturn = pu.mean_return(env, pi)
    print("Success Probability: " + str(successProbability))
    print("Mean Value Return: " + str(meanReturn))

def watchPolicyIterate(pi, P, env, goalState, numIters=2):
    V = pu.policy_evaluation(pi, P)
    for i in range(0, numIters):
        print()
        print("Iter: " + str(i))
        pi = pu.policy_improvement(V, P)
        pu.print_policy(pi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
        V = pu.policy_evaluation(pi, P)
        pu.print_state_value_function(V, P)

        successProbability = pu.probability_success(env, pi, goalState) * 100.0
        meanReturn = pu.mean_return(env, pi)
        print("Success Probability: " + str(successProbability))
        print("Mean Value Return: " + str(meanReturn))

    return pi


#######################################
#   iterate on random policy
#######################################
tb.printHeader("Frozen Lake 8x8")

#   init env/ goals etc
env = gym.make('FrozenLake8x8-v0')
P = env.env.P
initState = env.reset()
goalState = 8 * 8 - 1
LEFT, DOWN, RIGHT, UP = range(4)

#   make random policy
tb.printSubHeader("Random Policy")
randomPi = pu.genRandomPolicy(len(P), 4)

#   #   before
tb.printSubSubHeader("Before")
speculatePolicy(randomPi, P, env, goalState)

#   #   iterate
tb.printSubSubHeader("Iteration")
randomPi = watchPolicyIterate(randomPi, P, env, goalState, numIters=10)

#   #   after
tb.printSubSubHeader("After")
speculatePolicy(randomPi, P, env, goalState)
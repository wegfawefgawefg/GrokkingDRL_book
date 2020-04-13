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

def value_iteration(P, gamma=0.99, theta = 1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, new_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[new_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = lambda s : {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi

#######################################
#   
#######################################
# reproduce Russell & Norvig
tb.printHeader("RussellNorvigGridworld")
env = gym.make('RussellNorvigGridworld-v0')
V, pi = value_iteration(env.env.P, gamma=1.0)
pu.print_policy(pi, env.env.P)
pu.print_state_value_function(V, env.env.P)

# reproduce Abbeel & Klein
tb.printHeader("AbbeelKleinGridworld")
env = gym.make('AbbeelKleinGridworld-v0')
V, pi = value_iteration(env.env.P, gamma=1.0)
pu.print_policy(pi, env.env.P)
pu.print_state_value_function(V, env.env.P)

# solve frozen lake 8*8 with value iteration
#######################################
tb.printHeader("Frozen Lake 8x8")
env = gym.make('FrozenLake8x8-v0')
V, pi = value_iteration(env.env.P, gamma=1.0)
pu.print_policy(pi, env.env.P)
pu.print_state_value_function(V, env.env.P)

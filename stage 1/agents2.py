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
SETTINGS
'''
np.set_printoptions(suppress=True)
# random.seed(123); np.random.seed(123)

'''
print out larger frozen lake world
start a random policy
evaluate that policy
evaluate probability of success
itterate on the policy
evaluate that new policy
evaluate new probability of success

good job
'''
#######################################
#   policy creation and evaluation
#######################################
tb.printHeader("Policy Creation and Evaluation")

#   init env/ goals etc
env = gym.make('FrozenLake-v0')
P = env.env.P
initState = env.reset()
goalState = 15

LEFT, DOWN, RIGHT, UP = range(4)

#   make random policy
tb.printSubHeader("Random Policy")
randomPi = pu.genRandomPolicy(len(P), 4)
pu.print_policy(randomPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
successProbability = pu.probability_success(env, randomPi, goalState)
meanReturn = pu.mean_return(env, randomPi)

print(successProbability * 100.0)
print(meanReturn)

#   make human reasoned solution policy
tb.printSubHeader("Greedy Human Policy")
goGetPi = lambda s: {
    0:RIGHT, 1:RIGHT, 2:DOWN, 3:LEFT,
    4:DOWN, 5:LEFT, 6:DOWN, 7:LEFT,
    8:RIGHT, 9:RIGHT, 10:DOWN, 11:LEFT,
    12:LEFT, 13:RIGHT, 14:RIGHT, 15:LEFT
}[s]
pu.print_policy(goGetPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
successProbability = pu.probability_success(env, goGetPi, goalState)
meanReturn = pu.mean_return(env, goGetPi)

print(successProbability * 100.0)
print(meanReturn)

#######################################
#   policy iteration
#######################################
tb.printHeader("Policy Iteration")

#   lets improve the random policy
tb.printSubHeader("Improve Random Policy")

#   #   before
tb.printSubSubHeader("Before")
randomPi = pu.genRandomPolicy(len(P), 4)
pu.print_policy(randomPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
successProbability = pu.probability_success(env, randomPi, goalState)
meanReturn = pu.mean_return(env, randomPi)
print(successProbability * 100.0)
print(meanReturn)

#   #   iterate
for i in range(0, 3):
    V = pu.policy_evaluation(randomPi, P)
    randomPi = pu.policy_improvement(V, P)
    pu.print_policy(randomPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
    successProbability = pu.probability_success(env, randomPi, goalState)
    meanReturn = pu.mean_return(env, randomPi)
    print(successProbability * 100.0)
    print(meanReturn)

#   #   after
tb.printSubSubHeader("After")
pu.print_policy(randomPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
successProbability = pu.probability_success(env, randomPi, goalState)
meanReturn = pu.mean_return(env, randomPi)
print(successProbability * 100.0)
print(meanReturn)

#   lets improve the human reasoned policy
tb.printSubHeader("Improve Human Greedy Policy")

#   #   before
tb.printSubSubHeader("Before")
pu.print_policy(goGetPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
successProbability = pu.probability_success(env, goGetPi, goalState)
meanReturn = pu.mean_return(env, goGetPi)
print(successProbability * 100.0)
print(meanReturn)

#   #   iterate
for i in range(0, 3):
    V = pu.policy_evaluation(goGetPi, P)
    goGetPi = pu.policy_improvement(V, P)
    pu.print_policy(goGetPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
    successProbability = pu.probability_success(env, goGetPi, goalState)
    meanReturn = pu.mean_return(env, goGetPi)
    print(successProbability * 100.0)
    print(meanReturn)

#   #   after
tb.printSubSubHeader("After")
pu.print_policy(goGetPi, P, action_symbols=("<", 'v', '>', '^'), n_cols=4)
successProbability = pu.probability_success(env, goGetPi, goalState)
meanReturn = pu.mean_return(env, goGetPi)
print(successProbability * 100.0)
print(meanReturn)
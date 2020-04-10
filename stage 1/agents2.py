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
speculatePolicy(randomPi, P, env, goalState)

#   make human reasoned solution policy
tb.printSubHeader("Greedy Human Policy")
goGetPi = lambda s: {
    0:RIGHT, 1:RIGHT, 2:DOWN, 3:LEFT,
    4:DOWN, 5:LEFT, 6:DOWN, 7:LEFT,
    8:RIGHT, 9:RIGHT, 10:DOWN, 11:LEFT,
    12:LEFT, 13:RIGHT, 14:RIGHT, 15:LEFT
}[s]
speculatePolicy(goGetPi, P, env, goalState)

#######################################
#   policy iteration
#######################################
tb.printHeader("Policy Iteration")

#   lets improve the random policy
tb.printSubHeader("Improve Random Policy")

#   #   before
tb.printSubSubHeader("Before")
randomPi = pu.genRandomPolicy(len(P), 4)
speculatePolicy(randomPi, P, env, goalState)

#   #   iterate
tb.printSubSubHeader("Iteration")
randomPi = watchPolicyIterate(randomPi, P, env, goalState, numIters=2)

#   #   after
tb.printSubSubHeader("After")
speculatePolicy(randomPi, P, env, goalState)


#   lets improve the human reasoned policy
tb.printSubHeader("Improve Human Greedy Policy")

#   #   before
tb.printSubSubHeader("Before")
speculatePolicy(goGetPi, P, env, goalState)

#   #   iterate
tb.printSubSubHeader("Iteration")
goGetPi = watchPolicyIterate(goGetPi, P, env, goalState, numIters=2)

#   #   after
tb.printSubSubHeader("After")
speculatePolicy(goGetPi, P, env, goalState)
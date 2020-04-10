import numpy as np
import random
from itertools import cycle
import itertools

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    '''
    given an action policy, estimate the reward.
    has to be run multiple times because the reward kind of propogates to nearby cells
    so you probably have to run it atleast the "area" of the grid number of times
    so at firsst your RIGHT might seem like it has no reward,
        but then reward propogates from righter rights back to the squares your right considers
            then your RIGHT reward goes up 
    '''
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                value = prob * (reward + gamma * prev_V[next_state] * (not done))
                # print((s, value, prob, next_state, reward, done))
                V[s] += value
        if np.max(np.abs(prev_V - V)) < theta:
            break
        # print(V)
        prev_V = V.copy()
    return V
    
def policy_improvement(V, P, gamma=1.0):
    '''
    picks the max action for each state in the game
    '''
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    '''
    replays the game with the given policy a bunch of times
    stops whenever it hits an end state or the max number of steps
    stochastically approximates success probability
    '''
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)


def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    '''
    prints the value of each state,
        remember its a combination of all the actions per state
            (from the earlier propogation)
    '''
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def print_action_value_function(Q, 
                                optimal_Q=None, 
                                action_symbols=('<', '>'), 
                                prec=3, 
                                title='Action-value function:'):
    '''
    prints out the value of specific actions for each state
    '''
    vf_types=('',) if optimal_Q is None else ('', '*', 'err')
    headers = ['s',] + [' '.join(i) for i in list(itertools.product(vf_types, action_symbols))]
    print(title)
    states = np.arange(len(Q))[..., np.newaxis]
    arr = np.hstack((states, np.round(Q, prec)))
    if not (optimal_Q is None):
        arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q-Q, prec)))
    print(tabulate(arr, headers, tablefmt="fancy_grid"))

def genRandomPolicy(numStates, numActions):
    stateActions = {}
    for i in range(numStates):
        stateActions[i] = random.randint(0, numActions - 1)
    pi = lambda s: stateActions[s]
    return pi

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
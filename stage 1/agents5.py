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
should i do all of these?
or just 1 or 2 and then move on to the next chapter

#   i think ucb and thompson sampling are worth it
#   #   kinda already get the other ones
calculate regret in onne of these or something.
try optimistic initialization
try espilon decay
    -   exponential forms
    -   linear forms
    -   sin forms?
lol pure exploration
try softmax
try ucb
try thompson  sampling
    -   useful normal distribution information here to explore

compare the strategies
10 armed bandit
'''

#######################################
#   settings
#######################################
np.set_printoptions(suppress=True)
# random.seed(123); np.random.seed(123)

#######################################
#   helpers
#######################################


#######################################
#   main
#######################################

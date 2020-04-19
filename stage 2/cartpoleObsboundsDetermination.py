import gym
from gym import spaces
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import architectures as arc
import random


env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

cartPosMin = math.inf
cartPosMax = -math.inf
cartVelMin = math.inf
cartVelMax = -math.inf
poleAngleMin = math.inf
poleAngleMax = -math.inf
poleTipVelMin = math.inf
poleTipVelMax = -math.inf

for i in range(1000):
    obs = env.reset()
    for i in range(1000):
        obs, _, _, _ = env.step(random.sample([0, 1], 1)[0])
        cartPos, cartVel, poleAngle, poleTipVel = obs
        # print(cartPos, cartVel, poleAngle, poleTipVel)

        cartPosMin = min(cartPosMin, cartPos)
        cartPosMax = max(cartPosMax, cartPos)

        cartVelMin = min(cartVelMin, cartVel)
        cartVelMax = max(cartVelMax, cartVel)

        poleAngleMin = min(poleAngleMin, poleAngle)
        poleAngleMax = max(poleAngleMax, poleAngle)

        poleTipVelMin = min(poleTipVelMin, poleTipVel)
        poleTipVelMax = max(poleTipVelMax, poleTipVel)
env.close()

print("cartPosMin: " + str(cartPosMin))
print("cartPosMax: " + str(cartPosMax))
print("cartVelMin: " + str(cartVelMin))
print("cartVelMax: " + str(cartVelMax))
print("poleAngleMin: " + str(poleAngleMin))
print("poleAngleMax: " + str(poleAngleMax))
print("poleTipVelMin: " + str(poleTipVelMin))
print("poleTipVelMax: " + str(poleTipVelMax))
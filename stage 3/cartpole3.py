import gym
from gym import spaces
import random
from collections import namedtuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pprint import pprint
import architectures as arc
from collections import deque, namedtuple
import numpy as np

'''
gpu accelerate by batching the random sampling shit
'''

############################################################
####    HELPERS
############################################################
def computeReward(frame, done):
    absObs = frame.abs()
    punishment = -absObs.sum()
    value = 0.5 * punishment 
    return value

############################################################
####    SETTINGS
############################################################
GAMMA = 0.95

# MEMORY_SIZE = 1000000
MEMORY_SIZE = 7000
# BATCH_SIZE = 10
BATCH_SIZE = 64
# BATCH_SIZE = 256

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
# EPSILON_DECAY = 0.998

NUM_TRAINING_EPSISODES = 300000
MemoryFrame = namedtuple('MemoryFrame', 
    ['obsFrame', 'action', 'reward', 'nextObsFrame', 'done'])

############################################################
####    MAIN
############################################################
#   setup net
net = arc.FlexNet(4, 2, (24,))
# net = arc.FlexNet(4, 2, (64,64, 64))
# net = arc.TwoNet(4, 2, 24)
# net = arc.TwoNet(4, 2, 8)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
memoryBank = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_MAX

#   make env
env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

#   play infinite games
net.train()
recordReward, recordTimesteps, episode = 0, 0, 0
while True:
    obs = env.reset()
    obsFrame = torch.tensor(obs, dtype=torch.float32).view(-1, 4)

    #   run the game
    #   #   break out and start another on fail
    done = False
    frameNum, totalReward = 0, 0
    while True:
        frameNum += 1

        #   watching slows down games
        #   #   so dont watch until after it practiced a bit
        # env.render()
        if episode > NUM_TRAINING_EPSISODES:
            env.render()

        #   pick next action
        if np.random.rand() < epsilon:   #   EXPLORE: take random action
            action = random.randint(0, 1)
        else:   #   EXPLOIT: net chooses action
            output = net(obsFrame).clone().detach()
            values, index = output.max(0)
            action = int(index)

        #   tic game
        nextObs, reward, done, info = env.step(action)
        nextObsFrame = torch.tensor(nextObs, dtype=torch.float32).view(-1, 4)
        # reward = computeReward(nextObsFrame, done)
        reward = reward if not done else -reward
        totalReward += reward

        #   add memory frame
        memoryFrame = MemoryFrame(obsFrame, action, reward, nextObsFrame, done)
        memoryBank.append(memoryFrame)

        obsFrame = nextObsFrame

        if done:
            break

            # 'obsFrame', 'action', 'reward', 'nextObsFrame', 'done'])


        #   "EXPERIENCE REPLAY": have PTSD Flashbacks
        #   #   its good to remember some of the good times in life
        #   #   (also a lot of bad ones)
        if len(memoryBank) >= BATCH_SIZE:
            memoryBatch = random.sample(memoryBank, BATCH_SIZE)
            memoryBatch = MemoryFrame(*zip(*memoryBatch))

            memObsFrames = torch.stack(memoryBatch.obsFrame)
            memActions = torch.tensor(memoryBatch.action, dtype=torch.float32)
            memRewards = torch.tensor(memoryBatch.reward, dtype=torch.float32)
            memNextObsFrames = torch.stack(memoryBatch.nextObsFrame)
            memDones = torch.tensor(memoryBatch.done)

            futureOutputs = net(memNextObsFrames).clone().detach()
            futureActionValues, futureActions = futureOutputs.max(2)
            futureActions.int()
            futureActionValues = futureActionValues.view(BATCH_SIZE)
            #   mask out future aspect of rewards if sim is already Done though
            #   #   memDones.logical_not()
            qUpdate = memRewards + (GAMMA * futureActionValues.view(BATCH_SIZE)) * (memDones.logical_not())

            output = net(memObsFrames)
            newQValues = output.clone().detach().view(BATCH_SIZE, 2)
            futureActions = futureActions.view(BATCH_SIZE, 1)
            qUpdate = qUpdate.view(BATCH_SIZE, 1)
            print(newQValues.shape)
            print(futureActions.shape)
            newQValues = newQValues.scatter_(2, futureActions, qUpdate)
            quit() 

            loss = F.mse_loss(output, newQValues.detach(), reduction="mean")
            # print(loss)
            net.zero_grad()
            loss.backward()
            optimizer.step()

            epsilon *= EPSILON_DECAY
            epsilon = max(EPSILON_MIN, epsilon)
            
    #   post episode logic
    episode += 1
    if frameNum+1 > recordTimesteps:
        recordTimesteps = frameNum+1
    if totalReward > recordReward:
        recordReward = totalReward
    print("ep {} recRew {} recTs {} lastRew {} lastTs {} epsilon {}, memorySize {}".format(
        episode, recordReward, recordTimesteps, 
        totalReward, frameNum+1, 
        str(epsilon)[:4],
        len(memoryBank),
        ))
    
env.close()
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
import termBar as tbar
import datetime 
import math
from time import sleep

'''
TODO:
*   add offline mode
o   make it support other environments
try convolutional version
try one that works on pixels
add checkpoints to prevent forgetting
make snake environment
'''

'''
IDEAS:
let the network set epsilon somehow?
give it a memory of more than one frame
    right now we are relying on experience replay to be the memory, but its input could also include recent frames
        this way it can choose how important each of the distances into the past are
'''

############################################################
####    HELPERS
############################################################
def computeReward(frame, done):
    rFrame = frame[0][:6]
    absObs = rFrame.abs()
    punishment = -absObs.sum()
    value = 1.0 * punishment 
    return float(value)

# def computeReward(frame, done):
#     absObs = frame.abs()
#     punishment = -absObs.sum()
#     value = 0.5 * punishment 
#     return value

############################################################
####    SETTINGS
############################################################
ENV_NAME = "LunarLander-v2"
GAMMA = 0.95

# MEMORY_SIZE = 1000000
# MEMORY_SIZE = 7000
MEMORY_SIZE = 35000
BATCH_SIZE = 128
# MEMORY_SIZE = BATCH_SIZE

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
# EPSILON_DECAY = 0.995
# EPSILON_DECAY = 0.998
EPSILON_DECAY = 0.998


NUM_TRAINING_EPSISODES = 300
MemoryFrame = namedtuple('MemoryFrame', 
    ['obsFrame', 'action', 'reward', 'nextObsFrame', 'done'])

DONE_TRAINING_SCORE = 150
CONSECUTIVE_WIN_REQUIREMENTS = 2

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
############################################################
####    SETUP
############################################################
#    setup env
env = gym.make(ENV_NAME)
NUM_ACTIONS = env.action_space.n
NUM_OBS = env.observation_space.shape[0]
print("Num Actions: " + str(NUM_ACTIONS))
print("Num Obs: " + str(NUM_OBS))

#   setup net
net = arc.FlexNet(NUM_OBS, NUM_ACTIONS, (256,256))
# net = arc.TwoNet(NUM_OBS, NUM_ACTIONS, 128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
memoryBank = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_MAX
lastNScores = deque(maxlen=CONSECUTIVE_WIN_REQUIREMENTS)


############################################################
####    TRAIN
############################################################
tbar.printHeader("TRAIN MODE")
net.train()
recordReward, recordTimesteps, episode = -math.inf, math.inf, 0
#   play game until max score reached CONSECUTIVE_WIN_REQUIREMENTS times in a row
finishedTraining = False
while not finishedTraining:
    obs = env.reset()
    obsFrame = torch.tensor(obs, dtype=torch.float32).view(-1, NUM_OBS).to(device)
    #   run the game
    #   #   break out and start another on fail
    done = False
    frameNum, totalReward = 0, 0
    while True:
        env.render()
        # stepCompute_time_start = datetime.datetime.now()
        frameNum += 1

        #   pick next action
        if np.random.rand() < epsilon:   #   EXPLORE: take random action
            action = random.randint(0, NUM_ACTIONS-1)
        else:   #   EXPLOIT: net chooses action
            output = net(obsFrame).clone().detach()
            values, index = output.max(1)
            action = int(index)

        #   tic game
        nextObs, reward, done, info = env.step(action)
        nextObsFrame = torch.tensor(nextObs, dtype=torch.float32).view(-1, NUM_OBS).to(device)
        # complexReward = computeReward(nextObsFrame, done)
        # reward = reward if not done else -reward
        # print(str(reward) + " " + str(complexReward))
        # reward = complexReward + reward
        # reward = complexReward
        totalReward += reward

        #   add memory frame
        memoryFrame = MemoryFrame(obsFrame, action, reward, nextObsFrame, done)
        memoryBank.append(memoryFrame)

        obsFrame = nextObsFrame

        if done:
            break

        #   experience replay
        if len(memoryBank) >= BATCH_SIZE:
            memoryBatch = random.sample(memoryBank, BATCH_SIZE)
            # memoryBatch = list(memoryBank)[-BATCH_SIZE:]
            memoryBatch = MemoryFrame(*zip(*memoryBatch))

            memObsFrames =      torch.stack(    memoryBatch.obsFrame)
            memActions =        torch.tensor(   memoryBatch.action, dtype=torch.long)
            memRewards =        torch.tensor(   memoryBatch.reward, dtype=torch.float32)
            memNextObsFrames =  torch.stack(    memoryBatch.nextObsFrame)
            # memDones = torch.tensor(memoryBatch.done)

            futureOutputs = net(memNextObsFrames).clone().detach()
            futureActionValues, futureActions = futureOutputs.max(2)
            futureActionValues = futureActionValues.view(BATCH_SIZE)
            # qUpdate = memRewards + (GAMMA * futureActionValues.view(BATCH_SIZE)) * (memDones.logical_not())
            qUpdate = memRewards + GAMMA * futureActionValues.view(BATCH_SIZE)

            output = net(memObsFrames)
            newQValues = output.clone().detach().view(BATCH_SIZE, NUM_ACTIONS)
            memActions = memActions.view(BATCH_SIZE, 1)
            qUpdate = qUpdate.view(BATCH_SIZE, 1)
            newQValues = newQValues.scatter_(1, memActions, qUpdate).view(BATCH_SIZE, 1, NUM_ACTIONS)

            loss = F.mse_loss(output, newQValues, reduction="mean")
            net.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon *= EPSILON_DECAY
        epsilon = max(EPSILON_MIN, epsilon)

        # stepCompute_time_end = datetime.datetime.now()
        # stepComputeTime = stepCompute_time_end - stepCompute_time_start
        # print(stepComputeTime)
            
    #   post episode logic
    episode += 1
    lastNScores.append(totalReward)
    finishedTraining = all(score >= DONE_TRAINING_SCORE for score in lastNScores)
    if frameNum+1 < recordTimesteps:
        recordTimesteps = frameNum+1
    if totalReward > recordReward:
        recordReward = totalReward
    print("ep {} recRew {} recTs {} lastRew {} lastTs {} epsilon {}, memorySize {}".format(
        episode, 
        str(recordReward)[:6], recordTimesteps, 
        str(totalReward)[:6],
        frameNum+1, 
        str(epsilon)[:5],
        len(memoryBank),
        ))

############################################################
####    SAVE MODEL
############################################################



############################################################
####    EVAL
############################################################
tbar.printHeader("EVAL MODE")
net.eval()
recordReward, recordTimesteps, episode = -math.inf, math.inf, 0
while True:
    obs = env.reset()
    obsFrame = torch.tensor(obs, dtype=torch.float32).view(-1, NUM_OBS).to(device)

    #   run the game
    #   #   break out and start another on fail
    done = False
    frameNum, totalReward = 0, 0
    while True:
        frameNum += 1
        env.render()

        output = net(obsFrame).clone().detach()
        values, index = output.max(1)
        action = int(index)

        #   tic game
        nextObs, reward, done, info = env.step(action)
        nextObsFrame = torch.tensor(nextObs, dtype=torch.float32).view(-1, NUM_OBS).to(device)
        # reward = computeReward(nextObsFrame, done)
        reward = reward if not done else -reward
        totalReward += reward

        obsFrame = nextObsFrame

        if done:
            break

    #   post episode logic
    episode += 1
    if frameNum+1 < recordTimesteps:
        recordTimesteps = frameNum+1
    if totalReward > recordReward:
        recordReward = totalReward
    print("ep {} recRew {} recTs {} lastRew {} lastTs {} epsilon {}, memorySize {}".format(
        episode, 
        str(recordReward)[:6], recordTimesteps, 
        str(totalReward)[:6],
        frameNum+1, 
        str(epsilon)[:4],
        len(memoryBank),
        ))
env.close()


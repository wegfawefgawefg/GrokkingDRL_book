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

###############################################################################
#   !!!!! WARNING !!!!!
#   THIS VERSION IS SLOW AS ASS
#   IT EXISTS ONLY AS A DEMONSTRATION OF SEQUENTIALLY LEARNING EACH EXPERIENCE
#       CONVERGENCE IS FASTER, SO HIGHER SAMPLE EFFICIENCY
#       BUT IT RUNS LIKE 100-1000 TIMES SLOWER AND THAT SCALES WITH BATCH SIZE
#       SO ITS NOT WORTH IT
################################################################################

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
# BATCH_SIZE = 20
BATCH_SIZE = 64

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
# EPSILON_DECAY = 0.998

NUM_TRAINING_EPSISODES = 300
MemoryFrame = namedtuple('MemoryFrame', 
    ['obsFrame', 'action', 'reward', 'nextObsFrame', 'done'])

DONE_TRAINING_SCORE = 499
CONSECUTIVE_WIN_REQUIREMENTS = 5

############################################################
####    SETUP
############################################################
#   setup net
# net = arc.FlexNet(4, 2, (24,))
# net = arc.FlexNet(4, 2, (64,64, 64))
net = arc.TwoNet(4, 2, 24)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
memoryBank = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_MAX
lastNScores = deque(maxlen=CONSECUTIVE_WIN_REQUIREMENTS)

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

############################################################
####    TRAIN
############################################################
tbar.printHeader("TRAIN MODE")
net.train()
recordReward, recordTimesteps, episode = 0, 0, 0
#   play game until max score reached CONSECUTIVE_WIN_REQUIREMENTS times in a row
finishedTraining = False
while not finishedTraining:
    obs = env.reset()
    obsFrame = torch.tensor(obs, dtype=torch.float32).view(-1, 4)

    #   run the game
    #   #   break out and start another on fail
    done = False
    frameNum, totalReward = 0, 0
    while True:
        frameNum += 1

        #   pick next action
        if np.random.rand() < epsilon:   #   EXPLORE: take random action
            action = random.randint(0, 1)
        else:   #   EXPLOIT: net chooses action
            output = net(obsFrame).clone().detach()
            values, index = output.max(1)
            action = int(index)

        #   tic game
        nextObs, reward, done, info = env.step(action)
        nextObsFrame = torch.tensor(nextObs, dtype=torch.float32).view(-1, 4)
        complexReward = computeReward(nextObsFrame, done)
        reward = reward if not done else -reward
        reward = complexReward + reward
        totalReward += reward

        #   add memory frame
        memoryFrame = MemoryFrame(obsFrame, action, reward, nextObsFrame, done)
        memoryBank.append(memoryFrame)

        obsFrame = nextObsFrame

        if done:
            break

        #   "EXPERIENCE REPLAY": have PTSD Flashbacks
        #   #   its good to remember some of the good times in life
        #   #   (also a lot of bad ones)
        if len(memoryBank) >= BATCH_SIZE:
            memoryBatch = random.sample(memoryBank, BATCH_SIZE)
            for memory in memoryBatch:
                q_update = memory.reward
                if not memory.done:
                    futureOutput = net(memory.nextObsFrame).clone().detach()
                    values, index = futureOutput.max(1)
                    action = int(index)
                    futureActionValue = float(values[0])
                    #   'now reward' should include 'future reward'
                    #   #   iteratively distil the effects of the future
                    #   #   into our understanding of the effects of decisions now
                    q_update = memory.reward + GAMMA * futureActionValue
            
                output = net(memory.obsFrame)
                newQValues = output.clone().detach()
                # print("newQValues:")
                # print(newQValues)
                newQValues[0][memory.action] = q_update
                # print(newQValues)
                loss = criterion(output, newQValues)
                # print(loss)
                net.zero_grad()
                loss.backward()
                optimizer.step()

            epsilon *= EPSILON_DECAY
            epsilon = max(EPSILON_MIN, epsilon)
            
    #   post episode logic
    episode += 1
    lastNScores.append(frameNum)
    finishedTraining = all(score >= DONE_TRAINING_SCORE for score in lastNScores)
    if frameNum+1 > recordTimesteps:
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

############################################################
####    EVAL
############################################################
tbar.printHeader("EVAL MODE")
net.eval()
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
        env.render()

        output = net(obsFrame).clone().detach()
        values, index = output.max(1)
        action = int(index)

        #   tic game
        nextObs, reward, done, info = env.step(action)
        nextObsFrame = torch.tensor(nextObs, dtype=torch.float32).view(-1, 4)
        # reward = computeReward(nextObsFrame, done)
        reward = reward if not done else -reward
        totalReward += reward

        obsFrame = nextObsFrame

        if done:
            break

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


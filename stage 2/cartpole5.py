import gym
from gym import spaces
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import architectures as arc

'''
further ideas
go back and make the linear image color per pixel thing
prisoners dilema tournament with ai agents

'''

'''
make a pid controller
try limiting the response rate of the pid controller
why is there no noop action?

arbitrarily multiply the components togethor in every combination
    multiply by a constant, then output that

neural network
    how simple can it be and still work?
    any reason to do a convnet?
    can we store past frames? (yes)
    seems cool

try one of those decision tables
or like a genetic algorithm
'''

'''
simple solutions
----------------
2 frame memory
k frame memory

#   trying this now
2 frame memory including action
k frame memory including action

#   also prediction
ALSO,  make your outputs add up to 1


add next frame prediction
add past frame prediction
'''

'''
complicated solutions
q value estimator
    -estimate reward given current state, past actions and next action
        -do so for both actions
        -pick the one that is higher
        -punish the reward estimator, nothing else


make it output a sequence of actions
    punishment for sequence inconsistency across frames

make a big brain semi AGI solution

'''

'''
to reduce noise you have to learn batches
    but somehow accumulate rewards for the batches..
    and give context for the input?
'''

############################################################
####    SETTINGS
############################################################
numMemoryFrames = 4

############################################################
####    MAIN
############################################################
net = arc.TinyNet(6 * numMemoryFrames, 256, 2)
# net = arc.MedNet(6 * numMemoryFrames, 128, 2)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.0001)

net.train()

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

epsilon = 1.0
epsilonMin = 0.01
epsilonDecay = 0.993

recordReward = 0
recordTimesteps = 0
episode = 0

while True:
    #   pre first frame logic
    obs = env.reset()

    #   #   build memory
    obsTensor = torch.tensor(obs, dtype=torch.float32)
    firstAction = torch.tensor([0.0, 0.0], dtype=torch.float32) #   NO-OP, lol
    memory = []
    for i in range(numMemoryFrames):
        memoryFrame = torch.cat((obsTensor, firstAction))
        memory.append(memoryFrame.clone())

    #   #   get first action
    netData = torch.cat(memory)
    output = net(netData)
    value, index = output.max(0)
    action = int(index)

    done = False
    t = 0
    totalReward = 0
    framesPastDone = 0
    while not done or (framesPastDone < 50):
        if done:
            framesPastDone += 1
        else:
            t += 1

        # env.render()
        if episode > 900:
            env.render()

        obs, reward, done, info = env.step(action)
        totalReward += reward

        #   punish
        absobs = [abs(x) for x in obs]
        punishment = -sum(absobs)
        #   #   normalize punishment
        # cartPos = obs[0]    #   -4.8
        # poleAngle = obs[2]


        # punishment = -abs(obs[2])
        antiReward = -(1 - reward)
        value = 1.0 * punishment + 0.0 * antiReward
        # value = 3.0 * value
        value = 0.1 * value

        #   make target
        if random.random() < epsilon:
            target = torch.tensor([1, 1], dtype=torch.float32)  #   incentivise exploration
        else:
            target = output.clone() #   incentivise nothing
            # action = random.choice([0, 1])
        # target *= epsilon
        # target = torch.tensor([0, 0], dtype=torch.float32)  #   disincentivise
        # target[action] = output[action] + value
        # target[action] = 1 + value
        # target[action] += value
        target[action] = output[action] + value
        # print(net.l1.weight)

        #   update weights
        loss = criterion(output, target)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        
        #   rebuild memory
        obsTensor = torch.tensor(obs, dtype=torch.float32)
        actionTensor = torch.tensor([0, 0], dtype=torch.float32)
        actionTensor[action] = 1
        memoryFrame = torch.cat((obsTensor, actionTensor))
        memory.pop(0)
        memory.append(memoryFrame.clone())

        #   net choice
        netData = torch.cat(memory)
        output = net(netData)
        value, index = output.max(0)
        action = int(index)

        

    #   post episode logic
    episode += 1
    
    if t+1 > recordTimesteps:
        recordTimesteps = t+1
    if totalReward > recordReward:
        recordReward = totalReward
    # else:
    #     print("restore")
    #     net = copy.deepcopy(bestNet)
    print("ep {} recRew {} recTs {} lastRew {} lastTs {} e {}".format(
        episode, recordReward, recordTimesteps, totalReward, t+1, str(epsilon)[:4]))
    
    epsilon *= epsilonDecay
    if epsilon < epsilonMin:
        epsilon = epsilonMin
    # print(epsilon)

env.close()
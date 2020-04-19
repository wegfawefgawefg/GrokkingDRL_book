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
#   expandable memory
*   2 frame memory
*   k frame memory
#   #   with action
*   2 frame memory including action
*   k frame memory including action

#   add batches
#   #   verify batches are diverse
#   #   cleverly samplep batches
#   #   #   randomly
#   #   #   punish batch episodes based on time since now

#   also prediction
*   normalize the error (cheated by making it 1 or 0 so equiv to reward)

#   prediction to unsupervise the learning
add next frame prediction
add past frame prediction

#   numpyify batch frame creation because its fuckin slo
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
THOUGHTS:
should batch rewards be accumulated? instead of updated all at once?
or is the criterion already doing that

should i have equal number of 1 and 0 rewards in a batch?
should i ensure equal numbers of either action in a batch?
can i update with ann entire minibatch instead of one at a time
'''
############################################################
####    SETTINGS
############################################################
numMemoryFrames = 3
batchSize = 256
miniBatchSize = 1
numTrainingEpisodes = 1000
# maxFramesPastDone = 30
BatchFrame = namedtuple('BatchFrame', ['memory', 'output', 'action', 'envResponse'])
EnvResponse = namedtuple('EnvResponse', ['obs', 'reward', 'done', 'info'])

############################################################
####    MAIN
############################################################
net = arc.TinyNet(6 * numMemoryFrames, 1024, 2)
# net = arc.MedNet(6 * numMemoryFrames, 128, 2)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.01)

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

epsilon = 1.0
epsilonMin = 0.0001
epsilonDecay = 0.9

recordReward = 0
recordTimesteps = 0
episode = 0

batch = []
while True:
    ############################################################
    ####    Build Batch
    ############################################################
    net.eval()
    obs = env.reset()

    #   build memory
    obsTensor = torch.tensor(obs, dtype=torch.float32)
    firstAction = torch.tensor([0.0, 0.0], dtype=torch.float32) #   NO-OP, lol
    memory = []
    for i in range(numMemoryFrames):
        memoryFrame = torch.cat((obsTensor, firstAction))
        memory.append(memoryFrame.clone())

    #   net choice
    memoryFrame = torch.cat(memory)
    output = net(memoryFrame).detach()
    value, index = output.max(0)
    action = int(index)

    done = False
    frameNum, totalReward, framesPastDone = 0, 0, 0
    framesAlive = 0
    while (not done or (framesPastDone < (framesAlive * 3)) ) and len(batch) < batchSize:
        if done:
            framesAlive = frameNum
            framesPastDone += 1
        else:   #   didnt fail episode yet
            frameNum += 1

        # env.render()
        if episode > numTrainingEpisodes:
            env.render()

        obs, reward, done, info = env.step(action)
        totalReward += reward
        
        #   build a batch frame
        #   #   memory, output, response, reward
        batchFrame = BatchFrame(
            memory=memoryFrame.clone(),
            output=output.detach().clone(),
            action=action,
            envResponse=EnvResponse(obs, reward, done, info))
        batch.append(batchFrame)

        #   rebuild memory
        obsTensor = torch.tensor(obs, dtype=torch.float32)
        actionTensor = torch.tensor([0, 0], dtype=torch.float32)
        actionTensor[action] = 1
        memoryFrame = torch.cat((obsTensor, actionTensor))
        memory.pop(0)
        memory.append(memoryFrame.clone())
        
        #   net choice
        memoryFrame = torch.cat(memory)
        output = net(memoryFrame).detach()
        value, index = output.max(0)
        action = int(index)

        if random.random() < epsilon:
            action = random.randint(0, 1)

    #   post episode logic
    episode += 1
    if frameNum+1 > recordTimesteps:
        recordTimesteps = frameNum+1
    if totalReward > recordReward:
        recordReward = totalReward
    print("ep {} recRew {} recTs {} lastRew {} lastTs {} batchLen {} eps {}".format(
        episode, recordReward, recordTimesteps, 
        totalReward, frameNum+1, 
        len(batch),
        str(epsilon)[:4],
        ))
    
    ############################################################
    ####    Update Policy from Batch
    ############################################################
    if len(batch) == batchSize:
        print("Updating Policy")
        #   epsilon stuff
        epsilon *= epsilonDecay
        if epsilon < epsilonMin:
            epsilon = epsilonMin

        # someFrames = batch[:miniBatchSize]
        someFrames = random.sample(batch, miniBatchSize)
        actions = [someFrame.action for someFrame in someFrames]
        print(actions)

        #   create x
        memoryFrames = [frame.memory for frame in someFrames]
        # pprint(memoryFrames)
        # x = torch.stack(memoryFrames)

        #   create y
        targets = []
        for frame in someFrames:
            target = frame.output.clone()
            target[frame.action] = frame.envResponse.reward
            targets.append(target)
        # y = torch.stack(targets)

        #   update weights
        net.train()
        for i, xy in enumerate(zip(memoryFrames, targets)):
            x, y = xy
            net.zero_grad()
            output = net(x)
            target = output.clone()
            target[frame.action] = frame.envResponse.reward
            targets.append(target)
            if i == 0:
                print("assertion")
                print(output)
                print(y)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        batch = []


env.close()
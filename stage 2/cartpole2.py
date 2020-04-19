import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import architectures as arc

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

net = arc.TinyNet(4, 64, 2)
net = arc.MedNet(4, 256, 2)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.train()

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

epsilon = 1.0
epsilonDecay = 0.999

record = 0
episode = 0

while True:
    obs = env.reset()

    netData = torch.tensor(obs, dtype=torch.float32)
    output = net(netData)
    value, index = output.max(0)
    action = int(index)

    done = False
    t = 0
    while not done:
    # for t in range(50):
        # env.render()
        obs, reward, done, info = env.step(action)
        # print(obs)
        
        #   punish
        absobs = [abs(x) for x in obs]
        punishment = -sum(absobs)
        antiReward = -(1 - reward)
        value = 1.0 * punishment + 1.0 * antiReward
        # value = 0.8 * value
        value = 3.0 * value

        # print(output)

        #   make target
        # target = torch.ones(2, dtype=torch.float32)  #   incentivise exploration
        target = torch.tensor([1, 1], dtype=torch.float32)  #   incentivise exploration
        # target *= epsilon
        # target = torch.tensor([0, 0], dtype=torch.float32)  #   disincentivise
        # target = output.clone() #   incentivise nothing
        # target[action] = output[action] + value
        # target[action] = 1 + value
        target[action] += value

        #   update weights
        loss = criterion(output, target)
        net.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        #   net choice
        netData = torch.tensor(obs, dtype=torch.float32)
        output = net(netData)
        value, index = output.max(0)
        action = int(index)

        t += 1

    #   post episode logic
    episode += 1
    
    if t+1 > record:
        record = t+1
    print("ep {} record {} last {} e {}".format(episode, record, t+1, str(epsilon)[:4]))
    epsilon *= epsilonDecay
    # print(epsilon)

env.close()
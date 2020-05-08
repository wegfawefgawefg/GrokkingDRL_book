import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

'''
try with only one layer not two
'''

class ReplayBuffer():
    def __init__(self, maxSize, stateShape):
        self.memSize = maxSize
        self.memCount = 0
        self.stateMemory = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.actionMemory = np.zeros(self.memSize, dtype=np.int64)
        self.rewardMemory = np.zeros(self.memSize, dtype=np.float32)
        self.nextStateMemory = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.doneMemory = np.zeros(self.memSize, dtype=np.bool)

    def storeMemory(self, state, action, reward, nextState, done):
        memIndex = self.memCount % self.memSize 
        self.stateMemory[memIndex] = state
        self.actionMemory[memIndex] = action
        self.rewardMemory[memIndex] = reward
        self.nextStateMemory[memIndex] = nextState
        self.doneMemory[memIndex] = done

        self.memCount += 1

    def sample(self, sampleSize):
        memMax = min(self.memCount, self.memSize)
        batchIndecies = np.random.choice(memMax, sampleSize, replace=False)

        states = self.stateMemory[batchIndecies]
        actions = self.actionMemory[batchIndecies]
        rewards = self.rewardMemory[batchIndecies]
        nextStates = self.nextStateMemory[batchIndecies]
        dones = self.doneMemory[batchIndecies]

        return states, actions, rewards, nextStates, dones

class ConvActionAdvantageNetwork(nn.Module):
    def __init__(self, name, lr, inputShape, outputSize, checkpointDir):
        super().__init__()
        self.name = name
        self.checkpointDir = os.path.join(checkpointDir, name)

        self.inputshape = inputShape
        self.outputShape = outputSize

        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fc1 = nn.Linear(128*23*16, 256)
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, outputSize)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        #   incoming image is 210 by 160,
        #   #this should be a stack of frames though
        observation = torch.tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 210, 160)
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*23*16)
        x = F.relu(self.fc1(x))
        value = self.value(x)
        advantage = self.advantage(x)
        return value, advantage

    def save(self):
        print("... saving ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        print("... loading ...")
        self.load_state_dict(torch.load(self.checkpoint_file))

class DuelingQDiscreteAgent():
    def __init__(self, lr, inputShape, numActions, batchSize, 
            epsilon=1.0, gamma=0.99, layer1Size=256, 
            layer2Size=256, maxMemSize=10000, epsMin=0.01, epsDecay=5e-4, 
            replaceTargetCount = 1000, checkpointDir=".\\"):
        self.lr = lr
        self.epsilon = epsilon
        self.epsMin = epsMin
        self.epsDecay = epsDecay
        self.gamma = gamma
        self.batchSize = batchSize
        self.actionSpace = list(range(numActions))
        self.checkpointDir = checkpointDir

        self.learnStepCounter = 0
        self.replaceTargetCount = replaceTargetCount
        
        self.memory = ReplayBuffer(maxMemSize, inputShape)
        self.eval = ConvActionAdvantageNetwork(
            "eval", lr, inputShape, numActions, checkpointDir)
        self.next = ConvActionAdvantageNetwork(
            "next", lr, inputShape, numActions, checkpointDir)

    def replaceTargetNetwork(self):
        if self.learnStepCounter % self.replaceTargetCount == 0:
            self.next.load_state_dict(self.eval.state_dict())

    def chooseAction(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float32).to(self.eval.device)
            _, advantage = self.eval(state)
            action = torch.argmax(advantage).item()
            return action
        else:
            return np.random.choice(self.actionSpace)

    def storeMemory(self, state, action, reward, nextState, done):
        self.memory.storeMemory(state, action, reward, nextState, done)

    def learn(self):
        if self.memory.memCount < self.batchSize:
            return

        self.eval.optimizer.zero_grad()
        self.replaceTargetNetwork()
    
        stateBatch, actionBatch, rewardBatch, nextStateBatch, doneBatch = \
            self.memory.sample(self.batchSize)
        stateBatch = torch.tensor(stateBatch).to(self.eval.device)
        actionBatch = torch.tensor(actionBatch).to(self.eval.device)
        rewardBatch = torch.tensor(rewardBatch).to(self.eval.device)
        nextStateBatch = torch.tensor(nextStateBatch).to(self.eval.device)
        doneBatch = torch.tensor(doneBatch).to(self.eval.device)
        
        batchIndex = np.arange(self.batchSize, dtype=np.int64)

        vBatch, aBatch = self.eval(stateBatch)
        nextVBatchEval, nextABatchEval = self.eval(nextStateBatch)
        nextVBatch, nextABatch = self.next(nextStateBatch) 

        advantageDif = aBatch - aBatch.mean(dim=1, keepdim=True)
        actionQs = torch.add(vBatch, advantageDif)[batchIndex, actionBatch]

        nextAdvantageDif = nextABatch - nextABatch.mean(dim=1, keepdim=True)
        nextActionQs = torch.add(nextVBatch, nextAdvantageDif)

        #   get the action indecies from the eval network, not from the next network
        #   #   So the VALUE model is stable, but the action CHOICE is nearsighted
        evalNextAdvDif = nextABatchEval - nextABatchEval.mean(dim=1, keepdim=True)
        evalNextActionQs = torch.add(nextVBatchEval, evalNextAdvDif)
        evalMaxActionIndecies = torch.argmax(evalNextActionQs, dim=1)
        nextActionQs = nextActionQs[batchIndex, evalMaxActionIndecies]
        
        nextActionQs[doneBatch] = 0.0
        qTarget = rewardBatch + self.gamma * nextActionQs

        loss = self.eval.loss(qTarget, actionQs).to(self.eval.device)
        loss.backward()
        self.eval.optimizer.step()
        
        self.learnStepCounter += 1

        if self.epsilon > self.epsMin:
            self.epsilon -= self.epsDecay

def stackFrames(frameBuffer, stackSize):
        inputDims = frameBuffer[0].shape
        frameMagnitude = np.arange(stackSize) / stackSize + 1 / stackSize
        frameMagnitude = frameMagnitude
        frameStack = np.stack(reversed(list(frameBuffer)))
        frameStack = frameStack * frameMagnitude[:, np.newaxis, np.newaxis]
        frameStack = frameStack.sum(axis=0)
        return frameStack

if __name__ == '__main__':
    import gym
    import math
    from matplotlib import pyplot as plt
    from collections import deque
    from PIL import Image

    env = gym.make("SpaceInvaders-v0")
    # env = gym.make("Pong-v0")
    numActions = env.action_space.n
    inputDims = (210, 160)

    agent = DuelingQDiscreteAgent(
        lr=0.00001, inputShape=inputDims, numActions=numActions, batchSize=32, 
        epsilon=1.0, gamma=0.99, layer1Size=256, layer2Size=256, maxMemSize=2048, 
        epsMin=0.01, epsDecay=5e-4)

    scoreHistory = []
    numEpisodes = 2000
    numTrainingEpisodes = 0
    highScore = -math.inf
    recordTimeSteps = math.inf
    numStackedFrames = 12
    obsBuffer = deque(maxlen=numStackedFrames)
    for episode in range(numEpisodes):
        done = False
        observation = env.reset()

        #   prepare obsBuffer
        observation = np.mean(observation, axis=2)  #   remove rgb
        for i in range(0, numStackedFrames-1):
            obsBuffer.append(np.zeros_like(observation))
        obsBuffer.append(observation)
        frameStack = stackFrames(obsBuffer, numStackedFrames)

        score, frame = 0, 1
        while not done:
            if episode > numTrainingEpisodes:
                env.render()

            # if frame == 60:
            #     img = Image.fromarray(np.uint8(frameStack * 255) , 'L')
            #     img.show()

            action = agent.chooseAction(frameStack)
            nextObservation, reward, done, info = env.step(action)
            nextObservation = np.mean(nextObservation, axis=2)  #   remove rgb
            obsBuffer.append(nextObservation)
            nextFrameStack = stackFrames(obsBuffer, numStackedFrames)
            if done:
                reward -= 100
            agent.storeMemory(frameStack, action, reward, nextFrameStack, done)
            agent.learn()

            observation = nextObservation
            frameStack = nextFrameStack

            score += reward
            frame += 1
        scoreHistory.append(score)

        recordTimeSteps = min(recordTimeSteps, frame)
        highScore = max(highScore, score)
        print(( "ep {}: high-score {:12.3f}, shortest-time {:d}, "
                "score {:12.3f}, last-episode-time {:4d}, epsilon {:6.3f}").format(
            episode, 
            highScore, 
            recordTimeSteps, 
            score,
            frame,
            agent.epsilon
            ))

    fig = plt.figure()
    meanWindow = 10
    meanedScoreHistory = np.convolve(scoreHistory, np.ones(meanWindow), 'valid') / meanWindow
    plt.plot(np.arange(0, numEpisodes-1, 1.0), meanedScoreHistory)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.title("Training Scores")
    plt.show()
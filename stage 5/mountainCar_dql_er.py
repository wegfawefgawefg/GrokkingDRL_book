import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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

class DeepQNetwork(nn.Module):
    def __init__(self, lr, inputShape, fc1Size, fc2Size, outputSize):
        super().__init__()
        self.inputshape = inputShape
        self.fc1Size = fc1Size
        self.fc2Size = fc2Size
        self.outputShape = outputSize
        
        self.fc1 = nn.Linear(*inputShape, fc1Size)
        self.fc2 = nn.Linear(fc1Size, fc2Size)
        self.fc3 = nn.Linear(fc2Size, outputSize)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQDiscreteAgent():
    def __init__(self, lr, inputShape, numActions, batchSize, epsilon=1.0, gamma=0.99, layer1Size=256, 
            layer2Size=256, maxMemSize=100000, epsMin=0.01, epsDecay=5e-4):
        self.lr = lr
        self.epsilon = epsilon
        self.epsMin = epsMin
        self.epsDecay = epsDecay
        self.gamma = gamma
        self.batchSize = batchSize
        self.actionSpace = list(range(numActions))
        
        self.memory = ReplayBuffer(maxMemSize, inputShape)
        self.deepQNetwork = DeepQNetwork(lr, inputShape, layer1Size, layer2Size, numActions)

    def chooseAction(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float32).to(self.deepQNetwork.device)
            policy = self.deepQNetwork(state)
            action = torch.argmax(policy).item()
            return action
        else:
            return np.random.choice(self.actionSpace)

    def storeMemory(self, state, action, reward, nextState, done):
        self.memory.storeMemory(state, action, reward, nextState, done)

    def learn(self):
        if self.memory.memCount < self.batchSize:
            return

        self.deepQNetwork.optimizer.zero_grad()
    
        stateBatch, actionBatch, rewardBatch, nextStateBatch, doneBatch = \
            self.memory.sample(self.batchSize)
        stateBatch = torch.tensor(stateBatch).to(self.deepQNetwork.device)
        actionBatch = torch.tensor(actionBatch).to(self.deepQNetwork.device)
        rewardBatch = torch.tensor(rewardBatch).to(self.deepQNetwork.device)
        nextStateBatch = torch.tensor(nextStateBatch).to(self.deepQNetwork.device)
        doneBatch = torch.tensor(doneBatch).to(self.deepQNetwork.device)
        
        batchIndex = np.arange(self.batchSize, dtype=np.int64)


        actionQs = self.deepQNetwork(stateBatch)[batchIndex, actionBatch]
        allNextActionQs = self.deepQNetwork(nextStateBatch)
        nextActionQs = torch.max(allNextActionQs, dim=1)[0]
        nextActionQs[doneBatch] = 0.0
        qTarget = rewardBatch + self.gamma * nextActionQs

        loss = self.deepQNetwork.loss(qTarget, actionQs).to(self.deepQNetwork.device)
        loss.backward()
        self.deepQNetwork.optimizer.step()

        if self.epsilon > self.epsMin:
            self.epsilon -= self.epsDecay

if __name__ == '__main__':
    import gym
    import math
    from matplotlib import pyplot as plt
    
    agent = DQDiscreteAgent(lr=0.0001, inputShape=(2,), numActions=3, batchSize=64, 
        epsilon=1.0, gamma=0.99, layer1Size=64, layer2Size=64, maxMemSize=100000, 
        epsMin=0.01, epsDecay=5e-4)
    env = gym.make("MountainCar-v0")

    scoreHistory = []
    numEpisodes = 2000
    numTrainingEpisodes = 50
    highScore = -math.inf
    recordTimeSteps = math.inf
    for episode in range(numEpisodes):
        done = False
        observation = env.reset()
        score, frame = 0, 1
        maxPos = -math.inf
        maxVel = -math.inf
        while not done:
            if episode > numTrainingEpisodes:
                env.render()
            action = agent.chooseAction(observation)
            nextObservation, reward, done, info = env.step(action)
            pos = -abs(env.goal_position - observation[0])
            vel = abs(observation[1])
            maxPos = max(pos, maxPos)
            maxVel = max(vel, maxVel)
            # reward = maxPos + maxVel
            reward = vel
            agent.storeMemory(observation, action, reward, nextObservation, done)
            agent.learn()
            observation = nextObservation
            score += reward
            frame += 1
        scoreHistory.append(score)

        recordTimeSteps = min(recordTimeSteps, frame)
        highScore = max(highScore, score)
        print(( "ep {}: high-score {:12.3f}, shortest-time {:d}, "
                "score {:12.3f}, last-episode-time {:4d}").format(
            episode, 
            highScore, 
            recordTimeSteps, 
            score,
            frame,
            ))

    fig = plt.figure()
    meanWindow = 2
    meanedScoreHistory = np.convolve(scoreHistory, np.ones(meanWindow), 'valid') / meanWindow
    plt.plot(np.arange(0, numEpisodes-1, 1.0), meanedScoreHistory)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.title("Training Scores")
    plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ReplayBuffer():
    def __init__(self, maxSize, stateShape, numActions):
        self.memSize = maxSize
        self.memCount = 0
        self.stateMemory = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.actionMemory = np.zeros((self.memSize, numActions), dtype=np.int64)
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
    def __init__(self, lr, inputShape, fc1Size, fc2Size, numActions):
        super().__init__()
        self.inputshape = inputShape
        self.fc1Size = fc1Size
        self.fc2Size = fc2Size
        self.numActions = numActions

        #   shared backbone
        self.fc1 = nn.Linear(*inputShape, fc1Size)
        self.fc2 = nn.Linear(fc1Size, fc2Size)
        
        #   tail networks
        self.policy = nn.Linear(fc2Size, numActions*2)
        self.valueScalers = nn.Linear(fc2Size, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        batchSize = x.shape[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.policy(x).view(batchSize, 2, self.numActions)
        valueScalers = self.valueScalers(x)
        return policy, valueScalers

class DQAgent():
    def __init__(self, lr, inputShape, numActions, maxMemSize=1000, batchSize=64, epsilon=1.0, gamma=0.99, layer1Size=256, 
            layer2Size=256, epsMin=0.01, epsDecay=5e-4):
        self.lr = lr
        self.epsilon = epsilon
        self.epsMin = epsMin
        self.epsDecay = epsDecay
        self.gamma = gamma
        self.batchSize = batchSize
        self.numActions = numActions
        self.actionSpace = list(range(numActions))
        
        self.memory = ReplayBuffer(maxMemSize, inputShape, numActions)
        self.deepQNetwork = DeepQNetwork(lr, inputShape, layer1Size, layer2Size, numActions)

    def chooseActions(self, state):
        state.to(self.deepQNetwork.device)
        policy, valueScalers = self.deepQNetwork.forward(state)
        mus, sigmas = policy[:][:, 0], policy[:][:, 1]
        sigmas = torch.exp(sigmas)
        distributions = torch.distributions.Normal(mus, sigmas)   
        actions = distributions.sample()
        actions = torch.tanh(actions)

        return actions.cpu().numpy()

    def getActionValues(self, observation, actions):
        print(observation.shape)
        print(actions.shape)
        quit()
        state.to(self.deepQNetwork.device)
        policy, valueScalers = self.deepQNetwork.forward(state)
        mus, sigmas = policy[:][:, 0], policy[:][:, 1]
        sigmas = torch.exp(sigmas)
        distributions = torch.distributions.Normal(mus, sigmas)   
        logProbs = distributions.log_prob(actions)
        values = logProbs * valueScalers

        return values

    def chooseAndEvalAction(self, observation):
        policy, valueScalers = self.deepQNetwork.forward(observation)
        mus, sigmas = policy
        sigmas = torch.exp(sigmas)
        distributions = torch.distributions.Normal(mus, sigmas)        
        actions = distributions.sample()
        logProbs = distributions.log_prob(actions)
        actions = torch.tanh(actions)
        values = logProbs * valueScalers

        return actions.cpu().numpy(), values

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
        
        batchIndex = np.arange(self.memory.memSize, dtype=np.int64)


        actionsQs = self.getActionValues(stateBatch, actionBatch)

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
    
    agent = DQAgent(lr=0.01, inputShape=(8,), numActions=2, maxMemSize=1000, batchSize=4)
    env = gym.make("LunarLanderContinuous-v2")

    scoreHistory = []
    numEpisodes = 2000
    numTrainingEpisodes = 30
    highScore = -math.inf
    recordTimeSteps = math.inf
    for episode in range(numEpisodes):
        done = False
        observation = env.reset()
        score, frame = 0, 1
        while not done:
            print(frame)
            if episode > numTrainingEpisodes:
                env.render()

            state = torch.tensor(observation)
            state = state.unsqueeze(0)
            actions = agent.chooseActions(state)[0]

            nextObservation, reward, done, info = env.step(actions)
            agent.storeMemory(observation, actions, reward, nextObservation, done)
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
    meanWindow = 10
    meanedScoreHistory = np.convolve(scoreHistory, np.ones(meanWindow), 'valid') / meanWindow
    plt.plot(np.arange(0, numEpisodes-1, 1.0), meanedScoreHistory)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.title("Training Scores")
    plt.show()
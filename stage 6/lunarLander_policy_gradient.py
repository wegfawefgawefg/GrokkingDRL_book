import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
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

class PolicyGradientAgent():
    def __init__(self, lr, inputShape, numActions, gamma=0.99, layer1Size=256, 
            layer2Size=256, maxMemSize=1001):
        self.lr = lr
        self.gamma = gamma
        self.actionSpace = list(range(numActions))
        
        self.policyNet = PolicyNetwork(lr, inputShape, layer1Size, layer2Size, numActions)

        self.rewardMemory = []
        self.actionProbMemory = []

    def chooseAction(self, observation):
        state = torch.tensor(observation).to(self.policyNet.device)
        pred = self.policyNet(state)
        probabilities = F.softmax(pred)
        actionProbs = torch.distributions.Categorical(probabilities)
        action = actionProbs.sample()
        logProbs = actionProbs.log_prob(action)
        self.actionProbMemory.append(logProbs)

        return action.item()

    def learn(self):

        self.policyNet.optimizer.zero_grad()
    
        G = np.zeros_like(self.rewardMemory, dtype=np.float64)
        for t in range(len(self.rewardMemory)):
            gSum = 0
            discount = 1
            for k in range(t, len(self.rewardMemory)):
                gSum += self.rewardMemory[k] * discount
                discount *= self.gamma
            G[t] = gSum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std

        G = torch.tensor(G, dtype=torch.float).to(self.policyNet.device)

        loss = 0
        for g, logprob in zip(G, self.actionProbMemory):
            loss += -g * logprob

        loss.backward()
        self.policyNet.optimizer.step()

        self.actionProbMemory = []
        self.rewardMemory = []

if __name__ == '__main__':
    import gym
    import math
    from matplotlib import pyplot as plt
    
    agent = PolicyGradientAgent(lr=0.001, inputShape=(8,), numActions=4)
    env = gym.make("LunarLander-v2")

    scoreHistory = []
    numEpisodes = 2000
    numTrainingEpisodes = 500
    highScore = -math.inf
    recordTimeSteps = math.inf
    for episode in range(numEpisodes):
        done = False
        observation = env.reset()
        score, frame = 0, 1
        while not done:
            if episode > numTrainingEpisodes:
                env.render()
            action = agent.chooseAction(observation)
            nextObservation, reward, done, info = env.step(action)
            agent.rewardMemory.append(reward)
            observation = nextObservation
            score += reward
            frame += 1
        agent.learn()
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
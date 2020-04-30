import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, alpha, inputDims, fc1Dims, fc2Dims, numActions):
        super().__init__()
        self.inputDims = inputDims
        self.numActions = numActions
        self.fc1Dims = fc1Dims
        self.fc2Dims = fc2Dims

        #   primary network
        self.fc1 = nn.Linear(*inputDims, fc1Dims)
        self.fc2 = nn.Linear(fc1Dims, fc2Dims)

        #   tail networks
        self.policy = nn.Linear(self.fc2Dims, numActions)
        self.critic = nn.Linear(self.fc2Dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        state = torch.tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = self.policy(x)
        value = self.critic(x)
        return policy, value

class ActorCriticDiscreteDQAgent():
    def __init__(self, alpha, inputDims, gamma=0.99, layer1Size=256, layer2Size=256, numActions=2):
        self.gamma = gamma
        self.actorCritic = ActorCriticNetwork(alpha, inputDims, layer1Size, layer2Size, numActions)
        self.logProbs = None

    def chooseAction(self, observation):
        policy, _ = self.actorCritic.forward(observation)
        policy = F.softmax(policy, dim=0)
        actionProbs = torch.distributions.Categorical(policy)
        action = actionProbs.sample()
        self.logProbs = actionProbs.log_prob(action)
        return action.item()

    def learn(self, state, reward, nextState, done):
        self.actorCritic.optimizer.zero_grad()

        _, criticValue = self.actorCritic.forward(state)
        _, nextCriticValue = self.actorCritic.forward(nextState)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actorCritic.device)
        delta = reward + self.gamma * nextCriticValue * (1 - int(done)) - criticValue

        actorLoss = -self.logProbs * delta
        criticLoss = delta**2

        (actorLoss + criticLoss).backward()
        self.actorCritic.optimizer.step()

if __name__ == '__main__':
    import gym
    import math
    from matplotlib import pyplot as plt
    
    agent = ActorCriticDiscreteDQAgent(
        alpha=0.00001, inputDims=(8,), gamma=0.99, numActions=4, layer1Size=2048, layer2Size=512)
    agent = ActorCriticDiscreteDQAgent(
        alpha=0.00001, inputDims=(8,), gamma=0.99, numActions=4, layer1Size=128, layer2Size=64)
    env = gym.make("LunarLander-v2")

    scoreHistory = []
    numEpisodes = 2000
    numTrainingEpisodes = 50
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
            agent.learn(observation, reward, nextObservation, done)
            observation = nextObservation
            score += reward
            frame += 1
        scoreHistory.append(score)

        recordTimeSteps = min(recordTimeSteps, frame)
        highScore = max(highScore, score)
        print(( "ep {}: high-score {:12.3f}, shortest-time {:d}, "
                "score {:12.3f}, last-epidode-time {:4d}").format(
            episode, 
            highScore, 
            recordTimeSteps, 
            score,
            frame,
            ))

    fig = plt.figure()
    plt.plot(np.arange(0, numEpisodes, 1.0), scoreHistory)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.title("Training Scores")
    plt.show()
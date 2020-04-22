class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
        step += 1
        #env.render()
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        state_next = np.reshape(state_next, [1, observation_space])
        dqn_solver.remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print( "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
            break
        dqn_solver.experience_replay()

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.train()

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

epsilon = 1.0
epsilonDecay = 0.999

recordReward = 0
recordTimesteps = 0
episode = 0

while True:
    obs = env.reset()

    netData = torch.tensor(obs, dtype=torch.float32)
    output = net(netData)
    value, index = output.max(0)
    action = int(index)

    done = False
    t = 0
    totalReward = 0
    while not done:
    # for t in range(50):
        env.render()
        obs, reward, done, info = env.step(action)
        totalReward += reward

        #   punish

        #   make target
        target = torch.tensor([1, 1], dtype=torch.float32)  #   incentivise exploration
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
    
    if t+1 > recordTimesteps:
        recordTimesteps = t+1
    if totalReward > recordReward:
        recordReward = totalReward
    print("ep {} recRew {} recTs {} lastRew {} lastTs {} e {}".format(
        episode, recordReward, recordTimesteps, totalReward, t+1, str(epsilon)[:4]))
    epsilon *= epsilonDecay
    # print(epsilon)

env.close()
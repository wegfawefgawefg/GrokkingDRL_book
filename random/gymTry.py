import gym
env = gym.make('MountainCar-v0')
while True:
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample()) # take a random action
env.close()
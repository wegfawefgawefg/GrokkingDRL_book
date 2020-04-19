import gym
from gym import spaces

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

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)
while True:
    env.reset()
    nextAction = 0
    for t in range(1000):
        env.render()
        obs, reward, done, info = env.step(nextAction)
        pos, vel, angle, tipVel = obs
        if angle < 0:
            nextAction = 0
        else:
            nextAction = 1
        # print(obs)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
env.close()
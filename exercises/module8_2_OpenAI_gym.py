# Deep Learning with Pytorch
# Module 8: Reinforcement Learning
# Gym Demo

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
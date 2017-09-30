# Deep Learning with Pytorch
# Module 8: Reinforcement Learning
# Q-Leanring Demo for Cartpole

import gym
from gym import wrappers
import qlearning as qla
import collections

NUM_EPISODES = 2000
EPISODE_LENGTH = 200
DISCOUNT = 0.9
EXPLORATION_PROB = 0.02

def CartPoleFeaturExtractor(state, action):
    #Round so that features can generalize between states slightly
    state = (round(state[0],3), round(state[1],2), round(state[2],3), round(state[3],2))

    featureVector = []
    featureVector.append((state, 1))
    featureVector.append((("f0", state[0], action), 1))
    featureVector.append((("f1", state[1], action), 1))
    featureVector.append((("f2", state[2], action), 1))
    featureVector.append((("f3", state[3], action), 1))
    
    return featureVector

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, 'C:\\Users\\Tyler\\Downloads\\tmp\\cart-pole', force=True)

ql = qla.QLearningAlgorithm(env.action_space, DISCOUNT, CartPoleFeaturExtractor, EXPLORATION_PROB)
record = collections.deque(100*[0], 100)

for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    for t in range(EPISODE_LENGTH):
        env.render()

        # Your Code Here #
        action = ql.getAction(tuple(observation))
        oldObs = observation
        # End Your Code #

        observation, reward, done, info = env.step(action)

         # Your Code Here #
        true_reward = -1 if done is True and t < EPISODE_LENGTH-1 else 0 # Negative reward if agent fails
        ql.incorporateFeedback(tuple(oldObs), action, true_reward, tuple(observation))
        # End Your Code #

        if done:    #Print relevant info and start new episode
            record.appendleft(t+1)
            print("Episode ", i_episode+1, " finished after {} timesteps".format(t+1))
            if i_episode % 10 is 0:
                print("\tRolling Ave: ", float(sum(record)) / max(len(record), 1))
            break

env.close()
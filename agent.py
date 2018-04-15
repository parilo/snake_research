#!/usr/bin/env python

import argparse
import numpy as np
import random
#import gym
import cv2
from environments import Snake
from rl_client import RLClient

parser = argparse.ArgumentParser(description='snake game')
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--explore', dest='explore', action='store_true', default=False)
args = parser.parse_args()
vis = args.visualize
explore = args.explore

env = Snake()
env.reset()
# img = s.reset()
# s.plot_state()
#
# print(img.shape)
# print(img)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rl_client = RLClient ()

observation_size = 5 * 8 * 8
num_actions = 3

n_steps = 1

class RewardSequence (object):

    def __init__ (self, n_steps):
        self.n_steps = n_steps
        self.reset()

    def append_reward (self, reward):
        self.rewards.append(reward)
        self.rewards.pop(0)

    def get_sum_of_rewards (self):
        r_sum = 0.0
        for r in self.rewards:
            r_sum += r
        return r_sum

    def reset(self):
        self.rewards = [0.0] * self.n_steps

class ActionSequence (object):

    def __init__ (self, n_steps):
        self.n_steps = n_steps
        self.reset()

    def append_action (self, action):
        self.actions.append(action)
        self.actions.pop(0)

    def get_oldest (self):
        return self.actions[0]

    def get_youngest (self):
        return self.actions[-1]

    def reset(self):
        self.actions = [[0.0] * num_actions for _ in range(self.n_steps)]

class ObservationSequence (object):

    def __init__ (self, n_steps):
        self.n_steps = n_steps
        self.reset()

    def append_obs (self, obs):
        self.obs_seq = np.concatenate((self.obs_seq[1:], np.array([obs])), axis=0)

    def get_flatten_obs_seq (self):
        obs = np.copy(self.obs_seq)
        obs_flatten = obs.reshape((-1))
        return obs_flatten.tolist()

    def get_flatten_oldest(self):
        obs = np.copy(self.obs_seq[0])
        obs_flatten = obs.reshape((-1))
        return obs_flatten.tolist()

    def get_flatten_youngest(self):
        obs = np.copy(self.obs_seq[-1])
        obs_flatten = obs.reshape((-1))
        return obs_flatten.tolist()

    def reset(self):
        self.obs_seq = np.zeros((self.n_steps, observation_size))

class MeanCalculator (object):
    def __init__(self, size=100):
        self._size = size
        self._values = [0.0] * size
        self._mean = 0.0
        self._mean_add_coeff = 1.0 / size

    def push(self, val):
        self._mean += self._mean_add_coeff * val
        self._values.append(val)
        removed_val = self._values.pop(0)
        self._mean -= self._mean_add_coeff * removed_val

    def get_mean(self):
        return self._mean

episode_reward_mean = MeanCalculator()


def preprocess_observation(obs):
    return obs.reshape((-1))

def preprocess_observation_(obs):

    # 0 — поле
    # 1 — змейка
    # 2 — голова
    # 3 — еда

    # large_obs = np.zeros((24, 24, 1))
    # large_obs.fill(4)
    # large_obs[8:16, 8:16, :] = obs

    # x_head, y_head = env.get_head_coords()
    # # print('head {} {}'.format(x_head, y_head))
    # large_obs = np.roll(large_obs, 4-x_head, 0)
    # large_obs = np.roll(large_obs, 4-y_head, 1)

    # large_obs = np.rot90(large_obs, env.turn_state)
    # if env.turn_state == 1:
    #     large_obs = np.roll(large_obs, 1, 0)
    # elif env.turn_state == 2:
    #     large_obs = np.roll(large_obs, 1, 0)
    #     large_obs = np.roll(large_obs, 1, 1)
    # elif env.turn_state == 3:
    #     large_obs = np.roll(large_obs, 1, 1)

    # obs = large_obs[8:16, 8:16, :]

    # obs = np.rot90(obs, env.turn_state)

    body = np.zeros_like(obs)
    body[obs == 1] = 1
    body[obs == 2] = 1

    head = np.zeros_like(obs)
    head[obs == 2] = 1

    food = np.zeros_like(obs)
    food[obs == 3] = 1

    # print ('{} {} {} {}'.format(
    #     obs.shape, body.shape, head.shape, food.shape
    # ))

    state = np.concatenate([body, head, food], axis=0)
    # state1 = cv2.resize(state, (128, 3*128), cv2.INTER_NEAREST)

    # cv2.imshow('image', state1)
    # cv2.waitKey(0)
    return state.reshape((-1))

prev_obs_seq = ObservationSequence (n_steps)
next_obs_seq = ObservationSequence (n_steps)
rewards_seq = RewardSequence (n_steps)
actions_seq = ActionSequence (n_steps)

prev_observation = preprocess_observation(env.reset())
prev_action = [0.0] * num_actions
next_obs_seq.append_obs (prev_observation)
episode_reward = 0.0
episode_length = 0

# init_action = None
# def reset_init_action ():
#     global init_action
#     init_action = np.round(np.random.uniform (0, 0.7, size=18)).tolist()
# def get_init_action ():
#     return init_action
#
# init_i = 0
# reset_init_action ()

step_count = 0

action_queue = []

while True:

    # init_i += 1
    # if init_i < 0 and not vis:
    #     if init_i == 20:
    #         reset_init_action ()
    #     action_received2 = get_init_action ()
    #     # print (action_received)
    #     next_observation, reward, done, info = env.step(action_received2)
    #
    # else:

    prev_obs_seq.append_obs (prev_observation)

    #action_received = rl_client.act (prev_obs_seq.get_flatten_obs_seq())
    #action = (np.array(action_received) + np.random.normal(scale=0.02, size=num_actions)).tolist()
    #action[action > 1.0] = 1.0
    #action[action < 0.0] = 0.0
    # action_received = action
    # action = (np.array(action_received) + np.random.normal(scale=0.01, size=num_actions)).tolist()
    # action = action_received
    #action = process_act(action)
    # action = np.array(action_received)
    # action = action + np.random.normal(scale=0.05, size=18)

    # action_received = np.random.uniform(0.0, 1.0, 4)
    # if len(action_queue) == 0:
    #     action_received = rl_client.act (prev_obs_seq.get_flatten_obs_seq())
    #     a = np.array(action_received).reshape((-1, 4))
    #     action_queue.append(a[0])
    #     action_queue.append(a[1])

    # action = action_received.pop(0)

    # action_received = rl_client.act (prev_obs_seq.get_flatten_obs_seq())
    action_and_value = rl_client.act (prev_obs_seq.get_flatten_youngest())

    # if explore and random.randint(0, 19) == 0:
    #     action_received = np.random.uniform(-1.0, 1.0, size=num_actions).tolist()
    # else:
    action_received = action_and_value['action']

    action = np.argmax(np.array(action_received) + np.random.normal(scale=0.01, size=num_actions))
    # print('--- action: {} {}'.format(action, action_received))

    next_observation, reward, done = env.step(action)
    episode_reward += reward
    episode_length += 1
    next_observation = preprocess_observation(next_observation)
    next_obs_seq.append_obs (next_observation)
    rewards_seq.append_reward (reward)
    actions_seq.append_action (action_received)

    if vis:
        print('--- q: {: f} {: f} {} {}'.format(
            action_and_value['qvalue'][0],
            action_and_value['boltzmann_exploration_t'][0],
            '^' if action == 0 else '<' if action == 1 else '>',
            '1' if reward == 1 else ' '
        ))

    if episode_length > n_steps:
        # rl_client.store_exp (
        #     reward,
        #     action_received,
        #     prev_obs_seq.get_flatten_obs_seq(),
        #     next_obs_seq.get_flatten_obs_seq()
        # )
        rl_client.store_exp (
            rewards_seq.get_sum_of_rewards(),
            actions_seq.get_youngest(),
            prev_obs_seq.get_flatten_oldest(),
            next_obs_seq.get_flatten_youngest(),
            terminator=done
        )

    prev_observation = next_observation
    prev_action = action_received

    step_count += 1

    if done:
        # init_i = 0
        # reset_init_action ()

        if episode_length > 2:
            episode_reward_mean.push(episode_reward)
            if vis:
                print('done, reward {} {} {}'.format(episode_reward, episode_length, episode_reward_mean.get_mean()))
        episode_reward = 0.0
        episode_length = 0
        prev_observation = preprocess_observation(env.reset())

        prev_action = [0] * num_actions
        prev_obs_seq.reset()
        next_obs_seq.reset()
        rewards_seq.reset()
        actions_seq.reset()
        next_obs_seq.append_obs (prev_observation)

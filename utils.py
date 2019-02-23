#!/usr/bin/python
import os 
from abc import ABCMeta, abstractmethod
import gym
import numpy as np
from collections import deque
import random

class ReplayBuffer():

    def __init__(self,max_buffer_size):
        self.size = 0
        self.max_size = max_buffer_size
        self.buffer = deque()

    def add(self,s, a, r, d, s_nxt):
        experience = (s, a, r, d, s_nxt)

        if self.size >= self.max_size:
            self.buffer.popleft()
        else:
            self.size += 1

        self.buffer.append(experience)

    def sample(self,batch_size):

        batch = random.sample(self.buffer,min(batch_size,self.size))
        state_mb, action_mb, reward_mb, done_mb, nxt_state_mb  = [], [], [], [], []

        for s, a, r, d, nxt_s in batch:
            state_mb.append(s)
            action_mb.append(a)
            reward_mb.append(r)
            done_mb.append(d)
            nxt_state_mb.append(nxt_s)

        return np.array(state_mb), np.array(action_mb), np.array(reward_mb), np.array(done_mb), np.array(nxt_state_mb)




def get_run_nr(file_dir,starts_with="run"):

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    run_nr = 0

    for d in os.listdir(file_dir):
        if d.startswith(starts_with+"_"):
            run_nr += 1

    return run_nr

def evaluate_agent(agent, env, n_games=1):

    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.policy([state])[0]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done: break
        game_rewards.append(total_reward)
    return game_rewards


class EnvBatch(object):

    def __init__(self, name, n_envs=10):
        self.envs_ = [gym.make(name) for i in range(n_envs)]

    def reset(self):
        return np.array([env.reset() for env in self.envs_])

    def step(self, actions):
        self.results_ = [env.step(action) for action, env in zip(actions, self.envs_)]
        nxt_state, rewards, done, _ = map(np.array, zip(*self.results_))
        for i, d in enumerate(done):
            if d:
                nxt_state[i, :] = self.envs_[i].reset()
        return nxt_state, rewards, done

class BaseAgent():
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def policy(self,state):
        pass

    @abstractmethod
    def train(self):
        pass



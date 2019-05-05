#!/usr/bin/python
import tensorflow as tf
import os
from abc import ABCMeta, abstractmethod
import gym
import numpy as np
from collections import deque
import random

def compute_discounted_return(rewards, gamma=0.95, baseline=False):

    discounted_rewards = []
    acc_r = 0.0

    for r in reversed(rewards):
        acc_r = r + gamma*acc_r
        discounted_rewards.append(acc_r)

    if baseline:
        discounted_rewards = (discounted_rewards- np.mean(discounted_rewards))/np.std(discounted_rewards)

    return list(reversed(discounted_rewards))


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


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

def evaluate_agent(agent, env, n_games=1,greedy=False):

    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        while True:
            if greedy:
                action = agent.policy([state],greedy)[0]
            else:
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
        results_ = [env.step(action) for action, env in zip(actions, self.envs_)]
        nxt_state, rewards, done, _ = map(np.array, zip(*results_))
        for i, d in enumerate(done):
            if d:
                nxt_state[i, :] = self.envs_[i].reset()
        return nxt_state, rewards, done

class BaseDeepAgent:

    __metaclass__ = ABCMeta

    def __init__(self, env, sess, log_data=True):

        self.env = env
        self.sess = sess
        self.nr_env_interactions = 0
        self.action_bound = None
        self.n_actions = None
        self.a_dim = None
        self.is_discrete = False

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
            self.is_discrete = True
        else:
            self.a_dim = env.action_space.shape[0]
            self.action_bound = abs(env.action_space.low)

        self.s_dim = env.observation_space.shape[0]

        if log_data:

            self.env_name = self.env.spec.id
            run_nr = get_run_nr("runs_" + self.env_name, starts_with=str(self))
            self.writer = tf.summary.FileWriter("runs_%s/%s_%i" % (self.env_name, str(self),run_nr), sess.graph)

    @abstractmethod
    def _init_tb_summaries(self):
        NotImplementedError()

    @abstractmethod
    def policy(self, state):
        NotImplementedError()

    @abstractmethod
    def train(self):
        NotImplementedError()

    @abstractmethod
    def run_episodes(self,n_episodes,verbosity=0):
        NotImplementedError()

    def close(self):
        self.sess.close()
        self.writer.close()

class BaseActor:

    __metaclass__ = ABCMeta

    def __init__(self, env, sess, log_data=True):
        pass
    @abstractmethod
    def policy(self, state):
        NotImplementedError()

    @abstractmethod
    def train(self):
        NotImplementedError()


class BaseCritic:
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()
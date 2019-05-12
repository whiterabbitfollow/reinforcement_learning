#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:21:25 2019

@author: x
"""

from utils.base_agents import BaseDeepAgent
from utils.other import evaluate_agent

import numpy as np
import tensorflow as tf
import gym
import argparse

class QLearning(BaseDeepAgent):

    def __init__(self, env, sess, gamma=0.99, eps=0.9, lr=10 ** -3, num_layers=2, eps_decay=0.99, min_eps=0.3):

        super(QLearning, self).__init__(env, sess)
        self.gamma = gamma
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.eps = eps
        self._init_ph()
        self._init_net("QLearning",lr,num_layers)
        self._init_tb_summaries()


    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.loss_var = tf.Variable(0.0, name="loss")
            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.train_reward_var = tf.Variable(0.0, name="train_reward")
            self.eps_var = tf.Variable(0.0, name="eps")

            self.merged = tf.summary.merge([tf.summary.scalar("loss", self.loss_var),
                                            tf.summary.scalar("train_reward", self.train_reward_var),
                                            tf.summary.scalar("eps",self.eps_var)
                                            ])

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)

    def _init_ph(self):
        self.states_ph = tf.placeholder(tf.float32, (None, self.s_dim), name="state")
        self.actions_ph = tf.placeholder(tf.int32, (None,), name="action")
        self.action_gradients_ph = tf.placeholder(tf.float32, (None, 1), "action_grads")
        self.q_target_ph = tf.placeholder(tf.float32, (None, ), "Q_target")

    def _init_net(self, name, lr, nr_layers=2, reuse=False):

        with tf.variable_scope(name,reuse=reuse):

            out = self.states_ph
            for layer_nr in range(nr_layers):
                out = tf.layers.dense(out, 100, activation=tf.nn.relu)

            self.q_values = tf.layers.dense(out, self.n_actions, activation=None)

            action_ph_one_hot = tf.one_hot(self.actions_ph, self.n_actions)
            q_values_one_hot = tf.math.multiply(action_ph_one_hot, self.q_values)

            self.q = tf.reduce_sum(q_values_one_hot, axis=-1)

            self.TD_error = tf.math.squared_difference(self.q_target_ph, self.q)
            self.loss = tf.reduce_mean(self.TD_error)

            self.train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def policy(self, states, greedy=False, q_values=None):

        if q_values is None:
            q_values = self.get_q_values(states)[0]

        if np.random.random() > self.eps or greedy:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)

        return [action]

    def decay_eps(self):
        self.eps = max(self.min_eps, self.eps * self.eps_decay)

    def get_q_values(self,states):
        return self.sess.run(self.q_values, {self.states_ph:states})

    def train(self, q_targets, states, actions):

        feed_dict = {self.states_ph: states,
                     self.actions_ph: actions,
                     self.q_target_ph: q_targets}

        loss, _ = self.sess.run([self.loss, self.train_opt], feed_dict)

        return loss

    def run_episodes(self,n_epochs,n_episodes,verbosity=0,eval_freq=10, n_interact_2_evaluate=10):

        for i_epoch in range(n_epochs):

            mean_loss_all_eps = []
            mean_train_reward_all_eps = []
            for i_eps in range(n_episodes):

                train_reward, mean_loss = self.run_episode()

                mean_loss_all_eps.append(mean_loss)
                mean_train_reward_all_eps.append(train_reward)


            feed_dict = {self.train_reward_var: np.mean(mean_train_reward_all_eps),
                                                  self.loss_var: np.mean(mean_loss_all_eps),
                                                  self.eps_var: self.eps
                                                  }

            merge_summary = self.sess.run(self.merged, feed_dict)
            self.writer.add_summary(merge_summary, self.nr_env_interactions)
            self.writer.flush()

            if i_epoch % eval_freq == 0:

                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate,greedy=True))
                summary = self.sess.run(self.eval_summary, {self.eval_reward_var: np.mean(eval_rewards)})
                self.writer.add_summary(summary, self.nr_env_interactions)
                self.writer.flush()

            self.decay_eps()


    def __repr__(self):
        return "QLearning"

    def run_episode(self):

        done = False

        state = self.env.reset()

        acc_reward = 0.0
        acc_loss = 0.0
        nr_steps = 0

        while not done:

            action = self.policy([state])[0]

            next_state, reward, done, _ = self.env.step(action)

            q_values_next_state = self.get_q_values([next_state])

            q_values_next_state_max = np.max(q_values_next_state[0])

            q_target = reward + self.gamma * q_values_next_state_max * (1.0 - done)

            loss = self.train([q_target], [state], [action])

            state = next_state

            acc_reward += reward

            acc_loss += loss

            nr_steps += 1

            self.nr_env_interactions += 1

        return acc_reward, acc_loss


def main(args):

    env = gym.make(args["env"])

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = QLearning(env,sess,args["gamma"], args["eps"], args["learning_rate"], args["num_layers"], args["eps_decay"], args["min_eps"] )

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(args["n_epochs"], args["n_episodes"], args["verbosity"], args["eval_freq"], args["n_interact_2_evaluate"])

    agent.close()

if __name__=="__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--eval_freq", type=int, default=10)
    args.add_argument("--n_interact_2_evaluate", type=int, default=100)
    args.add_argument("--n_episodes", type=int, default=10)
    args.add_argument("--env", type=str, default="MountainCar-v0")
    args.add_argument("--n_epochs", type=int, default=1000)
    args.add_argument("--num_layers", type=int, default=2)

    args.add_argument("--eps", type=float, default=0.9)
    args.add_argument("--min_eps", type=float, default=0.3)
    args.add_argument("--eps_decay", type=float, default=0.995)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--verbosity", type=int, default=1)

    main(vars(args.parse_args()))

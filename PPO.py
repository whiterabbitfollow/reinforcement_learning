#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:43:42 2018
@author: x

"""
import tensorflow as tf
import gym
import numpy as np
from utils import evaluate_agent, BaseActor, BaseDeepAgent, BaseCritic
import argparse
import time

class Actor(BaseActor):

    def __init__(self, state_dim, action_dim, actor_lr, init_log_var,beta, clip_range=0.3, scope="actor"):

        self.scope = scope
        self.sess = tf.get_default_session()
        self.init_log_var = init_log_var

        assert self.sess is not None, "sess is None"

        with tf.variable_scope(self.scope):
            self._init_ph(state_dim,action_dim)
            self._build_net(action_dim)
            self._losses(clip_range,action_dim,beta)
            self._grads()
            self._assign_weights_ops()
            self._init_train_opt(actor_lr)

    def _init_ph(self,state_dim,action_dim):

        self.action_ph = tf.placeholder(tf.float32, (None,action_dim), name="action")
        self.state_ph = tf.placeholder(tf.float32, (None,state_dim) , name="state")
        self.old_means_ph = tf.placeholder(tf.float32, (None,action_dim), name="old_means")
        self.old_log_vars_ph = tf.placeholder(tf.float32, action_dim, name="old_log_vars")
        self.advantage_ph = tf.placeholder(tf.float32, (None,), name="advantage")

    def _build_net(self,action_dim):

            out = tf.layers.dense(self.state_ph, 64, activation=tf.tanh)
            out = tf.layers.dense(out, 64, activation=tf.tanh)

            self.means = tf.layers.dense(out, action_dim, activation=None)
            self.log_vars = tf.get_variable("logvars", action_dim, tf.float32, tf.constant_initializer(0.0)) + self.init_log_var

    def _losses(self,clip_range,action_dim, beta):

            self.logp = self.calc_logprob(self.means, self.action_ph, self.log_vars)
            self.logp_old = self.calc_logprob(self.old_means_ph, self.action_ph, self.old_log_vars_ph)

            log_det_cov_new = tf.reduce_sum(self.log_vars)
            log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)

            tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

            means_n_cov = tf.square(self.means-self.old_means_ph)/tf.exp(self.log_vars)

            log_det_cov_new_old = log_det_cov_new - log_det_cov_old

            k = action_dim

            self.kl = tf.reduce_sum(tr_old_new + means_n_cov - k + log_det_cov_new_old) * 0.5
            self.kl_pen = tf.reduce_mean(self.kl)

            self.entropy = 0.5 * (k + np.log(2*np.pi)*k + log_det_cov_new)
            self.ratio = tf.exp(self.logp - self.logp_old)

            self.pg_loss_unclipped = - tf.reduce_mean(self.ratio * self.advantage_ph) # obs sign!!
            self.pg_loss_clipped = - tf.clip_by_value(self.ratio, 1.0 - clip_range, 1.0 + clip_range)*self.advantage_ph
            self.sampled_action = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(action_dim,), dtype=tf.float32))  # TODO: read about this

            self.loss = tf.reduce_mean(tf.maximum(self.pg_loss_unclipped, self.pg_loss_clipped)) + beta * self.kl_pen # WARNING SIGN!!

    def _init_train_opt(self,actor_lr,max_grad_norm=None):

        self.grads = tf.gradients(self.loss, self.weights)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            # TODO ADD
            grads, grad_norm = tf.clip_by_global_norm(self.grads, max_grad_norm)

        self.train_opt = tf.train.AdamOptimizer(learning_rate=actor_lr).apply_gradients(list(zip(self.grads, self.weights)))

    def train(self, state_mb, action_mb, advantage_mb):
        feed_dict = {self.state_ph: state_mb,
                     self.action_ph: action_mb,
                     self.advantage_ph: advantage_mb
                     }

        old_means, old_log_vars = self.predict(state_mb)

        feed_dict[self.old_means_ph] = old_means
        feed_dict[self.old_log_vars_ph] = old_log_vars

        loss, kl_pen, _ = self.sess.run([self.loss, self.kl_pen, self.train_opt], feed_dict)

        return loss, kl_pen

    def _grads(self):
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def _assign_weights_ops(self):

        self.shapes = [v.shape for v in self.weights]
        self.size = np.sum([np.prod(shape) for shape in self.shapes])

        self.flat_weights_ph = tf.placeholder(tf.float32, (self.size,))
        self.assign_weights_ops = []

        assert len(self.weights) == len(self.shapes), "Bad shape!"

        start = 0

        for i, shape in enumerate(self.shapes):
            size = np.prod(shape)
            weight = tf.reshape(self.flat_weights_ph[start:(start + size)], shape)
            self.assign_weights_ops.append(self.weights[i].assign(weight))
            start += size

        assert start == self.size, "bad size"

    def policy(self, state):
        return self.sess.run(self.sampled_action, feed_dict={self.state_ph: state})

    def predict(self, state_mb):
        return self.sess.run([self.means, self.log_vars], feed_dict={self.state_ph: state_mb})

    def calc_logprob(self, mean_action, actions, log_vars):
        logp = -0.5 * tf.reduce_sum(log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(actions - mean_action) / tf.exp(log_vars), axis=1)
        return logp

class Critic(BaseCritic):

    def __init__(self, state_dim, lr, epochs, scope="critic"):
        self.epochs = epochs
        self.sess = tf.get_default_session()

        assert self.sess is not None, "sess is None"

        with tf.variable_scope(scope):
            self._init_ph(state_dim)
            self._build_net()
            self._losses(lr)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _init_ph(self,state_dim):
        self.state_ph = tf.placeholder(tf.float32, (None,state_dim))
        self.target_ph = tf.placeholder(tf.float32, (None,))

    def _build_net(self):

        out = tf.layers.dense(self.state_ph, 128, activation=tf.nn.tanh)
        out = tf.layers.dense(out, 64, activation=tf.nn.tanh)
        out = tf.layers.dense(out, 64, activation=tf.nn.tanh)

        self.value = tf.layers.dense(out, 1, activation=None)
        self.value = tf.squeeze(self.value)

    def _losses(self, lr):
        self.loss = tf.reduce_mean(tf.square(self.target_ph - self.value))
        self.train_opt = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self, state_mb, target_mb):

        num_batches = max(state_mb.shape[0] // 256, 1)
        batch_size = state_mb.shape[0] // num_batches

        indicies = np.arange(state_mb.shape[0])
        losses = []
        # np.random.shuffle()
        for epoch in range(self.epochs):
            np.random.shuffle(indicies)
            start = 0
            for j in range(num_batches):

                end = start + batch_size

                indicies_mb = indicies[start:end]

                state_train, target_train = state_mb[indicies_mb,:], target_mb[indicies_mb]

                l, _ = self.sess.run([self.loss, self.train_opt], {self.state_ph: state_train,
                                                                   self.target_ph: target_train})

                losses.append(l)
                start += batch_size

        return np.mean(losses)

    def predict(self, state_mb):
        state_value = self.sess.run(self.value, {self.state_ph: state_mb})
        return np.squeeze(state_value)

class PPO(BaseDeepAgent):

    def __init__(self, env, sess, actor_lr, init_log_var, beta, clip_range, critic_lr, epochs, batch_size, gamma=0.99, lam=0.96, max_trajectory_steps = 10000, timesteps_per_batch=1000):

        super(PPO, self).__init__(env, sess)
        self.actor = Actor(self.s_dim, self.a_dim, actor_lr, init_log_var,beta, clip_range)
        self.critic = Critic(self.s_dim, critic_lr, epochs)

        self.gamma = gamma
        self.lam = lam
        self.env = env
        self.max_trajectory_steps = max_trajectory_steps
        self.batch_size = batch_size
        self.timesteps_per_batch = timesteps_per_batch
        self.nr_eps = 0

        self._init_tb_summaries()

    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.train_reward_var = tf.Variable(0.0, name="train_reward")

            self.actor_loss_var = tf.Variable(0.0, name="actor_loss")
            self.critic_loss_var = tf.Variable(0.0, name="critic_loss")
            self.kl_var = tf.Variable(0.0, name="kl")
            self.adv_mean_var = tf.Variable(0.0, name="adv")
            self.target_var = tf.Variable(0.0, name="target")

            summaries = [tf.summary.scalar("train_reward", self.train_reward_var),
                         tf.summary.scalar("actor_loss", self.actor_loss_var),
                         tf.summary.scalar("critic_loss", self.critic_loss_var),
                         tf.summary.scalar("KL distance", self.kl_var),
                         tf.summary.scalar("Mean advantage", self.adv_mean_var),
                         tf.summary.scalar("Mean target", self.target_var)
                         ]

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)
            self.merged = tf.summary.merge(summaries)

    def run_episodes(self, n_episodes, verbosity=0, eval_freq=10, n_interact_2_evaluate=10):

        for i in range(n_episodes):

            iter_start = time.time()

            s_b, a_b, adv_b, disc_r_b, mean_train_reward = self.run_policy()

            critic_loss = self.critic.train(s_b, disc_r_b)

            actor_loss, kl_pen = self.actor.train(s_b, a_b, adv_b)

            feed_dict = {self.actor_loss_var: actor_loss,
                         self.critic_loss_var: critic_loss,
                         self.kl_var: kl_pen,
                         self.adv_mean_var: np.mean(np.abs(adv_b)),
                         self.target_var: np.mean(np.abs(disc_r_b)),
                         self.train_reward_var:mean_train_reward
                         }

            if i % eval_freq == 0:
                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate))
                summary = self.sess.run(self.eval_summary, {self.eval_reward_var: np.mean(eval_rewards)})
                self.writer.add_summary(summary, self.nr_env_interactions)

            summary = self.sess.run(self.merged, feed_dict)
            self.writer.add_summary(summary, self.nr_env_interactions)
            self.writer.flush()

            if verbosity > 0:
                print "Epsiode %i Time: %f s Nr env int: %i Train Reward: %f" % (i, time.time() - iter_start,
                                                                                 self.nr_env_interactions,
                                                                                 mean_train_reward)

    def _run_episode(self):

        s = self.env.reset()
        state_mb, action_mb, reward_mb, unscaled_state = [], [], [], []
        is_done = False
        i_step = 0

        while not is_done and i_step < self.max_trajectory_steps:

            state_mb.append(s.ravel())

            # TODO scale
            a = self.actor.policy(s.reshape(1,-1))[0]
            s_nxt, r, is_done, _ = self.env.step(a)

            action_mb.append(a)
            reward_mb.append(r)

            s = s_nxt
            i_step += 1
            self.nr_env_interactions += 1

            # TODO is_done_mb?

        state_mb.append(s.ravel())

        return np.array(state_mb), np.array(action_mb), np.array(reward_mb).ravel(), is_done


    def train(self):
        raise NotImplementedError()

    def policy(self,state):
        return self.sess.run(self.actor.sampled_action, feed_dict={self.actor.state_ph: state})


    def run_policy(self):

        total_steps = 0

        s_b, a_b, adv_b, disc_r_b = [], [], [], []

        mean_eps_reward = 0
        nr_eps = 0

        while total_steps < self.batch_size:

            s_mb, a_mb, r_mb, is_done = self._run_episode() # is this correct handled?

            state_values = self.critic.predict(s_mb)

            td_errors = r_mb + self.gamma * state_values[1::] - state_values[:-1]

            advantages = self._estimate_GAE(td_errors, self.gamma*self.lam)

            discounted_rewards = self._estimate_GAE(r_mb, self.gamma)

            s_b.extend(s_mb[:-1])
            a_b.extend(a_mb)
            adv_b.extend(advantages)
            disc_r_b.extend(discounted_rewards)

            total_steps += len(r_mb)

            if is_done:
                mean_eps_reward = (mean_eps_reward*nr_eps + np.sum(r_mb))/(nr_eps+1.0)
                nr_eps += 1

        s_b = np.array(s_b)
        a_b = np.array(a_b)
        adv_b = np.array(adv_b)
        adv_b = (adv_b-adv_b.mean())/(adv_b.std()+1e-6)

        disc_r_b = np.array(disc_r_b)
        disc_r_b = (disc_r_b - disc_r_b.mean()) / (disc_r_b.std() + 1e-6)

        return s_b, a_b, adv_b, disc_r_b, mean_eps_reward

    def _estimate_GAE(self, td_errors, coeff):

        # x_filt_filt = lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]
        gae_values = np.zeros_like(td_errors).astype(float)
        for i in range(len(td_errors)):
            for j in range(len(td_errors) - i):
                gae_values[i] += (coeff ** j) * td_errors[j + i]

        return gae_values

    def __repr__(self):
        return "PPO"

def main(args):

    env = gym.make(args["env"])

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = PPO(env,sess, args["actor_lr"], args["init_log_var"], args["beta"], args["clip_range"], args["critic_lr"],
                args["epochs"], args["batch_size"], args["gamma"], args["lam"], args["max_trajectory_steps"], args["timesteps_per_batch"])

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(args["max_iter"], args["verbosity"])

    agent.close()

if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--critic_lr", type=float, default=1e-3)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--lam", type=float, default=0.97)

    args.add_argument("--actor_lr", type=float, default=1e-4)
    args.add_argument("--beta", type=float, default=0.1) # TODO should change adaptively...
    args.add_argument("--clip_range", type=float, default=0.5) # 0.3

    args.add_argument("--init_log_var", type=float, default=-1)


    args.add_argument("--env", type=str, default="Pendulum-v0")

    args.add_argument("--epochs", type= int, default=10)
    args.add_argument("--batch_size", type=int, default=1000)

    args.add_argument("--max_trajectory_steps", type=int, default=1000)
    args.add_argument("--timesteps_per_batch", type=int, default=1000)
    args.add_argument("--verbosity", type=int, default=1)

    args.add_argument("--max_iter", type=int, default=10000)

    args = vars(args.parse_args())

    main(args)



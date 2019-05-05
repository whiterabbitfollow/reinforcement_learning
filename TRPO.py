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

def flatgrads(loss,weights_list):
    grads = tf.gradients(loss, weights_list)
    return tf.concat([tf.reshape(g, (-1,)) for g in grads], axis=0)

class Actor(BaseActor):

    def __init__(self, env, init_log_var, scope="actor"):

        self.action_dim = env.action_space.shape
        self.state_dim = env.observation_space.shape
        self.scope = scope
        self.sess = tf.get_default_session()
        self.init_log_var = init_log_var

        assert self.sess is not None, "sess is None"

        with tf.variable_scope(self.scope):
            self._init_ph()
            self._build_net()
            self._losses()
            self._grads()
            self._assign_weights_ops()

    def _init_ph(self):

        self.action_ph = tf.placeholder(tf.float32, (None,) + self.action_dim, name="action")
        self.state_ph = tf.placeholder(tf.float32, (None,) + self.state_dim, name="state")
        self.old_means_ph = tf.placeholder(tf.float32, (None,) + self.action_dim, name="old_means")
        self.old_log_vars_ph = tf.placeholder(tf.float32, self.action_dim[0], name="old_log_vars")
        self.advantage_ph = tf.placeholder(tf.float32, (None,), name="advantage")

    def _build_net(self):

            out = tf.layers.dense(self.state_ph, 64, activation=tf.tanh)
            out = tf.layers.dense(out, 64, activation=tf.tanh)

            self.means = tf.layers.dense(out, self.action_dim[0], activation=None)
            self.log_vars = tf.get_variable("logvars", self.action_dim[0], tf.float32, tf.constant_initializer(0.0)) + self.init_log_var

    def _losses(self):

            self.logp = self.calc_logprob(self.means, self.action_ph, self.log_vars)
            self.logp_old = self.calc_logprob(self.old_means_ph, self.action_ph, self.old_log_vars_ph)

            log_det_cov_new = tf.reduce_sum(self.log_vars)


            log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)

            tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

            means_n_cov = tf.square(self.means-self.old_means_ph)/tf.exp(self.log_vars)

            log_det_cov_new_old = log_det_cov_new - log_det_cov_old

            k = self.action_dim[0]

            self.kl = tf.reduce_sum(tr_old_new + means_n_cov - k + log_det_cov_new_old) * 0.5

            self.entropy = 0.5 * (k + np.log(2*np.pi)*k + log_det_cov_new)

            self.surrogate_loss = -tf.reduce_mean(tf.exp(self.logp - self.logp_old) * self.advantage_ph) # obs sign!!
            self.kl_pen = tf.reduce_mean(self.kl)
            self.sampled_action = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=self.action_dim, dtype=tf.float32))  # TODO: read about this


    def train(self):
        raise NotImplementedError()

    def _grads(self):

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

        # policy gradients
        self.pg = flatgrads(self.surrogate_loss, self.weights) # obs sign!!

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

    def policy_gradient(self, state_mb, action_mb, advantage_mb, old_means_mb, old_log_vars_mb):

        feed_dict = {self.state_ph: state_mb,
                     self.action_ph: action_mb,
                     self.advantage_ph: advantage_mb,
                     self.old_means_ph: old_means_mb,
                     self.old_log_vars_ph: old_log_vars_mb
                     }

        return self.sess.run(self.pg, feed_dict)

    def calc_logprob(self, mean_action, actions, log_vars):
        logp = -0.5 * tf.reduce_sum(log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(actions - mean_action) / tf.exp(log_vars), axis=1) # sign???
        # missing ... k * ln(2*pi)
        return logp

class TRPO_updater:

    def __init__(self, actor, delta, cg_damping):

        # hessian vector product
        # self.action_dim = action_dim
        # self.state_dim = state_dim

        self.delta = delta
        self.actor = actor
        self.shapes = self.actor.shapes
        self.sess = tf.get_default_session()
        self.cg_damping = cg_damping

        self._compute_hessian_vector_product()

        self.flat_weights = tf.concat([tf.reshape(weight, (-1,)) for weight in self.actor.weights], axis=0)

    def _compute_hessian_vector_product(self):

        self.p = tf.placeholder(tf.float32, self.actor.size)

        grads = tf.gradients(self.actor.kl_pen, self.actor.weights)
        tangents = []
        start = 0

        for shape in self.shapes:

            size = np.prod(shape)
            tangents.append(tf.reshape(self.p[start:(start+size)], shape))
            start += size

        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zip(grads, tangents)])  # TODO: could just use sum

        self.hvp = flatgrads(gvp, self.actor.weights)

    def get_flat_weights(self):
        return self.sess.run(self.flat_weights)

    def assign_vars(self, weights):
        self.sess.run(self.actor.assign_weights_ops,feed_dict={self.actor.flat_weights_ph: weights})

    def __call__(self, state_mb, action_mb, advantage_mb):

        feed_dict = {self.actor.state_ph: state_mb,
                     self.actor.action_ph: action_mb,
                     self.actor.advantage_ph: advantage_mb
                     }

        old_means, old_log_vars = self.actor.predict(state_mb)

        feed_dict[self.actor.old_means_ph] = old_means
        feed_dict[self.actor.old_log_vars_ph] = old_log_vars

        old_weights = self.get_flat_weights()

        def get_hvp(p):
            feed_dict[self.p] = p
            return self.sess.run(self.hvp, feed_dict) + self.cg_damping * p  # TODO: add cg_damping...

        def get_loss_kl(weights):
            self.assign_vars(weights)
            return self.sess.run([self.actor.surrogate_loss, self.actor.kl], feed_dict)

        def get_pg():
            return self.sess.run(self.actor.pg, feed_dict)

        pg = get_pg()

        if np.allclose(pg, 0):
            print "Got zero Gradient. Not updating... "
            return 0

        # solves x = F^-1 * pg
        # need to negate pg since minus sig in first place..

        # get_hvp calculates F * pg -> F * J(theta)
        step_unscaled = conjugate_gradient_method(b=-pg, A_dot_x=get_hvp)

        # step unscaled is now F^-1*J(theta)

        step = np.sqrt((2*self.delta)/(step_unscaled.dot(get_hvp(step_unscaled)))) * step_unscaled

        #-pg due to that we have already negated pg in first place

        weights, loss, kl, step_frac = line_search(old_weights, get_loss_kl, step, -pg.dot(step), self.delta)

        get_loss_kl(weights)

        return loss, kl, step_frac

def line_search(weights, func, full_step, expected_improve_rate, delta, max_backtracks=10, accept_ratio=0.1):

    loss, kl = func(weights)

    for step_frac in (0.5**np.arange(max_backtracks)):

        weights_new = weights + step_frac * full_step # this direction will maximize...

        loss_new, kl_new = func(weights_new)

        if kl_new > delta:
            loss_new += np.inf

        actual_improvement = loss - loss_new # wrong sign??
        expected_improvement = expected_improve_rate*step_frac
        ratio = actual_improvement/expected_improvement

        if ratio > accept_ratio and actual_improvement > 0:
            return weights_new, loss_new, kl_new, step_frac

    return weights, loss, kl, -1


def conjugate_gradient_method(b, A_dot_x, nr_iters=10, tol=1e-4):

    x = np.zeros_like(b)  # initial guess
    r = b.copy()
    p = r

    for i in range(nr_iters):

        A_dot_p = A_dot_x(p)
        rT_dot_r = r.T.dot(r)
        alfa = rT_dot_r/p.T.dot(A_dot_p)  # how far should move in direction p

        x_new = x + alfa * p # next point
        r_new = r - alfa * A_dot_p # remaining error from the optimal point

        r_newT_dot_r_new = r_new.T.dot(r_new)

        if r_newT_dot_r_new < tol:
            x = x_new
            break

        beta = r_newT_dot_r_new/rT_dot_r

        p = r_new + beta*p
        r = r_new
        x = x_new

    return x

class Critic(BaseCritic):

    def __init__(self, env, lr, epochs, scope="critic"):

        self.state_dim = env.observation_space.shape
        self.lr = lr
        self.epochs = epochs
        self.sess = tf.get_default_session()

        assert self.sess is not None, "sess is None"

        with tf.variable_scope(scope):
            self._init_ph()
            self._build_net()
            self._losses()

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _init_ph(self):
        self.state_ph = tf.placeholder(tf.float32, (None,) + self.state_dim)
        self.target_ph = tf.placeholder(tf.float32, (None,))

    def _build_net(self):
        out = tf.layers.dense(self.state_ph, 128, activation=tf.nn.tanh)
        out = tf.layers.dense(out, 64, activation=tf.nn.tanh)
        out = tf.layers.dense(out, 64, activation=tf.nn.tanh)
        self.value = tf.layers.dense(out, 1, activation=None)
        self.value = tf.squeeze(self.value)

    def _losses(self):
        self.loss = tf.reduce_mean(tf.square(self.target_ph - self.value))
        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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

class TRPO(BaseDeepAgent):

    def __init__(self, env, sess, init_log_var,critic_lr, epochs, batch_size, cg_damping, delta, gamma=0.99, lam=0.96, max_trajectory_steps = 10000, timesteps_per_batch=1000):

        super(TRPO, self).__init__(env, sess)

        self.actor = Actor(env, init_log_var)
        self.critic = Critic(env, critic_lr, epochs)
        self.trpo_updater = TRPO_updater(self.actor, delta, cg_damping)
        self.scalar = None # TODO fix this
        self.gamma = gamma
        self.lam = lam
        self.env = env
        self.max_trajectory_steps = max_trajectory_steps
        self.batch_size = batch_size
        self.timesteps_per_batch = timesteps_per_batch

        self._init_tb_summaries()


    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.train_reward_var = tf.Variable(0.0, name="train_reward")

            self.actor_loss_var = tf.Variable(0.0, name="actor_loss")
            self.critic_loss_var = tf.Variable(0.0, name="critic_loss")
            self.step_frac_var = tf.Variable(0.0, name="step_frac")
            self.kl_var = tf.Variable(0.0, name="kl")
            self.adv_mean_var = tf.Variable(0.0, name="adv")
            self.target_var = tf.Variable(0.0, name="target")

            summaries = [tf.summary.scalar("train_reward", self.train_reward_var),
                         tf.summary.scalar("actor_loss", self.actor_loss_var),
                         tf.summary.scalar("critic_loss", self.critic_loss_var),
                         tf.summary.scalar("KL distance", self.kl_var),
                         tf.summary.scalar("step frac", self.step_frac_var),
                         tf.summary.scalar("Mean advantage", self.adv_mean_var),
                         tf.summary.scalar("Mean target", self.target_var)
                         ]

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)
            self.merged = tf.summary.merge(summaries)

    def run_episodes(self, n_episodes,eval_freq=10, verbosity=0, n_interact_2_evaluate=10):

        for i in range(n_episodes):

            iter_start = time.time()

            s_b, a_b, adv_b, disc_r_b, train_reward = self.run_policy()

            critic_loss = self.critic.train(s_b, disc_r_b)

            actor_loss, kl, step_frac = self.trpo_updater(s_b, a_b, adv_b)

            feed_dict = {self.train_reward_var: train_reward,
                         self.actor_loss_var: actor_loss,
                         self.critic_loss_var: critic_loss,
                         self.kl_var: kl,
                         self.step_frac_var: step_frac,
                         self.adv_mean_var: np.mean(np.abs(adv_b)),
                         self.target_var: np.mean(np.abs(disc_r_b))
                         }

            if i % eval_freq == 0:

                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate))
                summary = self.sess.run(self.eval_summary, {self.eval_reward_var: np.mean(eval_rewards)})
                self.writer.add_summary(summary, self.nr_env_interactions)

            summary = self.sess.run(self.merged, feed_dict)

            self.writer.add_summary(summary, self.nr_env_interactions)
            self.writer.flush()

            if verbosity > 0:
                print "Epsiode %i Time: %f s Nr env interactions: %i Reward: %f" % (i, time.time() - iter_start,self.nr_env_interactions, train_reward)

    def _run_episode(self):

        s = self.env.reset()
        state_mb, action_mb, reward_mb, unscaled_state = [], [], [], []
        is_done = False
        i_step = 0
        acc_reward = 0.0

        while not is_done and i_step < self.max_trajectory_steps:

            state_mb.append(s.ravel())

            # TODO scale
            a = self.actor.policy(s.reshape(1,-1))[0]
            s_nxt, r, is_done, _ = self.env.step(a)

            action_mb.append(a)
            reward_mb.append(r)

            s = s_nxt
            acc_reward += r
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
        return "TRPO"

def main(args):

    env = gym.make(args["env"])

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = TRPO(env,sess,args["init_log_var"], args["critic_lr"], args["epochs"], args["batch_size"], args["cg_damping"],
                 args["delta"], args["gamma"], args["lam"], args["max_trajectory_steps"], args["timesteps_per_batch"])

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(args["max_iter"], args["verbosity"])

    agent.close()

if __name__ == "__main__":

    args = argparse.ArgumentParser()

    # args.add_argument("--critic_lr", type=float, default=1e-4)
    args.add_argument("--critic_lr", type=float, default=1e-3)
    args.add_argument("--delta", type=float, default=0.05)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--lam", type=float, default=0.97)
    args.add_argument("--cg_damping", type=float, default=0.1)
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

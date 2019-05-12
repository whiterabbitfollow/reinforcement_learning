#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:43:42 2018
@author: x

"""

import gym 
import tensorflow as tf
import numpy as np
from utils.utils import evaluate_agent, BaseDeepAgent, EnvBatch, BaseActor
import argparse

class ActorNetwork(BaseActor):

    def __init__(self, sess, states_ph, actions_ph, action_out, reuse=False, is_discrete_action=True, name="ActorNetwork"):

        self.init_log_var = -1 # TODO add config
        self.actions_ph = actions_ph
        self.states_ph = states_ph
        self.is_discrete_action = is_discrete_action
        self.sess = sess
        self.actions_out = action_out

        with tf.variable_scope(name, reuse=reuse):

            if not is_discrete_action:

                net = tf.layers.dense(states_ph, 128, activation=tf.nn.tanh, name="fcc1")
                self.means = tf.layers.dense(net, action_out, activation=None)
                self.log_vars = tf.get_variable("logvars", action_out, tf.float32, tf.constant_initializer(0.0)) + self.init_log_var

                self._policy = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(action_out,),
                                                                                            dtype=tf.float32))  # TODO: read about this

                self.neg_log_prob = -self.calc_logprob(self.means, self.actions_ph, self.log_vars)

            else:

                net = tf.layers.dense(states_ph, 128, activation=tf.nn.relu, name="fcc1")
                self.logits = tf.layers.dense(net, action_out, activation=None)

                self.actions_one_hot_enc = tf.one_hot(self.actions_ph, action_out)
                self._policy = tf.nn.softmax(self.logits)
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                                     labels=self.actions_one_hot_enc)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def policy(self,states):

        if self.is_discrete_action:
            action_probs = self.sess.run(self._policy,{self.states_ph:states})
            actions = [np.random.choice(range(self.actions_out), p=prob) for prob in action_probs]

        else:
            actions = self.sess.run(self._policy, {self.states_ph: states})

        return actions

    def train(self):
        raise NotImplementedError()

    def calc_logprob(self, mean_action, actions, log_vars):
        k = self.actions_out
        logp = -0.5 * tf.reduce_sum(log_vars) - k * np.log(2*np.pi) # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Properties
        logp += -0.5 * tf.reduce_sum(tf.square(actions - mean_action) / tf.exp(log_vars), axis=1)
        return logp

class CriticNetwork():

    def __init__(self,states_ph,reuse=False,name="CriticNetwork"):

        with tf.variable_scope(name,reuse=reuse):

            net = tf.layers.dense(states_ph,128,activation=tf.nn.relu, name="fcc1")
            self.state_value = tf.layers.dense(net,1,activation= None, name="out")
            self.mean_state_value = tf.reduce_mean(self.state_value)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

class AdvantageActorCritic(BaseDeepAgent):

    def __init__(self, env, sess, tau, gamma, actor_lr, critic_lr, nr_envs, use_target=False,n_steps_update=10, N=1):

        super(AdvantageActorCritic, self).__init__(env,sess)

        self.n_steps_update = n_steps_update
        self.N = N
        self.use_target = use_target
        self.gamma = gamma
        self.tau = tau
        self.sess = sess
        self.env_batch = EnvBatch(self.env_name, n_envs=nr_envs)

        self._init_ph()
        self._init_nets(use_target)
        self._intit_losses_n_train_opts(actor_lr,critic_lr)
        self._init_tb_summaries()

    def _init_ph(self):

        self.states_ph = tf.placeholder(tf.float32, (None, self.s_dim),name="state")

        if self.is_discrete:
            self.actions_ph = tf.placeholder(tf.int32, (None,), name="action")
        else:
            self.actions_ph = tf.placeholder(tf.float32, (None,self.a_dim), name="action")

        self.A_values = tf.placeholder(tf.float32, (None,),name="advantage")
        self.target = tf.placeholder(tf.float32, (None, 1), name="target_state_values")

    def _init_nets(self, use_target):

        # sess, states_ph, actions_ph, action_out, reuse = False, is_discrete_action = True, name = "ActorNetwork"
        if self.is_discrete:
            action_out = self.n_actions
        else:
            action_out = self.a_dim

        self.actor = ActorNetwork(self.sess,self.states_ph, self.actions_ph, action_out,is_discrete_action=self.is_discrete, name="actor")

        self.critic = CriticNetwork(self.states_ph, name="critic")

        if use_target:
            self.target_critic = CriticNetwork(self.states_ph, name="target_critic")
        else:
            self.target_critic = None

    def _intit_losses_n_train_opts(self,actor_lr,critic_lr):

        self.actor.loss = tf.reduce_mean(self.actor.neg_log_prob * self.A_values)
        self.actor.train_opt = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.actor.loss)

        self.critic.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic.state_value)
        self.critic.train_opt = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic.loss)

        # self.grads = tf.gradients(self.actor.loss, self.actor.weights)

    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.actor_loss_var = tf.Variable(0.0, name="actor_loss")
            self.critic_loss_var = tf.Variable(0.0, name="critic_loss")
            self.mean_pred_state_value_var = tf.Variable(0.0, name="mean_pred_state_value")

            summaries = [tf.summary.scalar("actor_loss", self.actor_loss_var),
                     tf.summary.scalar("critic_loss", self.critic_loss_var),
                     tf.summary.scalar("mean_pred_state_value", self.mean_pred_state_value_var)
                     ]

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)

            self.merged = tf.summary.merge(summaries)

    def train(self,s_mb,r_mb,a_mb,d_mb,nxt_s_mb):

        state_values = self.sess.run(self.critic.state_value,{self.states_ph:s_mb}).ravel()

        if self.target_critic is None:
            nxt_state_values = self.sess.run(self.critic.state_value, {self.states_ph: nxt_s_mb}).ravel()
        else:
            nxt_state_values = self.sess.run(self.target_critic.state_value,{self.states_ph:nxt_s_mb}).ravel()


        if self.N==1:
            target = r_mb + self.gamma * (1.0 - d_mb) * nxt_state_values
        else:
            target = r_mb + self.gamma * (1.0 - d_mb) * nxt_state_values


        advantage = target - state_values

        actor_loss, _ = self.sess.run([self.actor.loss,self.actor.train_opt],
                                     {self.states_ph:s_mb,self.actions_ph:a_mb,self.A_values:advantage})

        critic_loss, _, mean_pred_state_value = self.sess.run([self.critic.loss, self.critic.train_opt,self.critic.mean_state_value],
                                       {self.states_ph: s_mb, self.target: target.reshape(-1,1)})

        return actor_loss, critic_loss, mean_pred_state_value

    def policy(self, states):
        return self.actor.policy(states)


    def run_episodes(self,n_episodes,verbosity=0,eval_freq=10, n_interact_2_evaluate=10):

        states_mb = self.env_batch.reset()

        self.copy_network_parameters(tau=0.0) # why?

        for i in range(n_episodes):

            actions_batch = self.policy(states_mb)

            nxt_states_mb, rewards_mb, done_mb = self.env_batch.step(actions_batch)

            critic_loss, actor_loss, mean_pred_state_value = self.train(states_mb, rewards_mb, actions_batch, done_mb, nxt_states_mb)

            self.nr_env_interactions += len(rewards_mb)

            states_mb = nxt_states_mb

            if i % eval_freq == 0:

                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate))
                summary = self.sess.run(self.eval_summary, {self.eval_reward_var:np.mean(eval_rewards)})
                self.writer.add_summary(summary, self.nr_env_interactions)

                if verbosity > 0:
                    print "Episode %i Nr env interactions: %i Eval reward: %f" % (i, self.nr_env_interactions, eval_rewards)

            feed_dict = {self.critic_loss_var: critic_loss,
                         self.actor_loss_var: actor_loss,
                         self.mean_pred_state_value_var: mean_pred_state_value}

            summary = self.sess.run(self.merged,feed_dict)

            self.writer.add_summary(summary, self.nr_env_interactions)
            self.writer.flush()

            if self.use_target and i % self.n_steps_update == 0:
                self.copy_network_parameters()



    def copy_network_parameters(self,tau=None):

            if self.target_critic is None:
                return

            if tau is None:
                tau = self.tau

            assigns = []

            for w_reference, w_target in zip(self.critic.weights, self.target_critic.weights):

                val = tf.multiply(w_target, tau) + tf.multiply(w_reference, 1. - tau)
                assigns.append(tf.assign(w_target, val, validate_shape=True))

            self.sess.run(assigns)

    def __repr__(self):
        return "A2C"

def main(args):

    env = gym.make(args["env"])

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    agent = AdvantageActorCritic(env,sess,args["tau"],args["gamma"],args["actor_learning_rate"],
                                 args["critic_learning_rate"],
                                 args["nr_envs"],
                                 args["use_target_critic"],
                                 args["n_steps_update"])

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(n_episodes=args["max_nr_iter"],
                       n_interact_2_evaluate=args["n_interact_2_evaluate"],
                       eval_freq=args["eval_freq"],
                       verbosity=args["verbosity"]
                       )

    agent.close()


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--env", type=str, default="MountainCar-v0")
    args.add_argument("--max_nr_iter",type=int,default=100000)
    args.add_argument("--verbosity", type=int, default=1)
    args.add_argument("--n_interact_2_evaluate", type=int, default=100)
    args.add_argument("--eval_freq", type=int, default=100)
    args.add_argument("--tau",type=float,default=0.05)
    args.add_argument("--gamma",type=float,default=0.99)
    args.add_argument("--actor_learning_rate",type=float,default=0.001)
    args.add_argument("--critic_learning_rate",type=float,default=0.001)
    args.add_argument("--nr_envs",type=int,default=10)
    args.add_argument("--n_steps_update",type=int,default=20)
    args.add_argument("--use_target_critic", action="store_true")

    main(vars(args.parse_args()))





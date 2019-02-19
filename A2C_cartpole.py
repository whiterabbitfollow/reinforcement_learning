#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:43:42 2018
@author: x

"""

import gym 
import tensorflow as tf
import numpy as np
from utils import get_run_nr, evaluate_agent, BaseAgent, EnvBatch
import argparse

class ActorNetwork():

    def __init__(self, states_ph, n_actions, reuse=False, name="ActorNetwork"):
        with tf.variable_scope(name, reuse=reuse):

            net = tf.layers.dense(states_ph,128, activation=tf.nn.relu,name="fcc1")
            self.logits = tf.layers.dense(net, n_actions, activation = None)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

class CriticNetwork():

    def __init__(self,states_ph,reuse=False,name="CriticNetwork"):

        with tf.variable_scope(name,reuse=reuse):

            net = tf.layers.dense(states_ph,128,activation=tf.nn.relu, name="fcc1")
            self.state_value = tf.layers.dense(net,1,activation= None, name="out")
            self.mean_state_value = tf.reduce_mean(self.state_value)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

class AdvantageActorCritic(BaseAgent):

    def __init__(self, states_dim, n_actions, tau, gamma, actor_lr, critic_lr, use_target=False):

        super(BaseAgent, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.sess = tf.get_default_session()

        self.states_ph = tf.placeholder(tf.float32, (None, states_dim),name="state")
        self.actions_ph = tf.placeholder(tf.int32, (None),name="action")
        self.A_values = tf.placeholder(tf.float32, (None,),name="advantage")

        actions_one_hot_enc = tf.one_hot(self.actions_ph, n_actions)

        self.target = tf.placeholder(tf.float32, (None, 1),name="target_state_values")

        self.actor = ActorNetwork(self.states_ph,n_actions, name="actor")
        self.critic = CriticNetwork(self.states_ph, name="critic")

        if use_target:
            self.target_critic = CriticNetwork(self.states_ph, name="target_critic")
        else:
            self.target_critic = None

        self.actor_policy = tf.nn.softmax(self.actor.logits)
        self.actor_neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actor.logits, labels=actions_one_hot_enc)

        self.actor.loss = tf.reduce_mean(self.actor_neg_log_prob * self.A_values)
        self.actor.train_opt = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.actor.loss)

        self.critic.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic.state_value)
        self.critic.train_opt = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic.loss)

        # self.grads = tf.gradients(self.actor.loss, self.actor.weights)

    def train(self,s_mb,r_mb,a_mb,d_mb,nxt_s_mb):

        state_values = self.sess.run(self.critic.state_value,{self.states_ph:s_mb}).ravel()

        if self.target_critic is None:
            nxt_state_values = self.sess.run(self.critic.state_value, {self.states_ph: nxt_s_mb}).ravel()
        else:
            nxt_state_values = self.sess.run(self.target_critic.state_value,{self.states_ph:nxt_s_mb}).ravel()

        target = r_mb + self.gamma * (1.0 - d_mb) * nxt_state_values

        advantage = target - state_values

        actor_loss,_ = self.sess.run([self.actor.loss,self.actor.train_opt],
                                     {self.states_ph:s_mb,self.actions_ph:a_mb,self.A_values:advantage})

        critic_loss, _, mean_pred_state_value = self.sess.run([self.critic.loss, self.critic.train_opt,self.critic.mean_state_value],
                                       {self.states_ph: s_mb, self.target: target.reshape(-1,1)})

        return actor_loss, critic_loss, mean_pred_state_value

    def policy(self,states):
        action_probs = self.sess.run(self.actor_policy,{self.states_ph:states})
        return [ np.random.choice(range(self.n_actions),p=prob) for prob in action_probs ]

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

def main(args):

    env = gym.make("CartPole-v0")
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    n_interact_2_evaluate = 100
    n_steps_update = args["gamma"]
    max_nr_iter = args["max_nr_iter"]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.variable_scope("summaries"):

        mean_reward_var = tf.Variable(0.0,name="mean_reward")
        mean_reward_100_var = tf.Variable(0.0,name="mean_reward_100")

        actor_loss_var = tf.Variable(0.0,name="actor_loss")
        critic_loss_var = tf.Variable(0.0,name="critic_loss")
        mean_pred_state_value_var = tf.Variable(0.0,name="mean_pred_state_value")


    agent = AdvantageActorCritic(n_states,n_actions,args["tau"],args["gamma"],args["actor_learning_rate"],args["critic_learning_rate"],args["use_target_critic"])

    rewards_summaries = [tf.summary.scalar("mean_reward",mean_reward_var),
                         tf.summary.scalar("mean_reward_100",mean_reward_100_var)]
    loss_summaries = [ tf.summary.scalar("actor_loss",actor_loss_var),tf.summary.scalar("critic_loss",critic_loss_var),
                       tf.summary.scalar("mean_pred_state_value",mean_pred_state_value_var)]

    sess.run(tf.global_variables_initializer())

    run_nr = get_run_nr("runs",starts_with="A2C")
    writer = tf.summary.FileWriter("./runs/A2C_"+str(run_nr),sess.graph)

    merged = tf.summary.merge_all()
    loss_merged = tf.summary.merge(loss_summaries)
    env_batch = EnvBatch("CartPole-v0",n_envs=args["nr_envs"])

    states_mb = env_batch.reset()

    rewards_history = []

    agent.copy_network_parameters(tau=0.0)

    for i in range(max_nr_iter):

        actions_batch = agent.policy(states_mb)

        nxt_states_mb, rewards_mb, done_mb = env_batch.step(actions_batch)

        critic_loss, actor_loss, mean_pred_state_value = agent.train(states_mb,rewards_mb,actions_batch,done_mb,nxt_states_mb)

        states_mb = nxt_states_mb

        if i%500==0 or i==0:

            rewards_history.append(np.mean(evaluate_agent(agent, env, n_games=n_interact_2_evaluate)))
            nr_steps = min(100,len(rewards_history))

            summary = sess.run(merged,{critic_loss_var:critic_loss,
                actor_loss_var:actor_loss,
                mean_reward_var:rewards_history[-1],
                mean_reward_100_var:np.mean(rewards_history[-nr_steps:]),
                                       mean_pred_state_value_var:mean_pred_state_value})

            writer.add_summary(summary,i)

        else:
            summary = sess.run(loss_merged,{critic_loss_var:critic_loss,
                                            actor_loss_var:actor_loss,
                                            mean_pred_state_value_var:mean_pred_state_value})
            writer.add_summary(summary,i)

        if rewards_history[-1] > 195:
            print "Solved!"
            break

        if i%n_steps_update==0:
            agent.copy_network_parameters()

        writer.flush()

    sess.close()
    writer.close()


if __name__== "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--max_nr_iter",type=int,default=100000)
    args.add_argument("--tau",type=float,default=0.05)
    args.add_argument("--gamma",type=float,default=0.99)
    args.add_argument("--actor_learning_rate",type=float,default=0.001)
    args.add_argument("--critic_learning_rate",type=float,default=0.001)
    args.add_argument("--nr_envs",type=int,default=10)
    args.add_argument("--n_steps_update",type=int,default=20)
    args.add_argument("--use_target_critic",action="store_true")

    main(vars(args.parse_args()))





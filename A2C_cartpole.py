#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:43:42 2018

@author: x
"""

import gym 
import tensorflow as tf
import numpy as np
from utils import get_run_nr
import argparse

class ActorNetwork(object):

    def __init__(self,states_dim,n_actions,learning_rate=0.01,reuse=False,name="ActorNetwork"):

        self.n_actions = n_actions

        with tf.variable_scope(name,reuse=reuse):

            self.states_input = tf.placeholder(tf.float32,(None,states_dim))
            self.actions_input = tf.placeholder(tf.int32,(None))
            self.A_values = tf.placeholder(tf.float32,(None,))
            net = tf.layers.dense(self.states_input,24,activation=tf.nn.relu)
            self.out = tf.layers.dense(net,n_actions,activation=None)

            self.policy = tf.nn.softmax(self.out)

            actions_one_hot_enc = tf.one_hot(self.actions_input,n_actions)

            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out,labels=actions_one_hot_enc)
            self.loss = tf.reduce_mean(self.neg_log_prob*self.A_values)
            self.train_opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(self.loss)

    def train(self,states,actions,advantages):
        sess = tf.get_default_session()
        return sess.run([self.loss,self.train_opt],feed_dict={self.states_input:states,self.actions_input:actions,self.A_values:advantages})

    def sample(self,states):
        sess = tf.get_default_session()
        action_probs = sess.run(self.policy,{self.states_input:states})
        return [ np.random.choice(range(self.n_actions),p=prob) for prob in action_probs ]

class CriticNetwork(object):

    def __init__(self,states_dim,learning_rate=0.001,reuse=False,name="CriticNetwork"):

        with tf.variable_scope(name,reuse=reuse):

            self.states_input = tf.placeholder(tf.float32,(None,states_dim))
            self.target = tf.placeholder(tf.float32,(None,1)) 

            net = tf.layers.dense(self.states_input,24,activation=tf.nn.relu,name="fcc1")            
            self.out = tf.layers.dense(net,1,activation=None,name="out")

            self.loss = tf.losses.mean_squared_error(labels=self.target,predictions=self.out)

            self.train_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name)

    def predict(self,states):
        sess = tf.get_default_session()
        return sess.run(self.out,{self.states_input:states})

    def train(self,states,targets):
        sess = tf.get_default_session()
        return sess.run([self.loss,self.train_opt],{self.states_input:states,self.target:targets})




def copy_network_parameters(reference,target,tau=0.95):

    sess = tf.get_default_session()
    assigns = []

    for w_reference,w_target in zip(reference.weights,target.weights):
        assigns.append(tf.assign(w_target,tf.multiply(w_target,1.-tau) + tf.multiply(w_reference,tau),validate_shape=True))

    sess.run(assigns)

class EnvBatch(object):
    def __init__(self,name,n_envs=10):
        self.envs_ = [ gym.make(name) for i in range(n_envs)]

    def reset(self):
        return np.array([ env.reset() for env in self.envs_])

    def step(self,actions):
        self.results_ = [env.step(action) for action, env in zip(actions,self.envs_) ]
        nxt_state, rewards, done, _ = map(np.array,zip(*self.results_))
        for i,d in enumerate(done):
            if d:
                nxt_state[i,:] = self.envs_[i].reset()
        return nxt_state, rewards, done

def evaluate(agent, env, n_games=1):

    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.sample([state])[0]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done: break
        game_rewards.append(total_reward)
    return game_rewards


def main(max_nr_episodes,gamma,nr_envs,n_steps_update,tau,actor_learning_rate,critic_learning_rate):

    env = gym.make("CartPole-v0")

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    with tf.variable_scope("summaries"):

        mean_reward_var = tf.Variable(0.0,name="mean_reward")
        mean_reward_100_var = tf.Variable(0.0,name="mean_reward_100")

        actor_loss_var = tf.Variable(0.0,name="actor_loss")
        critic_loss_var = tf.Variable(0.0,name="critic_loss")

    actor = ActorNetwork(states_dim=n_states,n_actions=n_actions,learning_rate = actor_learning_rate,name="actor")
    critic = CriticNetwork(states_dim=n_states,learning_rate = critic_learning_rate,name="critic" )
    critic_target = CriticNetwork(states_dim=n_states,learning_rate = critic_learning_rate,name="critic_target")


    rewards_summaries = [tf.summary.scalar("mean_reward",mean_reward_var),tf.summary.scalar("mean_reward_100",mean_reward_100_var)]
    loss_summaries = [ tf.summary.scalar("actor_loss",actor_loss_var),tf.summary.scalar("critic_loss",critic_loss_var)]

    sess.run(tf.global_variables_initializer())

    n_interact_2_evaluate = 100

    run_nr = get_run_nr("runs",starts_with="A2C")
    writer = tf.summary.FileWriter("./runs/A2C_"+str(run_nr),sess.graph)

    merged = tf.summary.merge_all()
    loss_merged = tf.summary.merge(loss_summaries)
    env_batch = EnvBatch("CartPole-v0",n_envs=nr_envs)

    states_batch = env_batch.reset()

    rewards_history = []

    copy_network_parameters(critic,critic_target,tau=1.0)

    for w_reference, w_target in zip(critic.weights,critic_target.weights):
        sess.run(tf.assert_equal(w_target,w_reference)) 


    for i in range(max_nr_episodes):

        actions_batch = actor.sample(states_batch)

        nxt_states_batch, rewards_batch, done_batch = env_batch.step(actions_batch)

        state_values_batch = critic.predict(states_batch).ravel()
        nxt_state_values_batch = critic_target.predict(nxt_states_batch).ravel()

        target = rewards_batch + gamma*(1-done_batch)*nxt_state_values_batch

        advantage = (target - state_values_batch)

        actor_loss, _ = actor.train(states_batch,actions_batch,advantage)
        critic_loss, _ = critic.train(states_batch,target.reshape(-1,1))

        states_batch = nxt_states_batch

        if i%500==0 or i==0:

            rewards_history.append(np.mean(evaluate(actor, env, n_games=n_interact_2_evaluate)))
            nr_steps = min(100,len(rewards_history))

            summary = sess.run(merged,{critic_loss_var:critic_loss,
                actor_loss_var:actor_loss,
                mean_reward_var:rewards_history[-1],
                mean_reward_100_var:np.mean(rewards_history[-nr_steps:])})

            writer.add_summary(summary,i)

        else:
            summary = sess.run(loss_merged,{critic_loss_var:critic_loss,actor_loss_var:actor_loss})
            writer.add_summary(summary,i)

        if rewards_history[-1]>195:
            print "Solved!"
            break

        if i%n_steps_update==0:
            copy_network_parameters(critic,critic_target)

        writer.flush()

    sess.close()
    writer.close()


if __name__=="__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--max_nr_episodes",type=int,default=100000)
    args.add_argument("--tau",type=float,default=0.95)
    args.add_argument("--gamma",type=float,default=0.99)
    args.add_argument("--actor_learning_rate",type=float,default=0.001)
    args.add_argument("--critic_learning_rate",type=float,default=0.001)
    args.add_argument("--nr_envs",type=int,default=20)
    args.add_argument("--n_steps_update",type=int,default=20)


    main(**vars(args.parse_args()))





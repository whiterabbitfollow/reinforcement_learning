#!/usr/bin/python

import tensorflow as tf
import tensorboard
import numpy as np
import gym
import argparse
import os 

class PolicyGradientAgent():

    def __init__(self,state_dim,n_actions,learning_rate,nr_envs,entropy_coeff):

        self.n_actions = n_actions

        sess = tf.get_default_session()

        self.states_ph = tf.placeholder(tf.float32,(None,state_dim),name="states")
        self.actions_ph = tf.placeholder(tf.int32,(None,),name="actions")
        self.return_ph = tf.placeholder(tf.float32,(None,),name="dicounted_returns")
        self.rewards_ph = tf.placeholder(tf.float32,(None,),name="rewards")

        with tf.variable_scope("PGAgent"):

            net = tf.layers.dense(self.states_ph,32,activation=tf.nn.relu)
            net = tf.layers.dense(self.states_ph,32,activation=tf.nn.relu)
            self.logits = tf.layers.dense(net,n_actions,activation=None)

            self.log_probs = tf.nn.log_softmax(self.logits)
            self.probs = tf.nn.softmax(self.logits)
            self.neg_log_probs = - tf.reduce_sum(self.log_probs * tf.one_hot(self.actions_ph,depth=n_actions),axis=-1)

            self.entropy = -tf.reduce_sum(self.log_probs * self.probs,axis=-1)
            self.loss =  tf.reduce_mean(self.neg_log_probs*self.return_ph)  - entropy_coeff * tf.reduce_mean(self.entropy)
            self.train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

            tf.summary.scalar("loss",self.loss)
            tf.summary.scalar("reward_sum",tf.reduce_sum(self.rewards_ph)/nr_envs)
            tf.summary.scalar("mean_discounted_return",tf.reduce_mean(self.return_ph))
    
    def sample_action(self,states):

        sess = tf.get_default_session()

        logits_actions = sess.run(self.logits,feed_dict={self.states_ph:states}).tolist()
        probs_actions = []

        for logit in logits_actions:

            logits_exp = np.exp(logits_actions)
            probs_actions.append((logits_exp/np.sum(logits_exp)).ravel())

        return [ np.random.choice(self.n_actions,p=prob) for prob in probs_actions ]

    def train(self,states,actions,discounted_rewards):

        sess = tf.get_default_session()
        loss, _ = sess.run([self.loss,self.train_opt],feed_dict={self.states_ph:states,self.actions_ph:actions,self.return_ph:discounted_rewards})

        return loss


def compute_discounted_return(rewards,gamma=0.95,baseline=False):

    discounted_rewards = []
    acc_r = 0.0

    for r in reversed(rewards):
        acc_r = r + gamma*acc_r
        discounted_rewards.append(acc_r)

    if baseline:
        discounted_rewards = ( discounted_rewards- np.mean(discounted_rewards))/np.std(discounted_rewards)

    return list(reversed(discounted_rewards))

def main(learning_rate,gamma,nr_envs,entropy_coeff,max_iter,baseline):

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    env = gym.make("CartPole-v0")
    agent = PolicyGradientAgent(env.observation_space.shape[0],env.action_space.n,learning_rate,nr_envs,entropy_coeff)

    sess.run(tf.global_variables_initializer())

    if not os.path.isdir("pg_runs"):
        os.mkdir("pg_runs")

    run_nr = 0

    for d in os.listdir("pg_runs"):
        if d.startswith("run_"):
            run_nr += 1

    writer = tf.summary.FileWriter("./pg_runs/run_"+str(run_nr),sess.graph)
    merged = tf.summary.merge_all()

    reward_history = []

    for i in range(max_iter):

        all_discounted_rewards,  all_actions, all_states, all_rewards = [], [], [], []

        for i_env in range(nr_envs):

            is_done = False
            rewards_batch, states_batch, actions_batch = [], [], []

            state = env.reset()

            while not is_done:

                action = agent.sample_action([state])
                nxt_state, reward, is_done,_ = env.step(action[0])
                states_batch.append(state)
                actions_batch.append(action[0])
                rewards_batch.append(reward)
                state = nxt_state

            discounted_rewards = compute_discounted_return(rewards_batch,gamma,baseline)

            all_states.extend(states_batch)
            all_actions.extend(actions_batch)
            all_rewards.extend(rewards_batch)
            all_discounted_rewards.extend(discounted_rewards)


        loss = agent.train(all_states,all_actions,all_discounted_rewards)
        summary = sess.run(merged,feed_dict={agent.return_ph:all_discounted_rewards,agent.actions_ph:all_actions,agent.states_ph:all_states,agent.rewards_ph:all_rewards})
        writer.add_summary(summary,i)
        writer.flush()

        reward_history.append(sum(all_rewards)/nr_envs)

        if np.mean(reward_history[-min(10,len(reward_history)):]) > 190:
            print "Solved!"
            break

    sess.close()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",default=0.01,help="Learning rate for Neural netword",type=float)
    parser.add_argument("--gamma",default=0.95,help="Discounted reward coefficient",type=float)
    parser.add_argument("--nr_envs",default=4,help="Number parallel gym environments",type=int)
    parser.add_argument("--entropy_coeff",default=0.001,help="Entropy scaling",type=float)
    parser.add_argument("--max_iter",default=300,help="Maximum number of iterations",type=int)
    parser.add_argument("--baseline",help="Use baseline",action="store_true")

    args = vars(parser.parse_args())

    main(**args)







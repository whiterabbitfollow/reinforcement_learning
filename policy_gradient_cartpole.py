#!/usr/bin/python

import tensorflow as tf
import numpy as np
import gym
import argparse
from utils import get_run_nr, BaseAgent

class PolicyGradientAgent(BaseAgent):

    def __init__(self,state_dim,n_actions,learning_rate,entropy_coeff):

        super(PolicyGradientAgent, self).__init__()

        self.n_actions = n_actions

        self.sess = tf.get_default_session()

        self.states_ph = tf.placeholder(tf.float32,(None,state_dim),name="states")
        self.actions_ph = tf.placeholder(tf.int32,(None,),name="actions")
        self.return_ph = tf.placeholder(tf.float32,(None,),name="dicounted_returns")
        self.rewards_ph = tf.placeholder(tf.float32,(None,),name="rewards")

        with tf.variable_scope("PGAgent"):

            net = tf.layers.dense(self.states_ph,32,activation=tf.nn.relu)
            net = tf.layers.dense(net,32,activation=tf.nn.relu)
            self.logits = tf.layers.dense(net,n_actions,activation=None)

            self.log_probs = tf.nn.log_softmax(self.logits)
            self.probs = tf.nn.softmax(self.logits)
            self.neg_log_probs = - tf.reduce_sum(self.log_probs * tf.one_hot(self.actions_ph,depth=n_actions),axis=-1)

            self.entropy = -tf.reduce_sum(self.log_probs * self.probs,axis=-1)
            self.loss = tf.reduce_mean(self.neg_log_probs*self.return_ph)  - entropy_coeff * tf.reduce_mean(self.entropy)
            self.train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


    def policy(self,states):

        logits_actions = self.sess.run(self.logits,feed_dict={self.states_ph:states}).tolist()

        probs_actions = []

        for logit in logits_actions:

            logit_exp = np.exp(logit)
            probs_actions.append(logit_exp/np.sum(logit_exp,axis=-1))

        return [ np.random.choice(self.n_actions,p=prob) for prob in probs_actions ]

    def train(self, states, actions, discounted_rewards):

        loss, _ = self.sess.run([self.loss,self.train_opt],feed_dict={self.states_ph:states,self.actions_ph:actions,self.return_ph:discounted_rewards})

        return loss


def compute_discounted_return(rewards,gamma=0.95,baseline=False):

    discounted_rewards = []
    acc_r = 0.0

    for r in reversed(rewards):
        acc_r = r + gamma*acc_r
        discounted_rewards.append(acc_r)

    if baseline:
        discounted_rewards = (discounted_rewards- np.mean(discounted_rewards))/np.std(discounted_rewards)

    return list(reversed(discounted_rewards))

def main(args):

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    env = gym.make("CartPole-v0")

    with tf.variable_scope("summaries"):
        acc_reward_var = tf.Variable(0.0,"mean_reward")
        loss_var = tf.Variable(0.0,"loss")
        summary_list = [tf.summary.scalar("mean_reward", acc_reward_var),tf.summary.scalar("loss_var", loss_var)]

    agent = PolicyGradientAgent(env.observation_space.shape[0],env.action_space.n,args["learning_rate"],args["entropy_coeff"])

    sess.run(tf.global_variables_initializer())

    run_nr = get_run_nr("runs",starts_with="PG")

    writer = tf.summary.FileWriter("./runs/PG_"+str(run_nr),sess.graph)
    merged = tf.summary.merge_all()

    reward_history = []

    for i in range(args["max_iter"]):

        all_discounted_rewards,  all_actions, all_states, all_rewards = [], [], [], []

        for i_env in range(args["nr_envs"]):

            is_done = False
            rewards_batch, states_batch, actions_batch = [], [], []

            state = env.reset()

            while not is_done:

                action = agent.policy([state])
                nxt_state, reward, is_done,_ = env.step(action[0])
                states_batch.append(state)
                actions_batch.append(action[0])
                rewards_batch.append(reward)
                state = nxt_state

            discounted_rewards = compute_discounted_return(rewards_batch,args["gamma"],args["baseline"])

            all_states.extend(states_batch)
            all_actions.extend(actions_batch)
            all_rewards.extend(rewards_batch)
            all_discounted_rewards.extend(discounted_rewards)


        loss = agent.train(all_states,all_actions,all_discounted_rewards)


        mean_reward = np.sum(all_rewards)/args["nr_envs"]

        summary = sess.run(merged,feed_dict={acc_reward_var:mean_reward,loss_var:loss})

        writer.add_summary(summary,i)
        writer.flush()

        reward_history.append(mean_reward)

        if np.mean(reward_history[-min(100/args["nr_envs"],len(reward_history)):]) > 195:
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

    main(args)







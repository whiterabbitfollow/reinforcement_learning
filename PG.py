#!/usr/bin/python

import tensorflow as tf
import numpy as np
import gym
import argparse
from utils.utils import evaluate_agent, BaseDeepAgent, compute_discounted_return

class PolicyGradientAgent(BaseDeepAgent):

    def __init__(self,env,sess,gamma,baseline, learning_rate,entropy_coeff):

        super(PolicyGradientAgent, self).__init__(env,sess)
        self.gamma = gamma
        self.baseline = baseline
        self.init_log_var = -1 # TODO add as config..
        self._init_ph()
        self._init_net(learning_rate,entropy_coeff)
        self._init_tb_summaries()

    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.train_reward_var = tf.Variable(0.0, name="train_reward")
            self.loss_var = tf.Variable(0.0, "actor_loss")

            summaries = [tf.summary.scalar("train_reward", self.train_reward_var),
                         tf.summary.scalar("actor_loss", self.loss_var)
                         ]

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)

            self.merged = tf.summary.merge(summaries)

    def _init_ph(self):

        self.states_ph = tf.placeholder(tf.float32, (None, self.s_dim), name="states")

        if self.is_discrete:
            self.actions_ph = tf.placeholder(tf.int32, (None,), name="actions")
        else:
            self.actions_ph = tf.placeholder(tf.float32, (None,self.a_dim), name="actions")

        self.return_ph = tf.placeholder(tf.float32, (None,), name="dicounted_returns")
        self.rewards_ph = tf.placeholder(tf.float32, (None,), name="rewards")

    def _init_net(self,learning_rate,entropy_coeff):

        with tf.variable_scope("PG"):

            net = tf.layers.dense(self.states_ph,32,activation=tf.nn.relu)
            net = tf.layers.dense(net,32,activation=tf.nn.relu)

            if self.is_discrete:

                self.logits = tf.layers.dense(net,self.n_actions,activation=None)
                self.log_probs = tf.nn.log_softmax(self.logits)
                self.probs = tf.nn.softmax(self.logits)
                self.neg_log_probs = - tf.reduce_sum(self.log_probs * tf.one_hot(self.actions_ph,depth=self.n_actions), axis=-1)
                self.entropy = -tf.reduce_sum(self.log_probs * self.probs, axis=-1)

            else:

                self.means = tf.layers.dense(net, self.a_dim, activation=None)
                self.log_vars = tf.get_variable("logvars", self.a_dim, tf.float32, tf.constant_initializer(0.0)) + self.init_log_var

                self._policy = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.a_dim,), dtype=tf.float32))  # TODO: read about this
                self.neg_log_probs = -self.calc_logprob(self.means, self.actions_ph, self.log_vars, self.a_dim)

                log_det_cov = tf.reduce_sum(self.log_vars)
                self.entropy = 0.5 * (self.a_dim + np.log(2 * np.pi) * self.a_dim + log_det_cov)

                self.sampled_action = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.a_dim,),dtype=tf.float32))  # TODO: read about this

            self.loss = tf.reduce_mean(self.neg_log_probs*self.return_ph) - entropy_coeff * tf.reduce_mean(self.entropy)
            self.train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


    def calc_logprob(self, mean_action, actions, log_vars, k):
        logp = -0.5 * tf.reduce_sum(log_vars) - k * np.log(2 * np.pi)  # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Properties
        logp += -0.5 * tf.reduce_sum(tf.square(actions - mean_action) / tf.exp(log_vars), axis=1)
        return logp

    def policy(self,states):
        if self.is_discrete:

            logits_actions = self.sess.run(self.logits,feed_dict={self.states_ph:states}).tolist()

            probs_actions = []

            for logit in logits_actions:

                logit_exp = np.exp(logit)
                probs_actions.append(logit_exp/np.sum(logit_exp,axis=-1))

            actions = [ np.random.choice(self.n_actions,p=prob) for prob in probs_actions ]
        else:
            actions = self.sess.run(self.sampled_action, feed_dict={self.states_ph: states})

        return actions

    def train(self, states, actions, discounted_rewards):

        loss, _ = self.sess.run([self.loss,self.train_opt],feed_dict={self.states_ph:states,self.actions_ph:actions,self.return_ph:discounted_rewards})

        return loss


    def _run_eps(self,nr_eps):

        all_discounted_rewards, all_actions, all_states, all_rewards = [], [], [], []

        for i_env in range(nr_eps):

            is_done = False
            rewards_batch, states_batch, actions_batch = [], [], []

            state = self.env.reset()

            while not is_done:
                action = self.policy([state])
                nxt_state, reward, is_done, _ = self.env.step(action[0])
                states_batch.append(state)
                actions_batch.append(action[0])
                rewards_batch.append(reward)
                state = nxt_state
                self.nr_env_interactions += 1

            discounted_rewards = compute_discounted_return(rewards_batch, self.gamma, self.baseline)

            all_states.extend(states_batch)
            all_actions.extend(actions_batch)
            all_rewards.extend(rewards_batch)
            all_discounted_rewards.extend(discounted_rewards)

        return all_discounted_rewards, all_actions, all_states, all_rewards



    def run_episodes(self,n_episodes,nr_eps,verbosity=0, eval_freq=10, n_interact_2_evaluate=10):

        for i in range(n_episodes):

            all_discounted_rewards, all_actions, all_states, all_rewards = self._run_eps(nr_eps)

            loss = self.train(all_states, all_actions, all_discounted_rewards)

            mean_reward = np.sum(all_rewards) / nr_eps

            summary = self.sess.run(self.merged, feed_dict={self.train_reward_var:mean_reward,
                                                            self.loss_var: loss})

            self.writer.add_summary(summary, self.nr_env_interactions)
            self.writer.flush()

            if i % eval_freq == 0:
                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate))
                summary = self.sess.run(self.eval_summary, {self.eval_reward_var: np.mean(eval_rewards)})
                self.writer.add_summary(summary, self.nr_env_interactions)
                self.writer.flush()



    def __repr__(self):
        return "PG"

def main(args):

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    env = gym.make(args["env"])

    agent = PolicyGradientAgent(env,sess, args["gamma"], args["baseline"], args["learning_rate"], args["entropy_coeff"])

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(args["max_iter"],args["nr_eps"], eval_freq=10, n_interact_2_evaluate=100)

    agent.close()

if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCar-v0", help="", type=str)
    parser.add_argument("--learning_rate",default=0.001,help="Learning rate",type=float)
    parser.add_argument("--gamma",default=0.95,help="Discounted reward coefficient",type=float)
    parser.add_argument("--nr_eps",default=4,help="Number episodes to get data from",type=int)
    parser.add_argument("--entropy_coeff",default=1e-4,help="Entropy scaling",type=float)
    parser.add_argument("--max_iter",default=10000,help="Maximum number of iterations",type=int)
    parser.add_argument("--baseline",help="Use baseline",action="store_true")

    args = vars(parser.parse_args())
    main(args)







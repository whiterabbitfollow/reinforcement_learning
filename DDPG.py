#!/usr/bin/python

import tensorflow as tf
import numpy as np
import gym
from utils import BaseDeepAgent, ReplayBuffer, OrnsteinUhlenbeckActionNoise, evaluate_agent
import argparse

class ActorNetwork():

    def __init__(self, states_ph, action_dim, action_bound, name="ActorNetwork"):

        with tf.variable_scope(name):

            net = tf.layers.dense(states_ph, 400)
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 300)
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            out = tf.layers.dense(net,action_dim,activation=tf.nn.tanh,kernel_initializer=tf.initializers.random_uniform(minval=-0.003,maxval=0.003))
            self.policy = tf.multiply(out,action_bound)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name)

class CriticNetwork():

    def __init__(self,states_ph,actions_ph,reuse=False,name="CriticNetwork"):

        with tf.variable_scope(name,reuse=reuse):

            self.target_Q_values = tf.placeholder(tf.float32, (None, 1))

            net = tf.layers.dense(states_ph, 400)
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

            net1 = tf.layers.dense(net, 300, use_bias=False)
            net2 = tf.layers.dense(actions_ph, 300)

            out = tf.nn.relu(net1 + net2)
            self.Q_value = tf.layers.dense(out, 1, kernel_initializer=tf.initializers.random_uniform(minval=-0.003,
                                                                                                     maxval=0.003))

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

class DDPG(BaseDeepAgent):

    def __init__(self, env,buffer_size, batch_size, actor_lr, critic_lr, tau, gamma):

        assert abs(env.action_space.low) == abs(env.action_space.high), "Must have symmetric action bound"
        sess = tf.get_default_session()

        super(DDPG, self).__init__(env, sess)

        self.batch_size = batch_size
        self.buff = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.tau = tau

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim))

        self._init_ph()
        self._init_nets()
        self._init_losses_n_trainer(actor_lr,critic_lr)
        self._init_updater_ph()
        self._init_tb_summaries()

    def _init_ph(self):

        self.states_ph = tf.placeholder(tf.float32, (None, self.s_dim), name="state")
        self.actions_ph = tf.placeholder(tf.float32, (None, self.a_dim), name="action")
        self.action_gradients_ph = tf.placeholder(tf.float32, (None, 1), "action_grads")
        self.target_Q_values = tf.placeholder(tf.float32, (None, 1), "Q_target")

    def _init_nets(self):

        self.actor = ActorNetwork(self.states_ph, self.a_dim, self.action_bound, name="actor")
        self.target_actor = ActorNetwork(self.states_ph, self.a_dim, self.action_bound, name="target_actor")
        self.critic = CriticNetwork(self.states_ph, self.actions_ph, name="critic")
        self.target_critic = CriticNetwork(self.states_ph, self.actions_ph, name="target_critic")

    def _init_losses_n_trainer(self, actor_lr, critic_lr):

        self.unscaled_actor_gradient = tf.gradients(self.actor.policy, self.actor.weights, -self.action_gradients_ph)

        self.actor.loss = list(map(lambda x: tf.div(x, self.batch_size), self.unscaled_actor_gradient))
        self.actor.train_opt = tf.train.AdamOptimizer(learning_rate=actor_lr).apply_gradients(
            zip(self.actor.loss, self.actor.weights))

        self.action_gradients = tf.gradients(self.critic.Q_value, self.actions_ph)

        self.critic.loss = tf.losses.mean_squared_error(self.target_Q_values, self.critic.Q_value)
        self.critic.train_opt = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic.loss)

    def _init_updater_ph(self):
        self.update_target_network_params_critic = \
            [self.target_critic.weights[i].assign(tf.multiply(self.critic.weights[i], self.tau) +
                                                  tf.multiply(self.target_critic.weights[i], 1. - self.tau))
             for i in range(len(self.target_critic.weights))]

        self.update_target_network_params_actor = \
            [self.target_actor.weights[i].assign(tf.multiply(self.actor.weights[i], self.tau) +
                                                 tf.multiply(self.target_actor.weights[i], 1. - self.tau))
             for i in range(len(self.target_actor.weights))]

    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.critic_loss_var = tf.Variable(0.0, name="critic_loss")
            self.mean_Q_var = tf.Variable(0.0, name="mean_Q_value")
            self.max_Q_var = tf.Variable(0.0, name="max_Q_value")
            self.mean_target_Q_var = tf.Variable(0.0, name="mean_target_Q_value")
            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.mean_abs_grad_var = tf.Variable(0.0, name="mean_abs_grad")



            self.merged = tf.summary.merge([tf.summary.scalar("critic_loss", self.critic_loss_var),
                                         tf.summary.scalar("mean_Q", self.mean_Q_var),
                                         tf.summary.scalar("max_Q", self.max_Q_var),
                                         tf.summary.scalar("mean_Q_target", self.mean_target_Q_var),
                                         tf.summary.scalar("mean_abs_grad", self.mean_abs_grad_var)
                                         ])

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)

    def policy(self, state):
        return self.sess.run(self.actor.policy, feed_dict={self.states_ph: state})

    def target_policy(self, state):
        return self.sess.run(self.target_actor.policy, feed_dict={self.states_ph: state})

    def action_grads(self, state, action):
        return self.sess.run(self.action_gradients, feed_dict={self.states_ph: state, self.actions_ph: action})

    def target_Q_value(self, state, action):
        return self.sess.run(self.target_critic.Q_value, feed_dict={self.states_ph: state,self.actions_ph:action})

    def train(self, state_mb, action_mb, reward_mb, done_mb, nxt_state_mb):

        nxt_action_mb = self.target_policy(nxt_state_mb)

        nxt_Q_value = self.target_Q_value(nxt_state_mb,nxt_action_mb)

        target_Q_value = reward_mb + self.gamma*(1.0-done_mb)*nxt_Q_value.ravel()

        critic_loss, _, action_grads, Q_values = self.sess.run([self.critic.loss, self.critic.train_opt, self.action_gradients,
                                                      self.critic.Q_value],
                                                     feed_dict={self.target_Q_values:target_Q_value.reshape(-1,1),
                                                                self.states_ph: state_mb,
                                                                self.actions_ph: action_mb})

        actions = self.policy(state_mb) # should be same as action_mb??
        action_grads = self.action_grads(state_mb, actions)

        # could be wrong grad here...
        actor_loss, _ = self.sess.run([self.actor.loss,self.actor.train_opt],
                                      feed_dict={self.states_ph:state_mb,self.action_gradients_ph:action_grads[0]})

        return critic_loss, actor_loss, action_grads[0], target_Q_value, Q_values

    def copy_network_parameters(self):
        self.sess.run(self.update_target_network_params_actor)
        self.sess.run(self.update_target_network_params_critic)

    def _run_episode(self,max_nr_steps,verbosity=0):

        env = self.env
        batch_size = self.batch_size
        buff = self.buff

        done = False
        s = env.reset()
        i_step = 0
        acc_reward = 0

        while i_step < max_nr_steps and not done:

            a = self.policy(s.reshape(1, -1))[0] + self.actor_noise()

            s_nxt, r, done, _ = env.step(a)

            buff.add(s.ravel(), a, r, done, s_nxt.ravel())

            if buff.size >= batch_size:

                state_mb, action_mb, reward_mb, done_mb, nxt_state_mb = buff.sample(batch_size)

                logs = self.train(state_mb, action_mb, reward_mb, done_mb, nxt_state_mb)

                critic_loss, actor_loss, action_grads, target_Q_values, Q_values = logs

                f_dict = {self.critic_loss_var: critic_loss,
                          self.mean_Q_var: np.mean(Q_values), self.max_Q_var: np.max(Q_values),
                          self.mean_target_Q_var: np.mean(target_Q_values),
                          self.mean_abs_grad_var: np.mean(np.abs(action_grads))}

                summary = self.sess.run(self.merged, f_dict)

                self.writer.add_summary(summary, self.nr_env_interactions)
                self.writer.flush()
                self.copy_network_parameters()

            s = s_nxt

            acc_reward += r

            self.nr_env_interactions += 1
            i_step += 1

        return acc_reward

    def run_episodes(self,n_episodes,verbosity=0, max_nr_steps=500, eval_freq=10, n_interact_2_evaluate=10):

        eval_rewards = [0]

        for i_ep in range(n_episodes):

            ep_reward = self._run_episode(max_nr_steps,verbosity)

            if i_ep % eval_freq == 0:

                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate))

                summary = self.sess.run(self.eval_summary, {self.eval_reward_var: np.mean(eval_rewards)})

                self.writer.add_summary(summary, self.nr_env_interactions)
                self.writer.flush()

            if verbosity > 0:
                print "Episode: %i reward: %f nr_steps: %i eval_reward: %f" % (i_ep, ep_reward, self.nr_env_interactions,np.mean(eval_rewards))


    def __repr__(self):
        return "DDPG"

def main(args):

    env = gym.make(args["env"])

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = DDPG(env, args["buffer_size"], args["batch_size"], args["actor_learning_rate"], args["critic_learning_rate"], args["tau"], args["gamma"])

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(args["max_nr_episodes"], args["verbosity"], args["max_nr_steps"])

    agent.close()

if __name__=="__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--buffer_size", type=int, default=100000)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--env", type=str, default="Pendulum-v0")
    args.add_argument("--max_nr_steps", type=int, default=1000)
    args.add_argument("--max_nr_episodes", type=int, default=1000)
    args.add_argument("--tau", type=float, default=0.001)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--actor_learning_rate", type=float, default=0.0001)
    args.add_argument("--critic_learning_rate", type=float, default=0.001)
    args.add_argument("--use_target_critic", action="store_true")
    args.add_argument("--verbosity", type=int, default=1)
    main(vars(args.parse_args()))

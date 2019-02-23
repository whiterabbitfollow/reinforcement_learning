#!/usr/bin/python

import tensorflow as tf
import numpy as np
import gym
from utils import ReplayBuffer, BaseAgent, get_run_nr
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

class DDQN(BaseAgent):

    def __init__(self, s_dim, a_dim, action_bound, batch_size, actor_lr, critic_lr, tau, gamma):

        super(BaseAgent, self).__init__()

        self.sess = tf.get_default_session()
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.a_dim = a_dim
        self.sess = tf.get_default_session()

        self.states_ph = tf.placeholder(tf.float32, (None, s_dim),name="state")
        self.actions_ph = tf.placeholder(tf.float32, (None,a_dim),name="action")
        self.action_gradients_ph = tf.placeholder(tf.float32, (None, 1),"action_grads")
        self.target_Q_values = tf.placeholder(tf.float32, (None, 1),"Q_target")

        self.actor = ActorNetwork(self.states_ph,a_dim,action_bound,name="actor")
        self.target_actor = ActorNetwork(self.states_ph, a_dim, action_bound,name="target_actor")

        self.critic = CriticNetwork(self.states_ph,self.actions_ph,name="critic")
        self.target_critic = CriticNetwork(self.states_ph, self.actions_ph,name="target_critic")

        self.unscaled_actor_gradient = tf.gradients(self.actor.policy, self.actor.weights, -self.action_gradients_ph)

        self.actor.loss = list(map(lambda x: tf.div(x, self.batch_size), self.unscaled_actor_gradient))
        self.actor.train_opt = tf.train.AdamOptimizer(learning_rate=actor_lr).apply_gradients(zip(self.actor.loss, self.actor.weights))

        self.action_gradients = tf.gradients(self.critic.Q_value, self.actions_ph)

        self.critic.loss = tf.losses.mean_squared_error(self.target_Q_values, self.critic.Q_value)
        self.critic.train_opt = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic.loss)

        self.update_target_network_params_critic = \
            [self.target_critic.weights[i].assign(tf.multiply(self.critic.weights[i], self.tau) +
                                                  tf.multiply(self.target_critic.weights[i], 1. - self.tau))
             for i in range(len(self.target_critic.weights))]

        self.update_target_network_params_actor = \
            [self.target_actor.weights[i].assign(tf.multiply(self.actor.weights[i], self.tau) +
                                                       tf.multiply(self.target_actor.weights[i], 1. - self.tau))
             for i in range(len(self.target_actor.weights))]

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

        # assigns = []
        # tau = self.tau
        #
        # for w_reference, w_target in zip(self.actor.weights, self.target_actor.weights):
        #     assigns.append(tf.assign(w_target, tf.multiply(w_reference,tau) + tf.multiply(w_target, 1. - tau),
        #                              validate_shape=True))
        # assigns = []
        #
        # for w_reference, w_target in zip(self.critic.weights, self.target_critic.weights):
        #     assigns.append(tf.assign(w_target, tf.multiply(w_target, tau) + tf.multiply(w_reference,1. - tau),
        #                              validate_shape=True))
        #
        # self.sess.run(assigns)

        self.sess.run(self.update_target_network_params_actor)
        self.sess.run(self.update_target_network_params_critic)



# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def main(args):

    env = gym.make("Pendulum-v0")

    a_dim = env.action_space.shape[0]

    assert abs(env.action_space.low) == abs(env.action_space.high),"Must have symmetric action bound"

    action_bound = abs(env.action_space.low)

    s_dim = env.observation_space.shape[0]

    batch_size = args["batch_size"]
    max_nr_episodes = args["max_nr_episodes"]
    max_nr_steps = args["max_nr_steps"]
    verbosity = args["verbosity"]

    buff = ReplayBuffer(args["buffer_size"])

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.variable_scope("summaries"):

        critic_loss_var = tf.Variable(0.0,name="critic_loss")
        mean_Q_var = tf.Variable(0.0, name="mean_Q_value")
        max_Q_var = tf.Variable(0.0, name="max_Q_value")
        mean_target_Q_var = tf.Variable(0.0, name="mean_target_Q_value")
        acc_reward_var = tf.Variable(0.0,name="acc_reward")
        mean_abs_grad_var = tf.Variable(0.0,name="mean_abs_grad")

        loss_sum = tf.summary.merge([tf.summary.scalar("critic_loss",critic_loss_var),
                                     tf.summary.scalar("mean_Q", mean_Q_var),
                                     tf.summary.scalar("max_Q", max_Q_var),
                                     tf.summary.scalar("mean_Q_target", mean_target_Q_var),
                                     tf.summary.scalar("mean_abs_grad", mean_abs_grad_var)
                                     ])

        reward_summary = tf.summary.scalar("mean_reward", acc_reward_var)

    agent = DDQN(s_dim,a_dim,action_bound,args["batch_size"],args["actor_learning_rate"], args["critic_learning_rate"], args["tau"], args["gamma"])

    run_nr = get_run_nr("runs",starts_with="DDPG")

    writer = tf.summary.FileWriter("runs/DDPG_"+str(run_nr), sess.graph)

    sess.run(tf.global_variables_initializer())

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))

    i = 0

    for i_ep in range(max_nr_episodes):

        done = False
        s = env.reset()
        i_step = 0
        acc_reward = 0

        while i_step < max_nr_steps and not done:

            a = agent.policy(s.reshape(1,-1))[0] + actor_noise()

            s_nxt, r, done, _ = env.step(a)

            buff.add(s.ravel(), a, r, done, s_nxt.ravel())

            if buff.size >= batch_size:

                state_mb, action_mb, reward_mb, done_mb, nxt_state_mb = buff.sample(batch_size)

                logs = agent.train(state_mb, action_mb, reward_mb, done_mb, nxt_state_mb)

                critic_loss, actor_loss, action_grads, target_Q_values, Q_values = logs

                f_dict = {critic_loss_var:critic_loss,
                          mean_Q_var: np.mean(Q_values),max_Q_var: np.max(Q_values),
                          mean_target_Q_var:np.mean(target_Q_values),
                          mean_abs_grad_var:np.mean(np.abs(action_grads))}

                summary = sess.run(loss_sum,f_dict)

                writer.add_summary(summary,i)
                writer.flush()

                agent.copy_network_parameters()

            s = s_nxt

            acc_reward += r

            if done:

                summary = sess.run(reward_summary, {acc_reward_var:acc_reward})

                writer.add_summary(summary, i_ep)
                writer.flush()

                if verbosity > 0:
                    print "Episode: %i Reward: %f Nr_steps: %i"%(i_ep, acc_reward, i_step)

            i += 1
            i_step += 1

    sess.close()
    writer.close()


if __name__=="__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--buffer_size", type=int, default=100000)
    args.add_argument("--batch_size", type=int, default=64)

    args.add_argument("--max_nr_steps", type=int, default=1000)
    args.add_argument("--max_nr_episodes", type=int, default=1000)

    args.add_argument("--tau", type=float, default=0.001)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--actor_learning_rate", type=float, default=0.0001)
    args.add_argument("--critic_learning_rate", type=float, default=0.001)

    args.add_argument("--use_target_critic", action="store_true")

    args.add_argument("--verbosity", type=int, default=1)

    main(vars(args.parse_args()))

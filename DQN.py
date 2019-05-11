import tensorflow as tf
import numpy as np
import random
import gym
import time

from utils.base_agents import BaseDeepAgent
from utils.buffers import ReplayBuffer, FrameBuffer, PreprocessAtari
from utils.other import evaluate_agent


class ConvNet(object):

    def __init__(self, name, state_ph, n_actions):

        with tf.variable_scope(name):

            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=state_ph,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(), units=n_actions, activation=None)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)




class DenseNet(object):

    def __init__(self, name, state_ph, n_actions):

        with tf.variable_scope(name):

            out = tf.layers.dense(inputs=state_ph,
                                      units=100,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            out = tf.layers.dense(inputs=out,
                                      units=100,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc2")

            self.output = tf.layers.dense(inputs=out, kernel_initializer=tf.contrib.layers.xavier_initializer(), units=n_actions, activation=None)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DQN(BaseDeepAgent):

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size):

        super(DQN,self).__init__(env,sess)

        self.s_dim = env.observation_space.shape

        self.eps_decay = 0.985
        self.eps_min = 0.05

        self.gamma = gamma
        self.eps = eps
        self.tau = tau

        self.batch_size = batch_size
        self.buff = ReplayBuffer(max_buffer_size=max_buffer_size)

        self._init_ph()
        self._init_net()
        self._init_updater_ph()
        self._init_losses(lr)

    def _init_ph(self):

        self.state_ph = tf.placeholder(tf.float32,(None,) +self.s_dim, "state")
        self.nxt_state_ph = tf.placeholder(tf.float32, (None,) + self.s_dim, "nxt_state")
        self.reward_ph = tf.placeholder(tf.float32, (None,), "reward")
        self.action_ph = tf.placeholder(tf.int32, (None,), "action")
        self.is_done_ph = tf.placeholder(tf.float32, (None,), "is_done")
        self.target_q_ph = tf.placeholder(tf.float32, (None,), "is_done")
        self._init_tb_summaries()


    def _init_net(self):

        self.learning_net = ConvNet("learner", self.state_ph, self.n_actions)
        self.target_net = ConvNet("target", self.nxt_state_ph, self.n_actions)

        # self.learning_net = DenseNet("learner", self.state_ph, self.n_actions)
        # self.target_net = DenseNet("target", self.nxt_state_ph, self.n_actions)

    def _init_losses(self, lr):

        nxt_state_q = tf.reduce_max(self.target_net.output, axis=-1)

        self.target_q_values = self.reward_ph + self.gamma * nxt_state_q * (1.0-self.is_done_ph)

        self.q_values_state = tf.reduce_sum(tf.multiply(self.learning_net.output, tf.one_hot(self.action_ph, self.n_actions)),axis=1)

        self.loss = tf.reduce_mean((tf.stop_gradient(self.target_q_values) - self.q_values_state)**2.0)

        self.train_opt = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self, state_mb, action_mb, reward_mb, is_done_mb, nxt_state_mb):


        feed_dict = {self.state_ph:state_mb,
                     self.nxt_state_ph: nxt_state_mb,
                     self.action_ph: action_mb,
                     self.reward_ph: reward_mb,
                     self.is_done_ph: is_done_mb
                     }

        _, loss, qs = self.sess.run([self.train_opt, self.loss, self.q_values_state], feed_dict)

        return loss, qs

    def _init_updater_ph(self):

        self.update_target_network_params = [self.target_net.weights[i].assign(tf.multiply(self.learning_net.weights[i], self.tau)
                                                                                + tf.multiply(self.target_net.weights[i], 1. - self.tau))
                                             for i in range(len(self.target_net.weights))]

    def policy(self, state,greedy=False):

        if random.random() > self.eps or greedy:

            feed_dict = {self.state_ph: state}
            action = np.argmax(self.sess.run(self.learning_net.output, feed_dict), axis=1)

        else:

            action = [random.randint(0, self.n_actions - 1)]

        return action

    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.critic_loss_var = tf.Variable(0.0, name="critic_loss")
            self.train_reward_var = tf.Variable(0.0, name="train_reward")
            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.n_steps_var = tf.Variable(0.0, name="nr_eps_steps")
            self.mean_Q_var = tf.Variable(0.0, name="mean_Q_value")
            self.max_Q_var = tf.Variable(0.0, name="max_Q_value")
            self.eps_var = tf.Variable(0.0, name="eps")

            # self.mean_abs_grad_var = tf.Variable(0.0, name="mean_abs_grad")

            self.merged = tf.summary.merge([tf.summary.scalar("critic_loss", self.critic_loss_var),
                                            tf.summary.scalar("eps", self.eps_var),
                                            tf.summary.scalar("mean_Q", self.mean_Q_var),
                                            tf.summary.scalar("max_Q", self.max_Q_var)
                                            ])

            self.eps_sum = tf.summary.merge([tf.summary.scalar("train_reward", self.train_reward_var),
                                             tf.summary.scalar("n_steps", self.n_steps_var)
                                             ]
                                            )

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)

    def decay_eps(self):
        self.eps = max(self.eps*self.eps_decay,self.eps_min)

    def copy_network_parameters(self):
        self.sess.run(self.update_target_network_params)

    def _run_episode(self, max_nr_steps, train, verbosity=0):

        env = self.env
        batch_size = self.batch_size
        buff = self.buff

        done = False
        s = env.reset()
        i_step = 0
        acc_reward = 0

        while i_step < max_nr_steps and not done:

            a = self.policy([s])[0]

            s_nxt, r, done, _ = env.step(a)

            buff.add(s, a, r, done, s_nxt)

            if train:

                state_mb, action_mb, reward_mb, done_mb, nxt_state_mb = buff.sample(batch_size)

                critic_loss, qs = self.train(state_mb, action_mb, reward_mb, done_mb, nxt_state_mb)

                f_dict = {self.critic_loss_var: critic_loss,
                          self.eps_var:self.eps,
                          self.max_Q_var:np.max(qs),
                          self.mean_Q_var:np.mean(qs)
                          }

                summary = self.sess.run(self.merged, f_dict)

                self.writer.add_summary(summary, self.nr_env_interactions)

                self.writer.flush()



                self.copy_network_parameters()  # TODO: all the time?

            s = s_nxt
            acc_reward += r

            i_step += 1
            self.nr_env_interactions += 1

        if train:
            self.decay_eps()

        return acc_reward, i_step

    def run_episodes(self,n_episodes,verbosity=0, max_nr_steps=100000, eval=False, eval_freq=20, n_interact_2_evaluate=100):

        time_start = time.time()

        while not self.buff.have_stored_enough():
            _, _ = self._run_episode(max_nr_steps, verbosity)

            if verbosity > 1:
                print "Buffer size %i, init ratio %f" % (self.buff.size,self.buff.init_ratio() )

        if verbosity > 0:
            print "Buffer initialized in %f s with %i samples"%(time.time()-time_start,self.buff.size)

        eval_rewards = 0

        for i_ep in range(n_episodes):

            ep_reward, i_step = self._run_episode(max_nr_steps,verbosity)

            if eval and (i_ep % eval_freq == 0):

                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate, greedy=True))

                summary = self.sess.run(self.eval_summary, {self.eval_reward_var: eval_rewards})

                self.writer.add_summary(summary, self.nr_env_interactions)

                self.writer.flush()

            summary = self.sess.run(self.eps_sum, {self.train_reward_var: ep_reward,
                                                   self.n_steps_var:i_step
                                                   }
                                    )

            self.writer.add_summary(summary, self.nr_env_interactions)

            self.writer.flush()

            if verbosity > 0:

                out_str = "Episode: %i"%(i_ep)
                out_str += " Reward: %f"%(ep_reward)
                out_str += " nr_steps: %i" % (self.nr_env_interactions)
                if eval:
                    out_str += " eval_reward: %f" % (eval_rewards)
                out_str += " Buffer size: %i" % (self.buff.size)
                out_str += " eps: %f" % (self.eps)
                out_str += " episode_steps: %i" % (i_step)

                print out_str

    def __repr__(self):
        return "DQN"

def main():

    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make("BreakoutDeterministic-v4")
    #

    env = PreprocessAtari(env)
    env = FrameBuffer(env)

    # env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")
    #

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = DQN(env, sess, eps=1.0, max_buffer_size=100000, gamma=0.99, lr=1e-4, tau=0.01, batch_size=32)

    sess.run(tf.global_variables_initializer())

    # for chk_grad in tf.gradients(agent.target_q_values, agent.learning_net.weights):
    #     error_msg = "Reference q-values should have no gradient w.r.t. agent weights. Make sure you used target_network qvalues! "
    #     error_msg += "If you know what you're doing, ignore this assert."
    #     assert chk_grad is None or np.allclose(sess.run(chk_grad), sess.run(chk_grad * 0)), error_msg

    agent.run_episodes(1000, verbosity=1, eval=False)

    agent.close()

if __name__=="__main__":

    main()




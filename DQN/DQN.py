import tensorflow as tf
import numpy as np
import random
import gym
import time

from utils.base_agents import BaseDeepAgent
from utils.buffers import ReplayBuffer
from utils.preprocessing import FrameBuffer, PreprocessAtari, ProcessFrame84
from utils.other import evaluate_agent

class ConvNetOrg(object):

    def __init__(self, name, state_ph, n_actions):

        # from paper: "Playing Atari with Deep Reinforcement Learning"

        with tf.variable_scope(name):

            self.conv1 = tf.layers.conv2d(inputs=state_ph,
                                          filters=16,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                          name="conv1")

            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=32,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                          name="conv2")

            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")

            self.flatten = tf.contrib.layers.flatten(self.conv2_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=256,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.variance_scaling_initializer(scale=2), units=n_actions, activation=None)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)



class ConvNet(object):

    def __init__(self, name, state_ph, n_actions):

        with tf.variable_scope(name):

            self.conv1 = tf.layers.conv2d(inputs=state_ph,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          use_bias=False,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                          name="conv1")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                          use_bias=False,
                                          activation=tf.nn.relu,
                                          name="conv2")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="VALID",
                                          use_bias=False,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                          name="conv3")

            self.flatten = tf.contrib.layers.flatten(self.conv3)

            # tf.keras.initializers.VarianceScaling

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                          units=n_actions, activation=None)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DenseNet(object):

    def __init__(self, name, state_ph, n_actions):

        with tf.variable_scope(name):

            out = tf.layers.dense(inputs=state_ph,
                                      units=100,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      name="fc1")

            out = tf.layers.dense(inputs=out,
                                      units=100,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                      name="fc2")

            self.output = tf.layers.dense(inputs=out, kernel_initializer=tf.variance_scaling_initializer(scale=2), units=n_actions, activation=None)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DQN(BaseDeepAgent):

    def __init__(self, env, sess, eps, eps_decay_func, buff_obj, gamma, tau, max_nr_env_interactions, use_conv_net=False, lr=0.0000625):


        super(DQN,self).__init__(env, sess)

        self.s_dim = env.observation_space.shape

        self.max_nr_env_interactions = max_nr_env_interactions

        self.eps = eps
        self.decay_eps = eps_decay_func

        self.gamma = gamma
        self.buff = buff_obj
        self.tau = tau # TODO remove

        self._init_ph()
        self._init_net(use_conv_net)
        self._init_updater_ph()
        self._init_hardcopy_ph()
        self._init_losses(lr)
        self._init_tb_summaries()

    def _init_ph(self):

        self.state_ph = tf.placeholder(tf.float32,(None,) + self.s_dim, "state")
        self.nxt_state_ph = tf.placeholder(tf.float32, (None,) + self.s_dim, "nxt_state")
        self.reward_ph = tf.placeholder(tf.float32, (None,), "reward")
        self.action_ph = tf.placeholder(tf.int32, (None,), "action")
        self.is_done_ph = tf.placeholder(tf.float32, (None,), "is_done")
        self.target_q_ph = tf.placeholder(tf.float32, (None,), "target_q")


    def _init_net(self,use_conv_net):

        if use_conv_net:

            self.learning_net = ConvNet("learner", self.state_ph, self.n_actions)
            self.target_net = ConvNet("target", self.nxt_state_ph, self.n_actions)

        else:

            self.learning_net = DenseNet("learner", self.state_ph, self.n_actions)
            self.target_net = DenseNet("target", self.nxt_state_ph, self.n_actions)

    def _init_losses(self, lr):

        nxt_state_q = tf.reduce_max(self.target_net.output, axis=-1)

        self.target_q_values = self.reward_ph + self.gamma * nxt_state_q * (1.0-self.is_done_ph)

        self.q_values_state = tf.reduce_sum(tf.multiply(self.learning_net.output, tf.one_hot(self.action_ph, self.n_actions)),axis=1)

        # self.loss = tf.reduce_mean((tf.stop_gradient(self.target_q_values) - self.q_values_state)**2.0)

        self.loss = tf.losses.huber_loss(labels=tf.stop_gradient(self.target_q_values), predictions=self.q_values_state)

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


    def _init_hardcopy_ph(self):
        # seems to be needed in beginning..
        self.copy_target_network_params = [self.target_net.weights[i].assign(self.learning_net.weights[i]) for i in range(len(self.target_net.weights))]

    def policy(self, state, greedy=False):

        if random.random() > self.eps or greedy:

            # TODO do tf imp

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

    def copy_network_parameters(self):
        self.sess.run(self.update_target_network_params)

    def hard_copy_network_parameters(self):
        self.sess.run(self.copy_target_network_params)

    def _run_episode(self, max_nr_env_interactions, max_nr_steps, batch_size, nr_env_interact_before_copy=10000, update_freq=4):

        env = self.env
        buff = self.buff

        done = False
        s = env.reset()
        i_step = 0
        acc_reward = 0

        while i_step < max_nr_steps and not done and (self.nr_env_interactions < max_nr_env_interactions):

            a = self.policy([s])[0]

            s_nxt, r, done, _ = env.step(a)

            buff.add(s, a, r, done, s_nxt)

            if buff.have_stored_enough() and i_step % update_freq == 0:

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

                if self.nr_env_interactions % nr_env_interact_before_copy == 0:
                    self.hard_copy_network_parameters()

            s = s_nxt
            acc_reward += r

            self.eps = self.decay_eps(self.nr_env_interactions)

            i_step += 1

            self.nr_env_interactions += 1

        return acc_reward, i_step, done

    def run_episodes(self, batch_size=32, verbosity=0, max_nr_steps=100000, eval=False, eval_freq=20, n_interact_2_evaluate=100):

        time_start = time.time()

        self.hard_copy_network_parameters() # NEW...

        i_ep = 0

        while self.nr_env_interactions < self.max_nr_env_interactions:

            ep_reward, i_step, done = self._run_episode(self.max_nr_env_interactions, max_nr_steps, batch_size)

            if eval and (i_ep % eval_freq == 0):

                eval_rewards = np.mean(evaluate_agent(self, self.env, n_games=n_interact_2_evaluate,
                                                      greedy=False, verbosity=verbosity, render=False,
                                                      max_step=max_nr_steps))

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
                out_str += " elapsed time: %i" % (time.time()-time_start)

                print out_str
            i_ep += 1

    def __repr__(self):
        return "DQN"



class PieceWiseLinearEpsDecay(object):

    def __init__(self, eps, eps_annealing_frames=1000000, eps_final=0.1, eps_final_frame=0.01, max_nr_env_steps=1000000, buffer_start_size=50000):

        self.eps_initial = eps
        self.eps_annealing_frames = eps_annealing_frames

        # OLD
        # self.eps_decay = eps_decay
        # self.eps_min = eps_min
        # self.eps_start = eps
        # TODO make annealer funciton...?
        # as suggested by: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
        # https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
        self.replay_memory_start_size = buffer_start_size

        self.slope = -(self.eps_initial - eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = (eps_final_frame - eps_final) / (max_nr_env_steps - (self.eps_annealing_frames + self.replay_memory_start_size))
        self.intercept_2 = eps_final_frame - self.slope_2 * max_nr_env_steps

    def __call__(self, nr_env_interactions):

        # as suggested by: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb

        if nr_env_interactions < self.replay_memory_start_size:
            eps = self.eps_initial

        elif (nr_env_interactions >= self.replay_memory_start_size) and (nr_env_interactions < (self.replay_memory_start_size + self.eps_annealing_frames)):
            eps = self.slope * nr_env_interactions + self.intercept

        else: #  nr_env_interactions >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * nr_env_interactions + self.intercept_2

        return eps


        # OLD
        # self.eps = max(self.eps*self.eps_decay,self.eps_min)
        # self.eps = self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay * self.train_step)

def make_agent(env, sess, eps_init):
    # 1000000

    buffer = ReplayBuffer(max_buffer_size=70000, buffer_start_size=20000)

    # eps_decay_func = PieceWiseLinearEpsDecay(eps_init, eps_annealing_frames=1000000,
    #                                          eps_final=0.1, eps_final_frame=0.01,
    #                                          max_nr_env_steps=1000000, buffer_start_size=50000) ORG

    eps_decay_func = PieceWiseLinearEpsDecay(1.0, eps_annealing_frames=1e6,
                                             eps_final=0.1, eps_final_frame=0.01,
                                             max_nr_env_steps=3e6, buffer_start_size=20000)

    agent = DQN(env=env, sess=sess, eps=eps_init, eps_decay_func=eps_decay_func, buff_obj=buffer,
                gamma=0.99, tau=0, max_nr_env_interactions=10000000, use_conv_net=True, lr=1e-4) #0.0000625)

    return agent

def main():

    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make("BreakoutDeterministic-v4")

    # env = PreprocessAtari(env)
    env = ProcessFrame84(env)
    env = FrameBuffer(env)

    # env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = DQN(env, sess, eps=1.0, max_buffer_size=100000, gamma=0.99, lr=1e-4, tau=0.01, batch_size=32, use_conv_net=True)

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(10000, verbosity=1, eval=True, eval_freq=10, n_interact_2_evaluate=3)

    agent.close()

if __name__=="__main__":

    # main()

    import matplotlib.pyplot as plt


    eps_decay_func = PieceWiseLinearEpsDecay(1.0, eps_annealing_frames=1e6,
                                             eps_final=0.1, eps_final_frame=0.01,
                                             max_nr_env_steps=3e5, buffer_start_size=20000)

    x = [eps_decay_func(i)  for i in range(0,3*10**6,100)]

    plt.figure(1)
    plt.plot(range(0,3*10**6,100),x)
    plt.show()
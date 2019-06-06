from DQN import DQN
import tensorflow as tf
import numpy as np
from utils.buffers import PrioritizedReplayBuffer


class PERDQN(DQN):

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net=False, eps_decay =0.99, eps_min=0.05):

        super(PERDQN,self).__init__(env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net, eps_decay, eps_min)

        self.buff = PrioritizedReplayBuffer(max_buffer_size=max_buffer_size)
        self.buff.PER_b_increment_per_sampling = 0.000001

    def _init_ph(self):

        self.state_ph = tf.placeholder(tf.float32,(None,) + self.s_dim, "state")
        self.nxt_state_ph = tf.placeholder(tf.float32, (None,) + self.s_dim, "nxt_state")
        self.reward_ph = tf.placeholder(tf.float32, (None,), "reward")
        self.action_ph = tf.placeholder(tf.int32, (None,), "action")
        self.is_done_ph = tf.placeholder(tf.float32, (None,), "is_done")
        self.IS_weights_ph = tf.placeholder(tf.float32, (None,), "IS_weights")

    def _init_losses(self, lr):

        nxt_state_q = tf.reduce_max(self.target_net.output, axis=-1)

        self.target_q_values = self.reward_ph + self.gamma * nxt_state_q * (1.0-self.is_done_ph)

        self.q_values_state = tf.reduce_sum(tf.multiply(self.learning_net.output, tf.one_hot(self.action_ph, self.n_actions)),axis=1)

        self.abs_error = tf.math.abs(tf.stop_gradient(self.target_q_values) - self.q_values_state)

        self.loss = tf.reduce_mean(self.IS_weights_ph*tf.math.square(self.abs_error))

        self.train_opt = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def _init_tb_summaries(self):

        with tf.variable_scope("summaries"):

            self.critic_loss_var = tf.Variable(0.0, name="critic_loss")
            self.train_reward_var = tf.Variable(0.0, name="train_reward")
            self.eval_reward_var = tf.Variable(0.0, name="eval_reward")
            self.n_steps_var = tf.Variable(0.0, name="nr_eps_steps")
            self.mean_Q_var = tf.Variable(0.0, name="mean_Q_value")
            self.max_Q_var = tf.Variable(0.0, name="max_Q_value")
            self.eps_var = tf.Variable(0.0, name="eps")
            self.PER_b_var = tf.Variable(0.0, name="PER_b")

            # self.mean_abs_grad_var = tf.Variable(0.0, name="mean_abs_grad")

            self.merged = tf.summary.merge([tf.summary.scalar("critic_loss", self.critic_loss_var),
                                            tf.summary.scalar("eps", self.eps_var),
                                            tf.summary.scalar("mean_Q", self.mean_Q_var),
                                            tf.summary.scalar("max_Q", self.max_Q_var),
                                            tf.summary.scalar("PER_b", self.PER_b_var)
                                            ])

            self.eps_sum = tf.summary.merge([tf.summary.scalar("train_reward", self.train_reward_var),
                                             tf.summary.scalar("n_steps", self.n_steps_var)
                                             ]
                                            )

            self.eval_summary = tf.summary.scalar("eval_reward", self.eval_reward_var)


    def train(self, state_mb, action_mb, reward_mb, is_done_mb, nxt_state_mb, IS_weights_mb):

        feed_dict = {self.state_ph:state_mb,
                     self.nxt_state_ph: nxt_state_mb,
                     self.action_ph: action_mb,
                     self.reward_ph: reward_mb,
                     self.is_done_ph: is_done_mb,
                     self.IS_weights_ph:IS_weights_mb
                     }

        _, loss, qs, priority = self.sess.run([self.train_opt, self.loss, self.q_values_state, self.abs_error], feed_dict)

        return loss, qs, priority

    def get_priority(self,state_mb, action_mb, reward_mb, is_done_mb, nxt_state_mb):

        feed_dict = {self.state_ph: state_mb,
                     self.nxt_state_ph: nxt_state_mb,
                     self.action_ph: action_mb,
                     self.reward_ph: reward_mb,
                     self.is_done_ph: is_done_mb
                     }

        priority = self.sess.run(self.abs_error, feed_dict)

        return priority

    def __repr__(self):
        return "PERDQN"

    def _run_episode(self, max_nr_steps, train=False):

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

                state_mb, action_mb, reward_mb, done_mb, nxt_state_mb, IS_mb, IS_idx_mb = buff.sample(batch_size)

                critic_loss, qs, prio = self.train(state_mb, action_mb, reward_mb, done_mb, nxt_state_mb, IS_mb)

                buff.batch_update(IS_idx_mb, prio)

                f_dict = {self.critic_loss_var: critic_loss,
                          self.eps_var:self.eps,
                          self.max_Q_var:np.max(qs),
                          self.mean_Q_var:np.mean(qs),
                          self.PER_b_var:self.buff.PER_b
                          }

                summary = self.sess.run(self.merged, f_dict)

                self.writer.add_summary(summary, self.nr_env_interactions)

                self.writer.flush()

                self.copy_network_parameters()  # TODO: all the time?

                self.train_step += 1

            s = s_nxt
            acc_reward += r

            i_step += 1
            self.nr_env_interactions += 1

        if train:
            self.decay_eps()

        return acc_reward, i_step
from DQN import DQN
import tensorflow as tf

class DoubleDQN(DQN):

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net=False, eps_decay =0.99, eps_min=0.05):

        super(DoubleDQN,self).__init__(env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net, eps_decay, eps_min)

    def _init_ph(self):

        self.state_ph = tf.placeholder(tf.float32,(None,) + self.s_dim, "state")
        self.nxt_state_ph = tf.placeholder(tf.float32, (None,) + self.s_dim, "nxt_state")
        self.nxt_action_ph = tf.placeholder(tf.int32, (None,), "nxt_action")
        self.reward_ph = tf.placeholder(tf.float32, (None,), "reward")
        self.action_ph = tf.placeholder(tf.int32, (None,), "action")
        self.is_done_ph = tf.placeholder(tf.float32, (None,), "is_done")
        self.target_q_ph = tf.placeholder(tf.float32, (None,), "is_done")

    def _init_losses(self, lr):

        nxt_state_q = tf.reduce_sum(tf.multiply(self.target_net.output, tf.one_hot(self.nxt_action_ph,self.n_actions)), axis=-1)

        self.target_q_values = self.reward_ph + self.gamma * nxt_state_q * (1.0-self.is_done_ph)

        self.q_values_state = tf.reduce_sum(tf.multiply(self.learning_net.output, tf.one_hot(self.action_ph, self.n_actions)),axis=1)

        self.q_values_argmax = tf.math.argmax(self.learning_net.output, axis=1)

        self.loss = tf.reduce_mean((tf.stop_gradient(self.target_q_values) - self.q_values_state)**2.0)

        self.train_opt = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self, state_mb, action_mb, reward_mb, is_done_mb, nxt_state_mb):

        nxt_action_mb = self.sess.run(self.q_values_argmax, {self.state_ph: nxt_state_mb})

        feed_dict = {self.state_ph:state_mb,
                     self.nxt_state_ph: nxt_state_mb,
                     self.action_ph: action_mb,
                     self.reward_ph: reward_mb,
                     self.is_done_ph: is_done_mb,
                     self.nxt_action_ph:nxt_action_mb
                     }

        _, loss, qs = self.sess.run([self.train_opt, self.loss, self.q_values_state], feed_dict)

        return loss, qs

    def __repr__(self):
        return "DoubleDQN"

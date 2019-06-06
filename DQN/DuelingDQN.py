from DQN import DQN
import tensorflow as tf


class DuelingConvNet(object):

    def __init__(self, name, state_ph, n_actions):

        with tf.variable_scope(name):

            self.conv1 = tf.layers.conv2d(inputs=state_ph,
                                          filters=16,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=32,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.state_value_fc = tf.layers.dense(inputs=self.flatten,
                                      units=256,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="value_fc")

            self.state_value = tf.layers.dense(inputs=self.state_value_fc,
                                 units=1,
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="state_value")

            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                            units=256,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                                units=n_actions,
                                                activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage")

            self.output = self.state_value - tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=-1, keep_dims=True))

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DuelingDQN(DQN):

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net=False, eps_decay =0.99, eps_min=0.05):

        super(DuelingDQN,self).__init__(env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net, eps_decay, eps_min)


    def _init_ph(self):

        self.state_ph = tf.placeholder(tf.float32,(None,) + self.s_dim, "state")
        self.nxt_state_ph = tf.placeholder(tf.float32, (None,) + self.s_dim, "nxt_state")
        self.reward_ph = tf.placeholder(tf.float32, (None,), "reward")
        self.action_ph = tf.placeholder(tf.int32, (None,), "action")
        self.is_done_ph = tf.placeholder(tf.float32, (None,), "is_done")

    def _init_net(self, use_conv_net):

        if use_conv_net:
            self.learning_net = DuelingConvNet("learner", self.state_ph, self.n_actions)
            self.target_net = DuelingConvNet("target", self.nxt_state_ph, self.n_actions)

        else:
            raise NotImplementedError("")

    def __repr__(self):
        return "DuelingDQN"
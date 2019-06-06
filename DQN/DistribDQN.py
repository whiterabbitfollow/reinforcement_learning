from DQN import DQN
import tensorflow as tf
from utils.buffers import ReplayBuffer

# def atari_model():
#     model = cnn_to_dist_mlp(
#             convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
#             hiddens=[512])
#     return model


class DistribConvNet(object):

    def __init__(self, name, state_ph, n_actions, n_atoms, z_vec):

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

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=256,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(), units=n_actions*n_atoms, activation=None)

            out = tf.reshape(output, shape=[-1, n_actions, n_atoms])

            self.action_distribution = tf.nn.softmax(out, axis=-1, name='softmax')

            self.output = tf.tensordot(self.action_distribution, z_vec, (-1, -1))  # TODO check
            # tf.tensordot(self.learning_net, self.z, [[-1], [-1]])

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DistribDQN(DQN):

    # inspired by https://github.com/Silvicek/distributional-dqn

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net=False, eps_decay =0.99, eps_min=0.05, n_atoms=51, Vmin=-10, Vmax=10):

        self.buff = ReplayBuffer(max_buffer_size=max_buffer_size)

        self.n_atoms = n_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self._init_z()
        super(DistribDQN, self).__init__(env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net, eps_decay, eps_min)

        # TODO add Pri buff

    def _init_ph(self):

        self.state_ph = tf.placeholder(tf.float32,(None,) + self.s_dim, "state")
        self.nxt_state_ph = tf.placeholder(tf.float32, (None,) + self.s_dim, "nxt_state")
        self.reward_ph = tf.placeholder(tf.float32, (None,), "reward")
        self.action_ph = tf.placeholder(tf.int32, (None,), "action")
        self.is_done_ph = tf.placeholder(tf.float32, (None,), "is_done")
        self.target_q_ph = tf.placeholder(tf.float32, (None,), "target_q")
        self.batch_dim = tf.shape(self.reward_ph)[0]

    def _init_net(self,use_conv_net):

        if use_conv_net:

            self.learning_net = DistribConvNet("learner", self.state_ph, self.n_actions, self.n_atoms, self.z)
            self.target_net = DistribConvNet("target", self.nxt_state_ph, self.n_actions, self.n_atoms, self.z)

        else:

            raise NotImplementedError()

    def _init_losses(self, lr):

        self._init_atoms_projection()

        cat_idx = tf.transpose(tf.reshape(tf.concat([tf.range(self.batch_dim), self.action_ph], axis=0),
                                          [2, self.batch_dim]))

        choosen_action_distribution = tf.gather_nd(self.learning_net.action_distribution, cat_idx)

        self.q_values_state  = tf.gather_nd( self.learning_net.output, cat_idx)

        # could add tf.stop_gradient()

        cross_entropy = - self.ThTz * tf.log(choosen_action_distribution)  # works ??

        self.loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=-1))

        # compute optimization op (potentially with gradient clipping)

        self.train_opt = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=self.learning_net.weights)




    def _init_atoms_projection(self):

        # Taken from https://github.com/Silvicek/distributional-dqn
        """
        Builds the vectorized cathegorical algorithm following equation (7) of
        'A Distributional Perspective on Reinforcement Learning' - https://arxiv.org/abs/1707.06887
        """
        z, dz = self.z, self.dz

        a_nxt = tf.argmax(self.target_net.output, axis=-1, output_type=tf.int32)

        Vmin, Vmax, nb_atoms = self.Vmin, self.Vmax, self.n_atoms

        batch_dim = self.batch_dim

        with tf.variable_scope('cathegorical'):

            cat_idx = tf.transpose(tf.reshape(tf.concat([tf.range(batch_dim), a_nxt], axis=0), [2, batch_dim]))
            a_nxt_distributions = tf.gather_nd(self.target_net.action_distribution, cat_idx)

            big_z = tf.reshape(tf.tile(z, [batch_dim]), [batch_dim, nb_atoms])
            big_r = tf.reshape(tf.tile(self.reward_ph, [nb_atoms]), [batch_dim, nb_atoms])  # CHANGE

            Tz = tf.clip_by_value(big_r + self.gamma * tf.einsum('ij,i->ij', big_z, 1. - self.is_done_ph), Vmin, Vmax)

            big_Tz = tf.reshape(tf.tile(Tz, [1, nb_atoms]), [-1, nb_atoms, nb_atoms])
            big_big_z = tf.reshape(tf.tile(big_z, [1, nb_atoms]), [-1, nb_atoms, nb_atoms])

            Tzz = tf.abs(big_Tz - tf.transpose(big_big_z, [0, 2, 1])) / dz
            Thz = tf.clip_by_value(1 - Tzz, 0, 1)

            self.ThTz = tf.einsum('ijk,ik->ij', Thz, a_nxt_distributions)

            self.p_best = a_nxt_distributions


    def _init_z(self):

        # taken from https://github.com/Silvicek/distributional-dqn

        self.dz = (self.Vmax - self.Vmin) / (self.n_atoms - 1.0)
        self.z = tf.range(self.Vmin, self.Vmax + self.dz / 2.0, self.dz, dtype=tf.float32, name='z')

    def __repr__(self):
        return "DistribDQN"


def main():

    import gym
    from utils.buffers import PreprocessAtari, FrameBuffer

    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make("BreakoutDeterministic-v4")

    env = PreprocessAtari(env)
    env = FrameBuffer(env)

    # env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    agent = DistribDQN(env, sess, eps=1.0, max_buffer_size=100000, gamma=0.99, lr=1e-4, tau=0.01, batch_size=32,
                use_conv_net=True, n_atoms=51, Vmin=-10, Vmax=10)

    sess.run(tf.global_variables_initializer())

    agent.run_episodes(1, verbosity=1, eval=True, eval_freq=10, n_interact_2_evaluate=3)

    agent.close()


if __name__ == "__main__":

    main()
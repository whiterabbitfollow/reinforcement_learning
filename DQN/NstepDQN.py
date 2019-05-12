from DQN import DQN
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf


class NstepDQN(DQN):

    N = 4

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, N=4, use_conv_net=False, eps_decay =0.99, eps_min=0.05):

        super(NstepDQN,self).__init__(env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net, eps_decay, eps_min)
        self.N = N

    def _init_losses(self, lr):


        nxt_state_q = tf.reduce_max(self.target_net.output, axis=-1)

        self.target_q_values = self.reward_ph + (self.gamma**self.N) * nxt_state_q * (1.0-self.is_done_ph)

        self.q_values_state = tf.reduce_sum(tf.multiply(self.learning_net.output, tf.one_hot(self.action_ph, self.n_actions)),axis=1)

        self.loss = tf.reduce_mean((tf.stop_gradient(self.target_q_values) - self.q_values_state)**2.0)

        self.train_opt = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def _run_episode(self, max_nr_steps, train=False):

        env = self.env
        batch_size = self.batch_size
        buff = self.buff

        done = False
        s = env.reset()
        i_step = 0
        acc_reward = 0
        traj_N = deque([],maxlen=self.N)

        TransitionTuple = namedtuple("Transition",["s", "a", "r_tp1", "s_tp1", "done"])

        weights = [self.gamma**(i) for i in range(self.N)]

        while i_step < max_nr_steps and not done:

            a = self.policy([s])[0]

            s_nxt, r, done, _ = env.step(a)

            if i_step >= self.N:

                r_N = sum([tran.r_tp1*weights[i] for i,tran in enumerate(traj_N)])

                s_t_N, a_t_N, r_t_Np1, s_t_Np1, done_t_Np1 = traj_N[-1]

                s_t, a_t, r_tp1, s_tp1, done_tp1  = traj_N[0]

                buff.add(s_t, a_t, r_N, done_t_Np1, s_t_Np1)

            if train:

                state_mb, action_mb, reward_mb, done_mb, nxt_state_mb = buff.sample(batch_size)

                critic_loss, qs = self.train(state_mb, action_mb, reward_mb, done_mb, nxt_state_mb)

                f_dict = {self.critic_loss_var: critic_loss,
                          self.eps_var: self.eps,
                          self.max_Q_var: np.max(qs),
                          self.mean_Q_var: np.mean(qs)
                          }

                summary = self.sess.run(self.merged, f_dict)

                self.writer.add_summary(summary, self.nr_env_interactions)

                self.writer.flush()

                self.copy_network_parameters()  # TODO: all the time?

            traj_N.append(TransitionTuple(s, a, r, s_nxt, done))

            s = s_nxt
            acc_reward += r

            i_step += 1

            self.nr_env_interactions += 1

        if done:

            s_t_N, a_t_N, r_t_Np1, s_t_Np1, done_t_Np1 = traj_N[-1]

            for j in range(len(traj_N)-1):

                r_N = sum([traj_N[i].r_tp1*weights[i-j] for i in range(j,self.N)])

                s_t, a_t, r_tp1, done_tp1, s_tp1 = traj_N[j]

                buff.add(s_t, a_t, r_N, done_t_Np1, s_t_Np1)

            self.decay_eps()

        return acc_reward, i_step

    def __repr__(self):

        return "DQN(%i)"%(self.N)
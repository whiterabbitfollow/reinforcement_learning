from DQN import DQN
import tensorflow as tf
from utils.buffers import PrioritizedReplayBuffer

class DuelingDQN(DQN):

    def __init__(self, env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net=False, eps_decay =0.99, eps_min=0.05):

        super(DuelingDQN,self).__init__(env, sess, eps, max_buffer_size, gamma, lr, tau, batch_size, use_conv_net, eps_decay, eps_min)

        self.buff = PrioritizedReplayBuffer(max_buffer_size=max_buffer_size)


    def __repr__(self):
        return "DuelingDQN"

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

                # self.copy_network_parameters()  # TODO: all the time?

                if self.nr_env_interactions%500==0:
                    self.hard_copy_network_parameters()

            s = s_nxt
            acc_reward += r

            i_step += 1
            self.nr_env_interactions += 1

        if train:
            self.decay_eps()

        return acc_reward, i_step
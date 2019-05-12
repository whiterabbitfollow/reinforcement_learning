from DQN import DQN, DoubleDQN


from utils.buffers import PreprocessAtari, FrameBuffer
import gym
import tensorflow as tf

env = gym.make("BreakoutDeterministic-v4")

env = PreprocessAtari(env)

env = FrameBuffer(env)

tf.reset_default_graph()

sess = tf.InteractiveSession()

# agent = DQN(env, sess, eps=0.85, max_buffer_size=100000, gamma=0.99,
#             lr=1e-4, tau=0.01, batch_size=32, use_conv_net=True,
#             eps_decay=0.9995)

# agent = NstepDQN(env, sess, eps=0.85, max_buffer_size=100000, gamma=0.99,
#             lr=1e-4, tau=0.01, batch_size=32, use_conv_net=True,
#             eps_decay=0.9999,N=4)


agent = DoubleDQN(env, sess, eps=0.5, max_buffer_size=100000, gamma=0.99,
                  lr=1e-4, tau=0.01, batch_size=32, use_conv_net=True, eps_decay=0.9999)

sess.run(tf.global_variables_initializer())

agent.run_episodes(10000, verbosity=1, eval=True, eval_freq=10, n_interact_2_evaluate=3)

agent.close()



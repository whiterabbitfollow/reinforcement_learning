import os
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import gym

def get_run_nr(file_dir,starts_with="run"):

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    run_nr = 0

    for d in os.listdir(file_dir):
        if d.startswith(starts_with+"_"):
            run_nr += 1

    return run_nr


class BaseDeepAgent(object):

    __metaclass__ = ABCMeta

    def __init__(self, env, sess, log_data=True):

        self.env = env
        self.sess = sess
        self.nr_env_interactions = 0
        self.action_bound = None
        self.n_actions = None
        self.a_dim = None
        self.is_discrete = False

        path_2_main = "/home/x/Documents/reinforcement_learning/open_ai_gym/tmp/reinforcement_learning" # TODO fix hardcoded path

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
            self.is_discrete = True
        else:
            self.a_dim = env.action_space.shape[0]
            self.action_bound = abs(env.action_space.low)

        self.s_dim = env.observation_space.shape[0]

        if log_data:
            self.env_name = self.env.spec.id
            run_nr = get_run_nr(os.path.join(path_2_main,"runs_" + self.env_name), starts_with=str(self))
            self.writer = tf.summary.FileWriter(os.path.join(path_2_main,"runs_%s/%s_%i" % (self.env_name, str(self),run_nr)), sess.graph)

    @abstractmethod
    def _init_tb_summaries(self):
        NotImplementedError()

    @abstractmethod
    def policy(self, state):
        NotImplementedError()

    @abstractmethod
    def train(self):
        NotImplementedError()

    @abstractmethod
    def run_episodes(self,n_episodes,verbosity=0):
        NotImplementedError()

    def close(self):
        self.sess.close()
        self.writer.close()

class BaseActor(object):

    __metaclass__ = ABCMeta

    def __init__(self, env, sess, log_data=True):
        pass
    @abstractmethod
    def policy(self, state):
        NotImplementedError()

    @abstractmethod
    def train(self):
        NotImplementedError()


class BaseCritic(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()




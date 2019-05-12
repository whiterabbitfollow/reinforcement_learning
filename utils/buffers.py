import numpy as np

from collections import deque
import random
from abc import ABCMeta, abstractmethod

from gym.spaces.box import Box
from gym.core import Wrapper


from gym.core import ObservationWrapper
from gym.spaces import Box

# from scipy.misc import imresize
import cv2



class SumTree:

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1 )
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):

        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    @property
    def total_priority(self):
        return self.tree[0]

    @property
    def min(self):
        return np.min(self.tree[-self.capacity:])

    @property
    def max(self):
        return np.max(self.tree[-self.capacity:])



class BaseReplayBuffer(object):

    __metaclass__ = ABCMeta

    def __init__(self,max_buffer_size):
        self.size = 0
        self.max_size = max_buffer_size

    @abstractmethod
    def add(self):
        NotImplementedError()

    def init_ratio(self):
        return float(self.size)/(self.max_size * 0.2)  # hmm scale...

    def have_stored_enough(self):
        return self.size >= self.max_size*0.2 # hmm scale...

    @abstractmethod
    def sample(self, batch_size):
        NotImplementedError()

class PrioritizedReplayBuffer(BaseReplayBuffer):

    PER_e = 0.01    # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6     # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4     # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error


    def __init__(self,max_buffer_size):

        super(PrioritizedReplayBuffer,self).__init__(max_buffer_size)
        self.buffer = SumTree(self.max_size)


    def store(self, s, a, r, d, s_nxt):

        experience = (s, a, r, d, s_nxt)

        max_priority = self.buffer.max

        if max_priority == 0:

            max_priority = self.absolute_error_upper

        self.buffer.add(max_priority, experience)

        self.size  = min(self.size+1,self.max_buffer_size)


    def add(self, s, a, r, d, s_nxt, p):

        experience = (s, a, r, d, s_nxt)

        self.buffer.add(p,experience)

        self.size += 1

        print self.size

    def anneal(self):
        self.PER_b = min(1., self.PER_b + self.PER_b_increment_per_sampling)

    def sample(self, batch_size):

        b_idx, b_IS_weights = [], []

        priority_edge = self.buffer.total_priority / batch_size

        self.anneal()

        p_min = self.buffer.min / self.buffer.total_priority

        max_weight = (p_min * batch_size) ** (-self.PER_b)

        state_mb, action_mb, reward_mb, done_mb, nxt_state_mb = [], [], [], [], []

        for i in range(batch_size):

            a, b = priority_edge * i, priority_edge * (i + 1)

            value = np.random.uniform(a, b)

            idx, priority, data = self.buffer.get(value)

            # P(i)
            P_i = priority / self.buffer.total_priority

            # IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi

            b_IS_weights.append(np.power(batch_size * P_i, -self.PER_b) / max_weight) # why do we normalize with max_wi ??
            b_idx.append(idx)

            s, a, r, d, nxt_s = data

            state_mb.append(s)
            action_mb.append(a)
            reward_mb.append(r)
            done_mb.append(d)
            nxt_state_mb.append(nxt_s)

        return np.array(state_mb), np.array(action_mb), np.array(reward_mb),\
               np.array(done_mb), np.array(nxt_state_mb), np.array(b_IS_weights), np.array(b_idx)


    def batch_update(self,tree_idx, abs_error):

        clipped_errors = np.minimum(abs_error + self.PER_e, self.absolute_error_upper)

        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):

            self.buffer.update(ti, p)


class ReplayBuffer(BaseReplayBuffer):

    def __init__(self,max_buffer_size):

        super(ReplayBuffer,self).__init__(max_buffer_size)
        self.buffer = deque()

    def add(self,s, a, r, d, s_nxt):

        experience = (s, a, r, d, s_nxt)

        if self.size >= self.max_size:
            self.buffer.popleft()
        else:
            self.size += 1

        self.buffer.append(experience)

    def sample(self,batch_size):

        batch = random.sample(self.buffer,min(batch_size,self.size))
        state_mb, action_mb, reward_mb, done_mb, nxt_state_mb  = [], [], [], [], []

        for s, a, r, d, nxt_s in batch:
            state_mb.append(s)

            action_mb.append(a)
            reward_mb.append(r)
            done_mb.append(d)
            nxt_state_mb.append(nxt_s)

        return np.array(state_mb), np.array(action_mb), np.array(reward_mb), np.array(done_mb), np.array(nxt_state_mb)



class FrameBuffer(Wrapper):


    def __init__(self, env, n_frames=4, dim_order='tensorflow'):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)

        self.dim_order = dim_order

        if dim_order == 'tensorflow':
            height, width, n_channels = env.observation_space.shape
            obs_shape = [height, width, n_channels * n_frames]
        elif dim_order == 'pytorch':
            n_channels, height, width = env.observation_space.shape
            obs_shape = [n_channels * n_frames, height, width]
        else:
            raise ValueError('dim_order should be "tensorflow" or "pytorch", got {}'.format(dim_order))

        self.observation_space = Box(0.0, 1.0, obs_shape)

        self.framebuffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        if self.dim_order == 'tensorflow':
            offset = self.env.observation_space.shape[-1]
            axis = -1
            cropped_framebuffer = self.framebuffer[:, :, :-offset]
        elif self.dim_order == 'pytorch':
            offset = self.env.observation_space.shape[0]
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset]

        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)



class PreprocessAtari(ObservationWrapper):

    def __init__(self, env):

        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (64, 64)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):

        """what happens to each observation"""

        #img = img[34:-16:, :]

        img = img[34:-16, :, :]

        # resize image
        img = cv2.resize(img, self.img_size)

        # # grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # img = img[:,:,np.newaxis]
        #
        img = img.mean(-1, keepdims=True)
        # convert pixels to range (0,1)
        #         img = img/255
        img = img.astype('float32') / 255.

        return img

if __name__ == "__main__":

    import gym
    import time
    import matplotlib.pyplot as plt

    env = gym.make("PongNoFrameskip-v4")
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env)

    s = env.reset()

    #
    #
    #
    #
    # for i in range(10000):
    #
    #     # s = s[39:-16, :, :]
    #
    #     plt.imshow(s[:,:,0],cmap="gray")
    #     plt.pause(0.01)
    #
    #     s,_,_,_ = env.step(env.action_space.sample())
    #
    #     # env.render()
    #     # time.sleep(0.1)
    is_done = False
    while not is_done:

        obs, reward, is_done, _ = env.step(env.action_space.sample())

        print reward, is_done

        plt.figure(1)
        plt.clf()
        plt.imshow(env.render("rgb_array"))  # ,cmap="gray")
        plt.pause(0.01)

    # obs, _, _, _ = env.step(env.action_space.sample())
    # obs, _, _, _ = env.step(env.action_space.sample())
    # obs, _, _, _ = env.step(env.action_space.sample())
    # obs, _, _, _ = env.step(env.action_space.sample())

    # plt.title("Game image")
    #

    state_dim = env.observation_space.shape
    plt.figure(1)
    plt.title("Agent observation (4 frames left to right)")
    plt.imshow(obs.transpose([0, 2, 1]).reshape([state_dim[0], -1])) # ,cmap="gray")
    plt.show()

    plt.figure(2)
    plt.title("Agent observation (4 frames left to right)")
    plt.imshow(env.render("rgb_array"))  # ,cmap="gray")
    plt.show()




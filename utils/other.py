import gym
import numpy as np

def compute_discounted_return(rewards, gamma=0.95, baseline=False):

    discounted_rewards = []
    acc_r = 0.0

    for r in reversed(rewards):
        acc_r = r + gamma*acc_r
        discounted_rewards.append(acc_r)

    if baseline:
        discounted_rewards = (discounted_rewards- np.mean(discounted_rewards))/np.std(discounted_rewards)

    return list(reversed(discounted_rewards))

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def evaluate_agent(agent, env, n_games=1,greedy=False,verbosity=0,render=False,max_step=500):

    game_rewards = []

    if verbosity > 0:
        print "--- Eval ---"

    for i_game in range(n_games):

        state = env.reset()
        total_reward = 0
        i_step = 0
        is_done = False

        while not is_done and i_step<=max_step:

            if greedy:
                action = agent.policy([state],greedy)[0]
            else:
                action = agent.policy([state])[0]

            state, reward, is_done, _ = env.step(action)

            total_reward += reward

            i_step += 1

            if verbosity > 1:
                print "i_game: %i i_step: %i acc_reward:%f is_done: %i"%(i_game, i_step, total_reward, is_done)

            if render:
                env.render()

        if verbosity > 0:
            print "game %i reward: %f nr_steps: %i "%(i_game,total_reward,i_step)

        game_rewards.append(total_reward)

    if render:
        env.close()

    if verbosity > 0:
        print "mean: %f std: %f max:%f min:%f" % (np.mean(game_rewards), np.std(game_rewards), np.max(game_rewards), np.min(game_rewards))

    return game_rewards


class EnvBatch(object):

    def __init__(self, name, n_envs=10):
        self.envs_ = [gym.make(name) for i in range(n_envs)]

    def reset(self):
        return np.array([env.reset() for env in self.envs_])

    def step(self, actions):
        results_ = [env.step(action) for action, env in zip(actions, self.envs_)]
        nxt_state, rewards, done, _ = map(np.array, zip(*results_))
        for i, d in enumerate(done):
            if d:
                nxt_state[i, :] = self.envs_[i].reset()
        return nxt_state, rewards, done

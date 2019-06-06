# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import random

class BaseValueAgent:
    
    def __init__(self,env,gamma=0.99,eps=0.5,eps_decay=1.0,base_line=1.0,min_eps=0.5):
        self.env = env
        self.n_obs = env.observation_space.n
        self.n_actions = env.action_space.n
        self._Q = defaultdict(lambda: [base_line]*self.n_actions)
        self._V = np.zeros(self.n_obs)
        self._policy = defaultdict(lambda: [1.0/self.n_actions]*self.n_actions)        
        self.gamma = gamma                
        self.eps = eps
        self.min_eps = min(min_eps,eps)
        self.eps_decay = eps_decay
        self.no_learning_steps = 0
        self.best_acc_reward = -99999999

    def decay_eps(self):
        self.eps = max(self.eps*self.eps_decay,self.min_eps)
    
    def evaluate(self,render=False,verbosity=0):
            
            done = False    
            max_no_steps_in_same_state = 10
            no_steps_in_same_state = 0
            step_no = 0
            max_no_steps = 50
            agent_history = []
            
            s = self.env.reset()
            agent_history.append(s)
            acc_reward = 0
            
            while not done and no_steps_in_same_state<max_no_steps_in_same_state and step_no<max_no_steps:
                
                a_max = np.argmax(self._Q[s])
                
                if verbosity:
                    print self._Q[s], a_max, s
                
                s_nxt, r, done, _ = self.env.step(a_max)
                
                if render:
                    self.env.render()
            
                step_no += 1
                
                if s == s_nxt:
                    no_steps_in_same_state += 1
                else:
                    no_steps_in_same_state = 0
                    
                s = s_nxt 
                acc_reward += r
                agent_history.append(s)
            
            if done:
                self.best_acc_reward = max(self.best_acc_reward, acc_reward)

            else:
                acc_reward = -999

            return agent_history, acc_reward

    def Q(self,state,action):
        return self._Q[state][action]
        
    def get_state_value_function(self):
        
        V = np.zeros(self.n_obs)
        
        for s in range(self.n_obs):
            V[s] = np.max(self._Q[s])
        
        self._V = V        
        return V
            
    def get_state_visit(self):
        state_visit_cnt = []
        for s in sorted(self._state_visit_cnt):
            state_visit_cnt.append(sum(self._state_visit_cnt[s]))
        return state_visit_cnt


    def _write_info(self,verbosity,iter_no):
        
        if (verbosity==1 and iter_no%100==0):
            return True
        elif (verbosity==2 and iter_no%1000==0):
            return True
        elif (verbosity>3):
            return True
        else:
            return False

    def info(self,iter_no):
        return "%s > step no: %i, eps: %f, best_reward:%f, no learning steps: %i"%(str(self),
                                                                                iter_no,
                                                                                self.eps,
                                                                                self.best_acc_reward,
                                                                                self.no_learning_steps
                                                                                )


class TDBaseValueAgent(BaseValueAgent):
    
    
    def __init__(self,env,alpha=0.3,gamma=0.99,eps=0.5,eps_decay=1.0,base_line=1.0,min_eps=0.5):
        BaseValueAgent.__init__(self,env,gamma,eps,eps_decay,base_line,min_eps)
        self.alpha = alpha

        
    def update_Q(self,s,a,Q_target):
        
        Q_old = self._Q[s][a]
        
        TD_error = Q_target - self.Q(s,a)
        
        self._Q[s][a] += self.alpha*TD_error
        
        Q_new = self._Q[s][a]
        
        return Q_old-Q_new

    def run_episode(self):
        raise NotImplementedError()

    def train(self, max_no_iterations=10000, delta_thr=0.1,alpha_delta=0.3,verbosity=0):

        learning_curve_eval = []
        learning_curve_training = []
        delta_history = []

        iter_no = 0
        delta = 10
        
        while iter_no < max_no_iterations and delta > delta_thr:
            
            acc_training_reward, delta_eps = self.run_episode()
                
            self.decay_eps()
            
            _, acc_reward_eval = self.evaluate()
            
            delta = (1.0-alpha_delta) * delta +  delta_eps * alpha_delta
            
            learning_curve_eval.append(acc_reward_eval)
            learning_curve_training.append(acc_training_reward)
            delta_history.append((self.no_learning_steps, delta_eps))
            
            if self._write_info(verbosity,iter_no):
                print self.info(iter_no), delta
            
            
            iter_no += 1
            
            
        has_agent_converged = delta > delta_thr
        
        return learning_curve_training, learning_curve_eval, delta_history, has_agent_converged
        
    def policy(self,state):
        
        if random.random()>self.eps:
            return np.argmax(self._Q[state])
        else:
            return random.randint(0,self.n_actions-1)


def create_trajectory_from_history(history):

    N_COLS = 12
    rows = [cell_no//N_COLS for cell_no in history]
    cols = [cell_no - row*N_COLS for row,cell_no in zip(rows,history)]
    return rows, cols

        
        
        
        
        
        
        
        

import gym
import numpy as np
import matplotlib.pyplot as plt
from utils import TDBaseValueAgent, create_trajectory_from_history, BaseValueAgent

class QLearningAgent(TDBaseValueAgent):
    
    def __init__(self,env,alpha=0.3, gamma=0.99,eps=0.3,eps_decay=1.0,base_line=1.0,min_eps=0.5):

        TDBaseValueAgent.__init__(self,env,alpha,gamma,eps,eps_decay,base_line,min_eps)
        
    def run_episode(self):
        
        s = self.env.reset()
        done = False
        delta = -9999
        acc_reward = 0
        
        while not done:
            
            a = self.policy(s)
            
            nxt_s, r, done, _ = self.env.step(a)
            
            Q_target = self._compute_QLearning_target(s,a,r,nxt_s,done)
            
            delta_step = self.update_Q(s,a,Q_target)
            
            s = nxt_s
            self.no_learning_steps += 1
            
            delta = max(delta,delta_step)
            acc_reward += r
            
        return acc_reward, delta
    
    def _compute_QLearning_target(self,s,a,r,nxt_s,done):
        nxt_a = np.argmax(self._Q[nxt_s])        
        Q_target = r + self.gamma * self.Q(nxt_s,nxt_a) * (1-done)
        return Q_target
    
    def __repr__(self):
        return "Q_learning"
    
class SARSAAgent(TDBaseValueAgent):
    
    def __init__(self, env, alpha=0.3, gamma=0.99, eps=0.3, eps_decay=1.0, base_line=1.0, min_eps=0.5):
        
        TDBaseValueAgent.__init__(self,env,alpha,gamma,eps,eps_decay,base_line,min_eps)

    def run_episode(self):
        
        s = self.env.reset()
        done = False
        delta = -9999
        acc_reward = 0
        a = self.policy(s)
        
        while not done:
            
            nxt_s, r, done, _ = self.env.step(a)
            nxt_a = self.policy(nxt_s)
            
            Q_target = self._compute_SARSA_target(r,nxt_s,nxt_a,done)
            
            delta_step = self.update_Q(s,a,Q_target)
            
            self.no_learning_steps += 1
            delta = max(delta,np.abs(delta_step))
            acc_reward += r
            
            s = nxt_s
            a = nxt_a
        
        return acc_reward, delta
    
    def _compute_SARSA_target(self,r,nxt_s,nxt_a,done):
        Q_target = r + self.gamma * self.Q(nxt_s,nxt_a) * (1.0-done)
        return Q_target
    
    def __repr__(self):
        return "SARSA" 

class ExpectedSARSAAgent(TDBaseValueAgent):
    
    def __init__(self,env,alpha=0.3, gamma=0.99,eps=0.3,eps_decay=1.0,base_line=1.0,min_eps=0.5):

        TDBaseValueAgent.__init__(self,env,alpha,gamma,eps,eps_decay,base_line,min_eps)

    def run_episode(self):
        
        s = self.env.reset()
        done = False
        delta = -9999
        acc_reward = 0
        
        while not done:
            
            a = self.policy(s)    
            nxt_s, r, done, _ = self.env.step(a)
            
            Q_target = self._compute_ExpectedSARSA_target(r,nxt_s,done)
            
            delta_step = self.update_Q(s,a,Q_target)
            
            self.no_learning_steps += 1
            delta = max(delta,delta_step)
            acc_reward += r
            s = nxt_s
            
        return acc_reward, delta
    
    def _compute_ExpectedSARSA_target(self,r,nxt_s,done):
        
        prob = self.eps/self.n_actions
        a_max = np.argmax(self._Q[nxt_s])        
        Q_nxt_s_a = 0
        
        for nxt_a in range(self.n_actions):
            
            if nxt_a ==a_max:
                probability = prob + (1-self.eps)
            else:
                probability = prob
                
            Q_nxt_s_a += probability * self.Q(nxt_s,nxt_a)
            
        Q_target = r + self.gamma * Q_nxt_s_a * (1-done)
        
        return Q_target
    
    def __repr__(self):
        return "Expected_SARSA" 

class ValueIterationAgent(BaseValueAgent):

    def __init__(self,env,gamma=0.99):
        
        BaseValueAgent.__init__(self,env,gamma)
        
        self.P = self.env.P
        
    def train(self, max_no_iterations=10000, delta_thr=0.1,alpha_delta=0.3,  verbosity=0):
        
        delta = delta_thr + 1
        iter_no = 0

        learning_curve_eval = []
        learning_curve_training = []
        delta_history = []
            
        while iter_no < max_no_iterations and delta > delta_thr:
            
            delta = 0
            verbose = self._write_info(verbosity,iter_no)
            
            for s in self.P:
                
                P_s = self.P[s]
                v_max = -100000000
                v_old = self._V[s]

                all_V = []

                for a in P_s:

                    v = 0

                    for (p, nxt_s, reward, d) in P_s[a]:

                        v_nxt_state = self._V[nxt_s]*(1-d)

                        v += p * (reward + self.gamma * v_nxt_state)

                    all_V.append(v)

                v = np.max(all_V)

                v_max = max(v, v_max)

                self.no_learning_steps += 1
                    
                self._V[s] = v_max
                
                delta = max(delta, abs(v_old - v_max))
            
            
            self._make_policy_Q_greedy_wrt_V()
            
            _, acc_reward_eval = self.evaluate()
            
            delta_history.append((self.no_learning_steps, delta))
            learning_curve_eval.append(acc_reward_eval)
            
            
            if verbose:
                    print self.info(iter_no), delta
            
            iter_no += 1
        
        has_agent_converged = delta > delta_thr
        
        return learning_curve_training, learning_curve_eval, delta_history, has_agent_converged
    
    
    def get_state_value_function(self):
        return self._V
    
    def _make_policy_Q_greedy_wrt_V(self):
        
        for s in range(self.n_obs):
            
            v_max = -100000000
            a_max = 0
            
            for a in self.P[s]:

                v = 0

                for (p, nxt_s, reward, d) in self.P[s][a]:

                    v_nxt_state = self._V[nxt_s]*(1-d)

                    v += p * (reward + self.gamma * v_nxt_state)

                self._Q[s][a] = v
                
                if v > v_max:
                    v_max = v
                    a_max = a
                    
            self._policy[s] = a_max
            
    def __repr__(self):
        return "ValueIteration" 
            
if __name__=="__main__":
    
    env = gym.make("CliffWalking-v0")
    
    agent = SARSAAgent(env, eps=0.5, alpha=0.5, gamma=0.99)
    
    learning_curve_training, learning_curve_eval, delta_hist, has_agent_converged = agent.train(verbosity=2)
    
    history, r = agent.evaluate(render=False)
    
    plt.figure(1)
    plt.clf()
    plt.plot(learning_curve_eval)
    
    plt.figure(2)
    plt.clf()
    plt.plot(delta_hist)
    
    V =  agent.get_state_value_function()
    grid = np.array(V).reshape(4,12)
    rows, cols = create_trajectory_from_history(history)
    
    plt.figure(3)
    plt.clf()
    plt.imshow(grid)
    plt.plot(cols,rows)
    plt.colorbar()
    plt.show()
        

from utils import BaseValueAgent, create_trajectory_from_history
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gym

class MonteCarloFirstVisitAgent(BaseValueAgent):
    
    def __init__(self,env,gamma=0.99,eps=0.5,eps_decay=1.0,base_line=1.0,min_eps=0.5):
        
        BaseValueAgent.__init__(self,env,gamma,eps,eps_decay,base_line,min_eps)
        
        self._state_visit_cnt = defaultdict(lambda: [0]*self.n_actions)
        
    def generate_trajectory(self,verbosity=False):
        
        s = self.env.reset()
        trajectory = []
        done = False
        step_no = 0
        acc_reward = 0
        max_no_steps = 1000
        
        while not done and step_no<max_no_steps:
            
            a = np.random.choice(self.n_actions,p=self._policy[s])
            
            nxt_s, r, done, _ = self.env.step(a)
            
            trajectory.append((s,a,r))
            
            s = nxt_s
            step_no += 1
            acc_reward += r
            
        if verbosity:
            print "Len traj: %i"%(len(trajectory))
            
        return trajectory, acc_reward
    
    def evaluate_trajectory(self,tau):
        
        delta = 0                
        Gt = 0
        self.no_learning_steps += len(tau)
        
        for i, (s,a,r) in enumerate(reversed(tau)):
    
            Gt = r + self.gamma*Gt
            
            if (s,a,r) not in tau[0:-(i+1)]:
    
                N_old = self._state_visit_cnt[s][a]
                Q_old = self._Q[s][a]
                
                self._Q[s][a] = (Q_old*N_old + Gt)/(N_old+1)
                
                Q_new = self._Q[s][a]
                
                self._state_visit_cnt[s][a] += 1
                
                a_max = np.argmax(self._Q[s])
                
                self._policy[s] = [self.eps/self.n_actions]*self.n_actions
                self._policy[s][a_max] += 1.0 - self.eps
                
                delta = max(delta,np.abs(Q_old-Q_new))
        
        return delta
    
    
    def train(self,max_no_iterations=10000,delta_thr=0.1,alpha_delta=0.3,verbosity=0):
        
        learning_curve_eval = []
        learning_curve_training = []
        delta_history = []
        iter_no = 0
        delta = 10
        
        while iter_no < max_no_iterations and delta > delta_thr:
            
            
            verbose = self._write_info(verbosity,iter_no)
            
            traj, acc_training_reward = self.generate_trajectory(verbosity=verbose)
            
            self.decay_eps()
            
            delta_eps = self.evaluate_trajectory(traj)
            
            _, acc_reward_eval = self.evaluate()
            
            delta = (1-alpha_delta) * delta +  delta_eps * alpha_delta
            
            learning_curve_eval.append(acc_reward_eval)
            learning_curve_training.append(acc_training_reward)
            delta_history.append((self.no_learning_steps, delta_eps))
            
            iter_no += 1
            
            if verbose:
                print self.info(iter_no), delta
            
        has_agent_converged = delta > delta_thr
        
        return learning_curve_training, learning_curve_eval, delta_history, has_agent_converged
        
    def __repr__(self):
        return "First Visit Monte Carlo"
    
class MonteCarloEveryVisitAgent(MonteCarloFirstVisitAgent):
    
    def __init__(self,env,gamma=0.99,eps=0.5,eps_decay=1.0,base_line=1.0,min_eps=0.5):
        
        MonteCarloFirstVisitAgent.__init__(self,env,gamma,eps,eps_decay,base_line,min_eps)
        
        self._state_visit_cnt = defaultdict(lambda: [0]*self.n_actions)
    
    def evaluate_trajectory(self,tau):
        
        delta = 0                
        Gt = 0
        self.no_learning_steps += len(tau)
        
        for i, (s,a,r) in enumerate(reversed(tau)):
    
            Gt = r + self.gamma*Gt
            
            N_old = self._state_visit_cnt[s][a]
            Q_old = self._Q[s][a]
            
            self._Q[s][a] = (Q_old*N_old + Gt)/(N_old+1)
            
            Q_new = self._Q[s][a]
            
            self._state_visit_cnt[s][a] += 1
            
            a_max = np.argmax(self._Q[s])
            
            self._policy[s] = [self.eps/self.n_actions]*self.n_actions
            self._policy[s][a_max] += 1.0 - self.eps
            
            delta = max(delta,np.abs(Q_old-Q_new))
        
        return delta
    
    def __repr__(self):
        return "Evert Visit Monte Carlo"

if __name__=="__main__":
    
    env = gym.make("CliffWalking-v0")
    agent = MonteCarloFirstVisitAgent(env, eps=0.8, eps_decay=0.99, min_eps=0.01)
    
    learning_curve_training, learning_curve_eval, has_agent_converged = agent.train(verbosity=1)
    history, r = agent.evaluate(render=True)
    
    plt.figure(1)
    plt.plot(learning_curve_training)
    
    V =  agent.get_state_value_function()
    grid = np.array(V).reshape(4,12)
    rows, cols = create_trajectory_from_history(history)
    
    plt.figure(2)
    plt.clf()
    plt.imshow(grid)
    plt.plot(cols,rows)
    plt.colorbar()
    plt.show()
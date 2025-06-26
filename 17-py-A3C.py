import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import torch.multiprocessing as mp
import torch.nn.functional as F

N_GAMES = 3000
T_MAX = 5
#this class help us that local gradient for each worker, update global optimizer
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritics(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritics,self).__init__()

        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims,128)
        self.v1 = nn.Linear(*input_dims,128)
        self.pi = nn.Linear(128,n_actions)
        self.v = nn.Linear(128,1)

        self.actions,self.rewards,self.states = [],[],[]

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        v1 = F.relu(self.v1(x))

        pi = self.pi(pi1)
        v = self.v(v1)
        return pi, v

    def remember(self, state, action, reward):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    # first impliment other part of function to understand this
    def calc_R(self, done):
        
        states_t = torch.tensor(self.states,dtype=torch.float32)
        _,v = self.forward(states_t)

        R = v[-1] * int(1-done) # V next state
        batch_return = []
        for reward in self.rewards[::-1]: #this one goes reverse

            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float32)

    def calc_loss(self, done):

        state_t = torch.tensor(self.states)
        actions = torch.tensor(self.actions)
        
        pi, values = self.forward(state_t)
        
        dist = Categorical(pi)
        log_p = dist.log_prob(actions)

        Returns = self.calc_R(done)
        values = values.squeeze()
        Advantage = Returns - values

        actor_loss = - (log_p * Advantage)

        critics_loss = (Returns - values) **2

        total_loss = (critics_loss + actor_loss).mean()

        return total_loss
        

    def choose_action(self, observation):

         state = torch.tensor(observation)
         pi,v = self.forward(state)

         dist = Categorical(pi)
         action = dist.sample().item()

         return action

class Agent(mp.Process):
    def __init__(self,global_actor_critic,optimizer, input_dim, n_action, gamma, lr,global_idx,env_id):
        super(Agent,self).__init__()
        
        self.global_actor_critic = global_actor_critic
        self.episode_idx = global_idx
        self.env = gym.make(env_id)
        self.local_actor_critics = ActorCritics(input_dim, n_action)
        self.optimizer = optimizer
        
    def run(self):
        
        t_steps = 0
        while self.episode_idx.value  < N_GAMES:
            done = False
            obs, _ = self.env.reset()
            self.local_actor_critics.clear_memory()
            score = 0
            while not done:

                action = self.local_actor_critics.choose_action(obs)
                states,reward,done,truncate,_ = self.env.step(action)
                score += reward
                
                self.local_actor_critics.remember(states, action, reward)

                if t_steps % T_MAX ==0 or done:

                    loss = self.local_actor_critics.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(
                        self.local_actor_critics.parameters(),
                        self.global_actor_critic.parameters()):
                        global_param = local_param

                    self.optimizer.step()
                    self.local_actor_critics.load_state_dict(self.global_actor_critic.state_dict())

                t_steps +=1
                obs = states

            with self.episode_idx.get_lock():
                    self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)
    
                        
                
            
if __name__ == '__main__':
    mp.set_start_method('spawn')  # critical on Windows

    lr = 1e-4
    env_id = 'CartPole-v1'
    n_actions = 2
    input_dims = [4]

    global_actor_critic = ActorCritics(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)
    
    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    global_idx=global_ep,
                    env_id=env_id) for i in range(4)]
    [w.start() for w in workers]
    [w.join() for w in workers]
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.optim as optim
from collections import deque
import random
import numpy as np
import argparse
from Helper_bonus import softmax, argmax
import sys

parser = argparse.ArgumentParser(description="DQN with optional toggling of Experience Replay and Target Network")
parser.add_argument('--experience_replay', action='store_true', help='Enable experience replay.', dest='use_experience_replay')
parser.add_argument('--target_network', action='store_true', help='Enable target network', dest='use_target_network')
parser.set_defaults(use_experience_replay=False, use_target_network=False)

args = parser.parse_args()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # input_dim (4 for CartPole: cart position, cart velocity, pole angle, pole velocity).
        # output_dim (2 for CartPole: push left or right).
        self.input = nn.Linear(input_dim, 64)
        self.layer = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.layer(x))
        return self.output(x)

class experience_replay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)    

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        return states, actions, rewards, next_states, dones

class DQNAgent():
    def __init__(self, n_states, n_actions, learning_rate, gamma, buffer_size, batch_size):
        self.policy_net = DQN(n_states, n_actions).float()
        self.target_net = DQN(n_states, n_actions).float()
        #self.bonus_net = DQN(n_states, n_actions).int()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = experience_replay(buffer_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.temp = 1.0
    
    def select_action(self, state, policy='egreedy', epsilon=0.1, temp=None, anneal=False):
        state = torch.tensor(state, dtype=torch.float)  
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy().squeeze()
            q_next_values = self.policy_net(state+1).cpu().numpy().squeeze()
        
        if policy == 'greedy':
            action = argmax(q_values)

        elif policy == 'egreedy':
            if anneal == True:
                #epsilon = 1.0
                eps_dec = 5e-4
                eps_min = 0.01
                #print("check")
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = argmax(q_values)
                self.epsilon = self.epsilon - eps_dec if self.epsilon > eps_min \
                        else eps_min

            else:

                if epsilon is None:
                    raise KeyError("Provide an epsilon")

                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = argmax(q_values)

        elif policy == 'softmax':

            if anneal == True:
                #epsilon = 1.0
                temp_dec = 5e-4
                temp_min = 0.01
                action_probs = softmax(q_values, self.temp)
                action = np.random.choice(np.arange(self.n_actions), p=action_probs)
                self.temp = self.temp - temp_dec if self.temp > temp_min \
                    else temp_min

            else:
                if temp is None or temp <= 0:
                    raise KeyError("Provide a positive temperature")

                action_probs = softmax(q_values, temp)
                action = np.random.choice(np.arange(self.n_actions), p=action_probs)

        elif policy == 'bootstrapped':
            print(q_values)
            print(self.n_states)
            sys.exit()
            greedy_next = argmax(q_next_values)

            bonus = state

        return action
    
    def evaluate(self,env,n_eval_episodes=10):
        self.policy_net.eval()
        returns = []  
        
        for i in range(n_eval_episodes):
            s = env.reset()
            raw_state = s[0] if isinstance(s, tuple) else s
            if isinstance(raw_state, tuple):
                raw_state = np.array(raw_state)
            state = torch.from_numpy(raw_state).float().unsqueeze(0)
            R_ep = 0
            for t in range(500):
                a = self.select_action(state, 'greedy')
                action_int = a.item() if isinstance(a, torch.Tensor) else a
                state, r, done, *extras = env.step(action_int)
                R_ep += r
                if done:
                    break
            returns.append(R_ep)
        self.policy_net.train()
        return np.mean(returns)

    def direct_update(self, args, state, action, reward, next_state, done, experiment_target_network):

        state = torch.FloatTensor(state).unsqueeze(0) 
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action]).unsqueeze(1)
        reward = torch.FloatTensor([reward]).unsqueeze(0)
        done = torch.FloatTensor([done]).unsqueeze(0)
        
        curr_Q = self.policy_net(state).gather(1, action)  

        with torch.no_grad():
            if experiment_target_network or args.use_target_network:
                next_Q_values = self.target_net(next_state).max(1)[0].detach()
            else:
                next_Q_values = self.policy_net(next_state).max(1)[0].detach()
            expected_Q = reward + (self.gamma * next_Q_values * (1 - done)) 

        loss = F.mse_loss(curr_Q, expected_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def optimize_model(self, args, experiment_target_network):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

   
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        curr_Q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if experiment_target_network or args.use_target_network:
                next_Q_values = self.target_net(next_states).max(1)[0].detach()
            else:
                next_Q_values = self.policy_net(next_states).max(1)[0].detach()
            expected_Q = rewards + (self.gamma * next_Q_values * (1 - dones))

        loss = F.mse_loss(curr_Q, expected_Q.unsqueeze(1)) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()      

def DQN_Main(n_episodes, learning_rate, gamma, policy='egreedy', epsilon=0.01, temp=None, plot=True, buffer_size=10000, batch_size=64, experiment_experience_replay = False, experiment_target_network = False, target_update = 10, anneal=False, env = 'CartPole-v1'):

    env = gym.make(env)
    # env = gym.make('CartPole-v1', render_mode='human')
    
    input_dim = env.observation_space.shape[0]  
    output_dim = env.action_space.n  
    
    agent = DQNAgent(input_dim, output_dim, learning_rate, gamma, buffer_size, batch_size)
    
    Target_update = target_update
    num_episodes = n_episodes 
    steps_done = 0  
    returns_over_episode = []
    eval_returns = []
    eval_episodes = []
    eval_interval = 10
    
    print(experiment_experience_replay)
    print(experiment_target_network)
    print(args.use_experience_replay)
    print(args.use_target_network)
    
    for episode in range(num_episodes):
       
        reset_result = env.reset()
        raw_state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if isinstance(raw_state, tuple):
            raw_state = np.array(raw_state)
        state = torch.from_numpy(raw_state).float().unsqueeze(0)

        total_reward = 0 

        for iteration in range(500):
            action = agent.select_action(state, policy, epsilon, temp, anneal=anneal)
            action_int = action.item() if isinstance(action, torch.Tensor) else action

            step_result = env.step(action_int)
            next_state, reward, done, *extras = step_result

            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

            if  args.use_experience_replay or experiment_experience_replay :
                agent.memory.push(raw_state, action_int, reward, np.array(next_state), done)
                if len(agent.memory) > agent.batch_size:
                    agent.optimize_model(args, experiment_target_network)
            else:
                agent.direct_update(args, raw_state, action_int, reward, np.array(next_state), done, experiment_target_network)

            state = next_state_tensor
            raw_state = next_state
            total_reward += reward
            
            steps_done += 1    
            
            if done:
                break  

        if (experiment_target_network or args.use_target_network) and episode % Target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())    

        print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
        returns_over_episode.append(total_reward)
        
        eval_interation = episode + 1 
        if  (eval_interation)  % eval_interval == 0:
            mean_return = agent.evaluate(env)
            eval_returns.append(mean_return)
            eval_episodes.append(eval_interation)
            print("going for evaluation !!!!!!")
            print(np.array(eval_returns))
            print(np.array(eval_episodes))

    return np.array(eval_returns), np.array(eval_episodes)


def test():  
    buffer_size = 10000
    batch_size = 64   
    n_episodes = 1000
    gamma = 0.98
    learning_rate = 0.001
    parser.set_defaults(use_experience_replay=False, use_target_network=False)
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.01
    temp = 1.0

    plot = True

    DQN_Main(n_episodes, learning_rate, gamma, policy, epsilon, temp, plot, buffer_size, batch_size)

if __name__ == '__main__':
    test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from dqn import DQN_Main
from Helper import LearningCurvePlot, smooth

    
def average_over_repetitions(use_experience_replay, use_target_network, n_repetitions, n_episode, learning_rate, gamma, policy, epsilon, temp, smoothing_window, plot, neuron, architecture, batch_size=64, target_update = 1):
    returns_over_repetitions = []
    now = time.time()
    for repetition in range(n_repetitions):
        print(policy)
        print("Repetition Number: ", repetition + 1)
        returns, num_episodes = DQN_Main(
            n_episodes=n_episode,  
            learning_rate=learning_rate,
            gamma=gamma,
            policy=policy,
            epsilon=epsilon,
            temp=temp,
            plot=plot,
            buffer_size=10000,  
            batch_size=batch_size,  
            experiment_experience_replay=use_experience_replay,
            experiment_target_network=use_target_network,
            target_update = target_update,
            neurons = neuron,
            architecture = architecture
        )
        returns_over_repetitions.append(returns)
    
    # Average the learning curves across repetitions
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    avg_learning_curve = np.mean(returns_over_repetitions, axis=0)
    if smoothing_window is not None:
        avg_learning_curve = smooth(avg_learning_curve, smoothing_window)
    return avg_learning_curve , num_episodes  

def experiment():

    n_repetitions = 10
    n_episode = 500
    learning_rate = 0.001
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 1.0
    smoothing_window = 9
    plot = False
    
    #### Assignment 1: DQN neurons Variations
    use_experience_replay = True
    use_target_network = True
    policy = 'egreedy'
    epsilon = 0.01
    learning_rate = 0.001
    architectures = [1, 2]
    neurons = [32, 64, 128]
    Plot = LearningCurvePlot(title = 'DQN Architecture Analysis')    
    Plot.set_ylim(0,500) 
    for architecture in architectures:
        for neuron in neurons:        
            avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,n_repetitions, n_episode, learning_rate,gamma,policy,epsilon,temp,smoothing_window,plot, neuron, architecture)
            Plot.add_curve(num_episodes, avg_learning_curve, label=r'Hidden layer = {0}, Neurons = {1}'.format(architecture, neuron))
    Plot.save('ArchitectureAnalysis.png') 
    

    #### Assignment 2: DQN epsilon variation and softmax variation
    use_experience_replay = True
    use_target_network = True
    policy = 'egreedy'
    epsilons = [0.01,0.1,0.3]
    learning_rate = 0.001
    neuron = 128
    architecture = 2
    Plot = LearningCurvePlot(title = 'DQN Exploration: $\epsilon$-greedy versus softmax exploration')    
    Plot.set_ylim(0,500) 
    for epsilon in epsilons:        
        avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,n_repetitions, n_episode, learning_rate,gamma,policy,epsilon,temp,smoothing_window,plot, neuron, architecture)
        Plot.add_curve(num_episodes, avg_learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon)) 
    
    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    for temp in temps:
        avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,n_repetitions,n_episode, learning_rate,gamma,policy,epsilon,temp,smoothing_window,plot, neuron, architecture)
        Plot.add_curve(num_episodes, avg_learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
    Plot.save('ExplorationStrategies.png')
    
    #### Assignment 3: DQN discount factor variation
    use_experience_replay = True
    use_target_network = True
    gammas = [1.0, 0.99, 0.95, 0.90]
    policy = 'egreedy'
    epsilon = 0.01
    learning_rate = 0.001
    neuron = 128
    architecture = 2
    temp = 1.0
    Plot = LearningCurvePlot(title = 'DQN: Discount Factor ($\gamma$)') 
    Plot.set_ylim(0,500) 
    for gamma in gammas:
        avg_learning_curve, num_episodes= average_over_repetitions(use_experience_replay, use_target_network,n_repetitions,n_episode, learning_rate, gamma, policy,epsilon,temp,smoothing_window,plot, neuron, architecture)
        Plot.add_curve(num_episodes, avg_learning_curve,label=r'$ \gamma $ = {}'.format(gamma))
    Plot.save('DiscountFactorVariation.png')
    
    ### Assignment 4: DQN learning rate variation
    use_experience_replay = True
    use_target_network = True
    gamma =  0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 1.0
    neuron = 128
    architecture = 2
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    Plot = LearningCurvePlot(title = 'DQN: Learning Rate') 
    Plot.set_ylim(0,500) 
    for learning_rate in learning_rates:
        avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,n_repetitions,n_episode, learning_rate, gamma, policy,epsilon,temp,smoothing_window,plot, neuron, architecture)
        Plot.add_curve(num_episodes, avg_learning_curve,label=r' Learning Rate = {}'.format(learning_rate))
    Plot.save('LearningRateVariation.png')
    
    #### Assignment 5: DQN Batch Size
    use_experience_replay = True
    use_target_network = True
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 1.0
    learning_rate = 0.001
    neuron = 128
    architecture = 2
    batch_sizes = [16, 32, 64, 128]
    Plot = LearningCurvePlot(title = 'DQN: Batch Sizes') 
    Plot.set_ylim(0,500) 
    for batch_size in batch_sizes:
        avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,n_repetitions,n_episode, learning_rate, gamma, policy,epsilon,temp,smoothing_window,plot, neuron, architecture, batch_size)
        Plot.add_curve(num_episodes, avg_learning_curve,label=r'Batch Size = {}'.format(batch_size))
    Plot.save('BatchSizes.png')
    
    #### Assignment 6: DQN Target Network Updation
    use_experience_replay = True
    use_target_network = True
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 1.0
    learning_rate = 0.001
    target_updates = [1, 10, 50]
    neuron = 128
    architecture = 2
    batch_size = 64
    Plot = LearningCurvePlot(title = 'DQN: Target Network Updation') 
    Plot.set_ylim(0,500) 
    for target_update in target_updates:
        avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,n_repetitions,n_episode, learning_rate, gamma, policy, epsilon, temp, smoothing_window,plot, neuron, architecture, batch_size, target_update)
        Plot.add_curve(num_episodes, avg_learning_curve,label=r' Update TN after episodes = {}'.format(target_update))
    Plot.save('TargetNetworkUpdation.png')
    
    #### Assignment 7: DQN Variations
    learning_rate = 0.001
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 1.0
    neuron = 128
    architecture = 2

    Variations = [('DQN', True, True), ('DQN - ER', False, True), ('DQN - TN', True, False), ('DQN - ER - TN', False, False),]
    
    Plot = LearningCurvePlot(title='DQN Variations')    
    Plot.set_ylim(0, 500)
    
    for label, use_experience_replay, use_target_network in Variations:
        avg_learning_curve, num_episodes  = average_over_repetitions(use_experience_replay,use_target_network,n_repetitions, n_episode, learning_rate,gamma,policy,epsilon,temp,smoothing_window,plot, neuron, architecture)
        Plot.add_curve(num_episodes, avg_learning_curve, label=label)
    
    Plot.save('dqn_variations.png')
    
if __name__ == '__main__':
    experiment()
   

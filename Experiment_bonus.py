#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from dqn_bonus import DQN_Main
from Helper_bonus import LearningCurvePlot, smooth

    
def average_over_repetitions(use_experience_replay, use_target_network, n_repetitions, n_episode, learning_rate, gamma, policy, epsilon, temp, smoothing_window, plot, batch_size=64, target_update = 1, anneal=False, env='CartPole-v1'):
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
            anneal=anneal,
            env = env
        )
        returns_over_repetitions.append(returns)
    
    # Average the learning curves across repetitions
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    avg_learning_curve = np.mean(returns_over_repetitions, axis=0)
    if smoothing_window is not None:
        avg_learning_curve = smooth(avg_learning_curve, smoothing_window)
    return avg_learning_curve , num_episodes  


def experiment_anneal():
    n_repetitions = 10
    n_episode = 500
    learning_rate = 0.001
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 0.1
    smoothing_window = 9
    plot = False
    use_experience_replay = True
    use_target_network = True
    anneals = [True, False]

    learning_rate = 0.001
    gamma = 0.99
    policies = ['egreedy', 'softmax']
    epsilon = 0.01
    temp = 0.1

    Variations = [('DQN', True, True), ('DQN - ER', False, True), ('DQN - TN', True, False),
                  ('DQN - ER - TN', False, False), ]

    Plot = LearningCurvePlot(title='DQN Exploration: Annealing versus constant parameters')
    Plot.set_ylim(0, 600)

    #for label, use_experience_replay, use_target_network in Variations:
    for p in policies:
        for anneal in anneals:
            avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,
                                                                        n_repetitions, n_episode, learning_rate, gamma,
                                                                        p, epsilon, temp, smoothing_window, plot, anneal=anneal)

            if p == 'egreedy':
                if anneal == True:
                    Plot.add_curve(num_episodes, avg_learning_curve, label=f'$\epsilon$-greedy with anneal')
                else:
                    Plot.add_curve(num_episodes, avg_learning_curve, label=f'$\epsilon$-greedy, $\epsilon$ = {epsilon}')
            else:
                if anneal == True:
                    Plot.add_curve(num_episodes, avg_learning_curve, label=f'{p} with anneal')
                else:
                    Plot.add_curve(num_episodes, avg_learning_curve, label=r'softmax, $ \tau $ = {}'.format(temp))


    Plot.save('dqn_anneal_both.png')

def experiment_acrobot():
    n_repetitions = 20
    n_episode = 300
    learning_rate = 0.001
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.01
    temp = 0.1
    smoothing_window = 9
    plot = False
    use_experience_replay = True
    use_target_network = True
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    environments = ['CartPole-v1', 'Acrobot-v1']

    Plot = LearningCurvePlot(title='DQN for different environments')
    Plot.set_ylim(0, 600)

    for env in environments:

        avg_learning_curve, num_episodes = average_over_repetitions(use_experience_replay, use_target_network,
                                                                    n_repetitions, n_episode, learning_rate, gamma,
                                                                    policy, epsilon, temp, smoothing_window, plot, env=env)
        if env=='Acrobot-v1':
            avg_learning_curve += 500
        Plot.add_curve(num_episodes, avg_learning_curve, label=f'Environment: {env}')
    Plot.save('environments.png')
    
if __name__ == '__main__':
    experiment_anneal()
    experiment_acrobot()
   
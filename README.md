# Deep Q-Learning for Cartpole - Reinforcement Learning
## Overview

This repository focuses on using Deep Q-Learning (DQN) for solving the Cartpole problem from OpenAI's Gym environment. The repository also includes an ablation study comparing different variations of DQN and exploration strategies.

### Problem Description: Cartpole

The Cartpole environment involves balancing a pole on a moving cart. The objective is to keep the pole upright by applying forces to the cart to keep it balanced. The environment is included in OpenAI Gym.

### Key Features Implemented:

1. **Deep Q-Learning (DQN)**: A neural network is used to approximate the Q-value function, which predicts the expected reward for taking an action in a given state.
2. **Exploration Strategies**: The agent is trained using multiple exploration strategies, including:
   - ϵ-greedy
   - Boltzmann exploration
3. **Experience Replay**: A replay buffer is used to store experiences and replay them to stabilize training.
4. **Target Network**: A separate target network is periodically updated to make training more stable by reducing non-stationary updates.
5. **Ablation Study**: Ablation experiments comparing:
   - DQN without experience replay (`DQN-ER`)
   - DQN without the target network (`DQN-TN`)
   - DQN with neither (`DQN-ER-TN`)
6. **Bonus Experiments** (optional): Additional experiments such as implementing **Double DQN** and **Dueling DQN** for further exploration.

---

## File Structure

- **`dqn.py`**: Main implementation of the DQN agent.
- **`dqn_bonus.py`**: Implementation of bonus experiments such as Double DQN or Dueling DQN.
- **`Experiment.py`**: Script to run experiments and perform an ablation study on different DQN configurations.
- **`Experiment_bonus.py`**: Script for running additional experiments on bonus strategies (Double DQN, Dueling DQN).
- **`Helper.py`**: Utility functions for running experiments, managing the replay buffer, and processing training data.
- **`Helper_bonus.py`**: Additional helper functions for bonus experiments.

---

## How to Run the Code

### Prerequisites

1. Install Python 3.x.
2. Install the required libraries using the following command:
   ```bash
   pip install -r requirements.txt
3. Ensure that you have OpenAI Gym installed:
   ```bash
   pip install gym
   
### Running the DQN Agent
### To run the basic DQN implementation:

- python dqn.py

### To run the ablation study on the DQN agent:

- python Experiment.py --config ablation

### To run the bonus DQN experiments (e.g., Double DQN, Dueling DQN):


- python dqn_bonus.py

### To run the bonus experiment study:

- python Experiment_bonus.py --config bonus


---

### Exploration Strategies
ϵ-greedy: The agent explores the environment by taking a random action with probability ϵ and the best-known action otherwise.
Boltzmann Exploration: Actions are selected according to their estimated values' probabilities, controlled by a temperature parameter.

---

### Experience Replay
Experience replay stores past experiences (state, action, reward, next state) in a buffer. During training, random samples are drawn from this buffer to break correlation between consecutive experiences and stabilize learning.

---

### Target Network
A target network is used to make updates more stable by fixing the network used to generate target Q-values for some time while the main network is being trained.

---

### Ablation Study
The ablation study systematically removes one or more components of the DQN algorithm to understand their impact on learning performance. The study compares:

---

DQN: Standard Deep Q-Network with all features.
DQN-ER: DQN without experience replay.
DQN-TN: DQN without a target network.
DQN-ER-TN: DQN without both experience replay and target network.
Results and Conclusion
The DQN agent successfully learns to balance the cartpole in the OpenAI Gym environment.
The ablation study highlights the importance of experience replay and the target network in improving learning stability and performance.
Bonus experiments, including Double DQN and Dueling DQN, showed improvements in some cases, particularly in terms of stability.

---

### References
OpenAI Gym: https://gym.openai.com/
Original Deep Q-Learning Paper by DeepMind: https://arxiv.org/pdf/1312.5602.pdf
Exploration Strategies: https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html

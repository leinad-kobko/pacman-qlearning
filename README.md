# Running the code
Before running the code make sure you have a working version of Python installed.

To view the original learning agent from the original repository run:
```console
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

To view my improved learning agent run:
```console
python pacman.py -p ApproximateQAgent -a extractor=myExtractor -x 50 -n 60 -l mediumGrid
```

- x, represents the number of training episodes for Pacman
- n, represents the total number of episodes
- mediumGrid can be exchanged for another file in layouts (without the .lay extension)

# Project 3: Reinforcement Learning

This project implements model-based and model-free reinforcement learning algorithms.

1. **Value Iteration Agent**: It utilizes an MDP and runs value iteration for set iterations before the constructor returns. It implements both asynchronous & prioritized sweeping.

2. **Q-Learning**: A RL agent that learns by trial and error from interactions with the environment through its update(state, action, nextState, reward) method. Approximate Q-learning is also implemented

# Code added by Daniel Kobko

- Modified theFeatureExtractors.py to create better rewards for Pacman when he's learning.
- Created a new layout called mylayout.lay
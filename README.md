Repository File Descriptions

Required Packages
To run the files in this repository, you need the following packages:

numpy
matplotlib
torch
tensorboardX
tqdm
jsonargparse
pygame
random
tqdm
time

#environment.py
Defines the Environment class, which simulates the dynamics and interactions of multiple agents within a defined airspace.

#agent.py
Contains the Agent class, responsible for managing the state and actions of individual agents within the simulation environment.

#CDR_DQN.py
Implements the DQN (Deep Q-Network) class, a reinforcement learning algorithm for training agents based on their interactions within the environment.

#main.py
The main executable script that sets up the environment, initializes DQN parameters, and runs the training loop for the conflict resolution model.

#plot_training_data.py
Utility script for visualizing training data such as rewards and conflicts over episodes using matplotlib.

#run_trained_model.py
Script for loading a trained model and running simulations to evaluate its performance and render the simulation.

#sensitivity.py
Contains functions to perform sensitivity analysis on various parameters such as number of flights, environment size, and cruise speed.

#simulate.py
Script that provides functionalities to simulate flight dynamics and interactions in a controlled airspace environment.

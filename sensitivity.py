from simulate import Environment
from CDR_DQN import DQN
import numpy as np
from tensorboardX import SummaryWriter
import torch
from jsonargparse import ArgumentParser, ActionConfigFile
from tqdm import tqdm
import time
import matplotlib.pyplot as plt





# Hyper-parameters for DQN
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
EPSILON = 1  # Set to 0 for pure exploitation during evaluation
GAMMA = 0.9
TARGET_REPLACE_ITER = 500
MEMORY_CAPACITY = 20000
N_STATES = 3 + 4 * 4  # Number of neurons in the input layer (considering 4 neighbors)
N_HIDDEN = N_STATES * 2  # Number of neurons in the hidden layer
N_ACTION = 9  # Number of neurons in the output layer

#First simulation
EPISODE1 = 80000  
t1 = '08311834' 

#Second simulation
EPISODE2 = 99000  
t2 = '08201450'  

def run_trained_model(num_episodes, num_flights=10, env_size=30, cruise_speed=2, episode=80000, t='08311834'):

    # Environment Parameters
    cruise_speed = cruise_speed  # [m/s]
    max_speed = 4  # [m/s]
    min_speed = 1  # [m/s]
    min_distance = 1  # [m]
    size_airspace = env_size  # [m]
    timestep = 1  # [s]
    num_flights = num_flights
    dheading = 15 * np.pi / 180
    dspeed = 0.5


    # Initialize the DQN with the provided hyperparameters
    rl = DQN(BATCH_SIZE=BATCH_SIZE,
                LEARNING_RATE=LEARNING_RATE,
                EPSILON=EPSILON,
                GAMMA=GAMMA,
                TARGET_REPLACE_ITER=TARGET_REPLACE_ITER,
                MEMORY_CAPACITY=MEMORY_CAPACITY,
                N_HIDDEN=N_HIDDEN,
                N_ACTIONS=N_ACTION,
                N_STATES=N_STATES,
                ENV_A_SHAPE=0,
                EPISODE=episode,
                SHOW_ITER=1,
                TRAINED_NN=f'n_ac_10_episode_{episode}_{t}'
                )

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for evaluating a trained DQN policy',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=num_episodes)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # Parse arguments
    args = parser.parse_args()

    total_conflicts = 0
    total_not_done = 0

    # Execute episodes
    for episode in tqdm(range(args.episodes)):
        # Initialize environment
        env = Environment(num_flights,
                            size_airspace,
                            begin_speed=cruise_speed,
                            min_speed=min_speed,
                            max_speed=max_speed,
                            min_distance=min_distance,
                            max_heading_change=dheading,
                            max_speed_change=dspeed,
                            real_time = False,
                            timestep = timestep)

        # Initialize state, reward, action lists
        S = []
        R = []
        A = []

        # Initialize scenario and get the first observation
        obs = env.reset()
        S.append(obs)
        done = False

        # Initialize performance metrics
        number_conflict = 0
        not_done = 0

        # Execute simulation
        while not done:
            # Get actions
            action = []
            for agent in env.agents:
                if agent.done:
                    action.append(np.nan)
                else:
                    action.append(rl.choose_action(S[-1][agent.id], 0))  # Use pure exploitation

            # Update step
            rew, obs, done = env.step(action)
            S.append(obs)
            R.append(rew)
            A.append(action)

            # Count performance metrics
            for agent in env.agents:
                if not agent.done and agent.collision:
                    number_conflict += 1

        # Close environment after episode
        env.close()

        for agent in env.agents:
            if agent.failed:
                not_done += 1
        
        total_conflicts += number_conflict
        total_not_done += not_done

    return total_conflicts, total_not_done

def sensitivity_analysis(num_episodes, episode1, t1, episode2, t2):
    n_flights = range(5, 15)
    env_sizes = np.arange(25, 35)
    cruise_speeds = np.linspace(1, 4, 10)
    
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))  # Smaller figure size for compactness

    # Plot settings for both runs
    settings = [
        (episode1, t1, 'blue', 'R1'),
        (episode2, t2, 'orange', 'R2')
    ]

    # Plotting functions
    for i, (episode, t, color, label) in enumerate(settings):
        list_conflicts_flights, list_not_done_flights = [], []
        list_conflicts_env, list_not_done_env = [], []
        list_conflicts_speed, list_not_done_speed = [], []

        for n in n_flights:
            total_conflicts, total_not_done = run_trained_model(num_episodes, num_flights=n, episode=episode, t=t)
            list_conflicts_flights.append(total_conflicts / num_episodes)
            list_not_done_flights.append(total_not_done / num_episodes)

        for size in env_sizes:
            total_conflicts, total_not_done = run_trained_model(num_episodes, env_size=size, episode=episode, t=t)
            list_conflicts_env.append(total_conflicts / num_episodes)
            list_not_done_env.append(total_not_done / num_episodes)

        for speed in cruise_speeds:
            total_conflicts, total_not_done = run_trained_model(num_episodes, cruise_speed=speed, episode=episode, t=t)
            list_conflicts_speed.append(total_conflicts / num_episodes)
            list_not_done_speed.append(total_not_done / num_episodes)

        # Plots for number of flights
        axs[0, 0].plot(n_flights, list_conflicts_flights, marker='o', color=color, label=label)
        axs[0, 1].plot(n_flights, list_not_done_flights, marker='o', color=color, label=label)

        # Plots for environment size
        axs[1, 0].plot(env_sizes, list_conflicts_env, marker='o', color=color, label=label)
        axs[1, 1].plot(env_sizes, list_not_done_env, marker='o', color=color, label=label)

        # Plots for cruise speed
        axs[2, 0].plot(cruise_speeds, list_conflicts_speed, marker='o', color=color, label=label)
        axs[2, 1].plot(cruise_speeds, list_not_done_speed, marker='o', color=color, label=label)

    # Adding titles, labels, grids, and legends
    titles = ['Conflicts vs. Number of Flights', 'Not Done vs. Number of Flights',
              'Conflicts vs. Environment Size', 'Not Done vs. Environment Size',
              'Conflicts vs. Cruise Speed', 'Not Done vs. Cruise Speed']
    x_labels = ['Number of Flights', 'Number of Flights', 'Environment Size', 'Environment Size', 'Cruise Speed', 'Cruise Speed']
    y_labels = ['Mean Conflicts per Episode', 'Mean Failures per Episode'] * 3

    for ax, title, xlabel, ylabel in zip(axs.flat, titles, x_labels, y_labels):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True)
        ax.legend()

    plt.tight_layout(pad=1.0)
    plt.show()


total_conflicts, total_not_done = run_trained_model(1000, episode=EPISODE1, t=t1)
total_conflicts2, total_not_done2 = run_trained_model(1000, episode=EPISODE2, t=t2)

print(total_conflicts/1000, total_not_done/1000)
print(total_conflicts2/1000, total_not_done2/1000)


#sensitivity_analysis(100, EPISODE1, t1, EPISODE2, t2)  # Assuming 10 episodes for the sensitivity analysis.
import matplotlib.pyplot as plt
import numpy as np

def plot_states(state_matrix, agent_id, scale):
    # Visualize one of the metrics (e.g., speed) over time for the last episode
    i = agent_id  # Index of the agent to visualize
    speed_list = []
    heading_list = []
    relative_bearing_list = []
    distances_list = []
    relative_headings_list = []
    CPA_times_list = []
    CPA_distances_list = []

    for state in state_matrix:
        if not isinstance(state[i], float):
            agent_state = state[i]
            speed_list.append(agent_state[0] / scale)
            heading_list.append(agent_state[1])
            relative_bearing_list.append(agent_state[2])

            num_distances = len(agent_state[3:]) // 4
            distances_list.append(agent_state[3:3 + num_distances])
            relative_headings_list.append(agent_state[3 + num_distances:3 + 2 * num_distances])
            CPA_times_list.append(agent_state[3 + 2 * num_distances:3 + 3 * num_distances])
            CPA_distances_list.append(agent_state[3 + 3 * num_distances:3 + 4 * num_distances])


    t = range(len(speed_list))
    #t = range(10)

    # Plot all the states
    plt.figure(figsize=(15, 20))

    # Speed
    plt.subplot(7, 1, 1)
    plt.plot(t, speed_list, label='Speed')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)

    # Heading
    plt.subplot(7, 1, 2)
    plt.plot(t, heading_list, label='Heading')
    plt.xlabel('Time')
    plt.ylabel('Heading')
    plt.legend()
    plt.grid(True)

    # Relative Bearing
    plt.subplot(7, 1, 3)
    plt.plot(t, relative_bearing_list, label='Relative Bearing')
    plt.xlabel('Time')
    plt.ylabel('Relative Bearing')
    plt.legend()
    plt.grid(True)

    # Distances
    plt.subplot(7, 1, 4)
    plt.plot(t, np.array(distances_list), label='Distances')
    plt.xlabel('Time')
    plt.ylabel('Distances')
    plt.legend()
    plt.grid(True)

    # Relative Headings
    plt.subplot(7, 1, 5)
    plt.plot(t, np.array(relative_headings_list), label='Relative Headings')
    plt.xlabel('Time')
    plt.ylabel('Relative Headings')
    plt.legend()
    plt.grid(True)

    # CPA Times
    plt.subplot(7, 1, 6)
    plt.plot(t, np.array(CPA_times_list), label='CPA Times')
    plt.xlabel('Time')
    plt.ylabel('CPA Times')
    plt.legend()
    plt.grid(True)

    # CPA Distances
    plt.subplot(7, 1, 7)
    plt.plot(t, np.array(CPA_distances_list), label='CPA Distances')
    plt.xlabel('Time')
    plt.ylabel('CPA Distances')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from simulate import Environment
    from CDR_DQN import DQN
    import numpy as np
    from tensorboardX import SummaryWriter
    import torch
    from jsonargparse import ArgumentParser, ActionConfigFile
    from tqdm import tqdm
    import time

    # Hyper-parameters for DQN
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001
    EPSILON = 0  # Set to 0 for pure exploitation during evaluation
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 500
    MEMORY_CAPACITY = 20000
    N_STATES = 3 + 4 * 4  # Number of neurons in the input layer (considering 4 neighbors)
    N_HIDDEN = N_STATES * 2  # Number of neurons in the hidden layer
    N_ACTION = 9  # Number of neurons in the output layer

    #First simulation
    EPISODE = 80000  
    t = '08311834' 

    #Second simulation
    #EPISODE = 99000  
    #t = '08201450'  

    cruise_speed = 2 # [m/s]
    max_speed = 4 # [m/s]
    min_speed = 1 # [m/s]
    min_distance = 1 # [m]
    size_airspace = 30 # [m]
    timestep = 1  # [s]
    num_flights = 10
    dheading = 15*np.pi/180
    dspeed = 0.5
    size_drone = 0.5
    scale = 600/size_airspace

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
             EPISODE=EPISODE,
             SHOW_ITER=1,
             TRAINED_NN=f'n_ac_{num_flights}_episode_{EPISODE}_{t}'
             )

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for evaluating a trained DQN policy',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # Parse arguments
    args = parser.parse_args()

    # Initialize Tensorboard writer
    writer = SummaryWriter(comment='EuroInno_Eval')

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
                          real_time = True,
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

        # Calculate reward statistics
        num_rew = sum(1 for i in range(len(R)) for j in range(len(R[i])) if R[i][j] is not np.nan)
        total_rew = sum(R[i][j] for i in range(len(R)) for j in range(len(R[i])) if R[i][j] is not np.nan)

        # Log results to console
        print(f'Episode {episode + 1}/{args.episodes}')
        print(f'Number of conflicts: {number_conflict}')
        print(f'Total reward: {total_rew}')
        if num_rew > 0:
            print(f'Reward per action: {total_rew / num_rew} \n')

        # Log metrics to Tensorboard
        writer.add_scalar('Eval/Number_of_Conflicts', number_conflict, episode)
        writer.add_scalar('Eval/Total_Reward', total_rew, episode)
        if num_rew > 0:
            writer.add_scalar('Eval/Average_Reward_Per_Action', total_rew / num_rew, episode)

    # Close Tensorboard writer
    writer.close()



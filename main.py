if __name__ == "__main__":
    from simulate import Environment
    from CDR_DQN import DQN
    import numpy as np
    from tensorboardX import SummaryWriter
    import torch
    import time
    from jsonargparse import ArgumentParser, ActionConfigFile
    from tqdm import tqdm
    from run_trained_model import plot_states
    #import matplotlib.pyplot as plt

    #To start Tensorboard copy into terminal: tensorboard --logdir=runs
    
    #Airbus A320
    cruise_speed = 2 # [m/s]
    max_speed = 4 # [m/s]
    min_speed = 1# [m/s]
    min_distance = 1 # [m]
    size_airspace = 30 # [m]
    timestep = 1  # [s]
    num_flights = 10
    dheading = 15*np.pi/180
    dspeed = 0.5
    size_drone = 0.5
    scale = 600/size_airspace

    # Hyper-parameters for DQN
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001
    EPSILON = 0.8
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 500
    MEMORY_CAPACITY = 20000
    N_STATES = 3 + 4*(5-1) # look at 5 neighbours only
    N_HIDDEN = N_STATES*2  # number of neurons in the hidden layer
    N_ACTION = 9  # number of neurons in the output layer
    TRAINED_NN = None  # file name of the trained neural network, if provided
 
    # episode setting
    MAX_EPISODE = 100001
    SHOW_ITER = 100
    SAVE_ITER = 1000

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
             EPISODE=MAX_EPISODE,
             SHOW_ITER=SHOW_ITER,
             TRAINED_NN=TRAINED_NN
             )

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=MAX_EPISODE)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()
    # init environment
    env = Environment(num_flights, 
                      size_airspace, 
                      begin_speed = cruise_speed,
                      min_speed = min_speed,
                      max_speed = max_speed, 
                      min_distance= min_distance, 
                      max_heading_change= dheading, 
                      max_speed_change= dspeed,
                      timestep= timestep)

    # create file for tensorboardX
    writer = SummaryWriter(comment='EuroInno')

    # execute episode
    for episode in tqdm(range(args.episodes)):
        #print('\n')
        # initialize MDP set
        S = []  # state/ partial observation
        R = []  # reward
        A = []  # action

        # initialize the scenario and get the 1st observation
        obs = env.reset()
        S.append(obs)
        done = False

        # initialize performance metrics
        number_conflict = 0
        # execute simulation
        while not done:
            # get actions
            action = []
            for agent in env.agents:
                if agent.done:
                    action.append(np.nan)
                else:
                    action.append(rl.choose_action(S[-1][agent.id], episode))
   
            # update step
            rew, obs, done = env.step(action)
            S.append(obs)
            R.append(rew)
            A.append(action)

            # count performance metrics
            for agent in env.agents:
                if not agent.done:
                    if agent.collision:
                        number_conflict += 1

            # store transition
            for agent in env.agents:
                if not agent.done:
                    rl.store_transition(S[-2][agent.id], A[-1][agent.id], R[-1][agent.id], S[-1][agent.id])
        # close environment
        env.close()

        # reward statistics
        num_rew = 0
        total_rew = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] is not np.nan:
                    num_rew += 1
                    total_rew += R[i][j]
        
        # flight distance statistics
        total_optimal_trajectory_length = 0
        total_trajectory_length = 0

        # reinforcement learning
        if rl.memory_counter > MEMORY_CAPACITY:
            rl.learn()


        # record performance metrics for the episode by tensorboardX
        # View statistical charts: open '...\atcenv-main' in terminal, then input command 't ensorboard --logdir=runs'
        if episode % SHOW_ITER == 0:
            writer.add_scalar('2D_training/1_num_conflicts', number_conflict, episode)
            writer.add_scalar('2D_training/3_total_reward', total_rew, episode)
            writer.add_scalar('2D_training/4_average_reward_per_action', total_rew / num_rew, episode)

        # save neural network parameters
        if episode % SAVE_ITER == 0:
            t = time.strftime('%m%d%H%M', time.localtime())
            torch.save(rl.eval_net,
                       f'dump/n_ac_{num_flights}_episode_{episode}_{t}')
    writer.close()
    print(A)

from environment import Airspace
from agent import Agent
import pygame
import numpy as np
import random


class Environment(object):
    def __init__(self, 
                 n_agents = 40, 
                 size = 50, 
                 begin_speed=10,
                 min_speed = 5,
                 max_speed = 18, 
                 min_distance=10,
                 max_heading_change=(15*np.pi)/180,
                 max_speed_change=5,
                 size_ac=0.5,
                 timestep=1,
                 show_render=True,
                 real_time = False):
        
        self.done = False

        # Initialize environment
        self.scale = 600/size # Scale for visualizing the environment
        self.size = size*self.scale
        self.airspace = Airspace(self.size)

        # Initialize agents
        self.min_distance = min_distance
        self.timestep = timestep
        self.n_agents = n_agents
        self.num_moves = 5
        self.begin_speed = begin_speed*self.scale/self.num_moves
        self.min_speed = min_speed*self.scale/self.num_moves
        self.max_speed = max_speed*self.scale/self.num_moves
        self.max_heading_change = max_heading_change/self.num_moves
        self.max_speed_change = max_speed_change*self.scale/self.num_moves
        self.goal_accuracy = max_speed*self.scale/2
        self.size_ac = size_ac
        self.show_render = show_render
        self.real_time = real_time

        # Initialize ATC parameters
        self.min_distance = min_distance*self.scale

        #Initialize simulation parameters
        self.i = 0 # Step counter
        self.max_episode = round(1.5*np.sqrt(2*size**2)/(min_speed*timestep)) # Maximum number of episodes
        self.viewer = None
        self.clock = pygame.time.Clock()

        pygame.init()
        self.black = 0, 0, 0
        self.screen = pygame.display.set_mode(self.airspace.size)
        
    
    def random_border_position(self, size, start=None):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        while edge==start:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            return [random.uniform(0, size), 0], edge
        elif edge == 'bottom':
            return [random.uniform(0, size), size], edge
        elif edge == 'left':
            return [0, random.uniform(0, size)], edge
        else:  # right
            return [size, random.uniform(0, size)], edge
        

    def create_agents(self):
        agents = []
        for i in range(self.n_agents):
            start_loc, edge_start = self.random_border_position(self.airspace.width)
            while any(np.linalg.norm(np.array(start_loc) - np.array(agent.start_loc)) < self.min_distance*2 for agent in agents):
                start_loc, edge_start = self.random_border_position(self.airspace.width)
            goal_loc, _= self.random_border_position(self.airspace.width, start=edge_start)
            while np.linalg.norm(np.array(goal_loc) - np.array(start_loc)) < self.begin_speed*3:
                goal_loc, _ = self.random_border_position(self.airspace.width, start=edge_start)            
            agent = Agent(i, 
                          self.airspace, 
                          start_loc=start_loc, 
                          goal_loc=goal_loc, 
                          timestep=self.timestep, 
                          speed=self.begin_speed, 
                          min_speed=self.min_speed,
                          max_speed=self.max_speed, 
                          min_distance=self.min_distance,
                          max_heading_change=self.max_heading_change,
                          max_speed_change=self.max_speed_change,
                          goal_accuracy=self.goal_accuracy)
            agents.append(agent)
        return agents
    
    def action(self, actions):
        for agent in self.agents:
            action = actions[agent.id]
            if not agent.done:
                agent.perform_action(action)
        return None

    def reward(self):
        #rewards = [-agent.collision - abs(agent.relative_bearing) if not agent.done else np.nan for agent in self.agents]
        rewards = [-agent.collision*10 - abs(agent.relative_bearing) if not agent.done else np.nan for agent in self.agents]

        return rewards
   
    def set_neighbours(self):
        for agent in self.agents:
            neighbours = [other_agent for other_agent in self.agents if other_agent != agent]
            agent.neighbours = sorted(neighbours, key=lambda x: np.linalg.norm(x.start_loc - agent.start_loc))[:4]

    def state(self):
        states = []
        for agent in self.agents:
            if agent.done:
                states.append(np.nan)
            else:
                agent.calculate_state()
                state = agent.flat_state_matrix
                states.append(state)
        return states
    
    def update_collisions(self):
        for agent in self.agents:
            if not agent.done:
                agent.check_collision()
    
    def update_done(self):
        if all(agent.done for agent in self.agents):
            self.done = True
        else:
            for agent in self.agents:
                if not agent.done:
                    agent.goal_reached()
    
    def outside_environment(self):
        for agent in self.agents:
            if not agent.done:
                if agent.start_loc[0] + 0.2*self.size < 0:
                    agent.failed = True
                    agent.done = True
                elif agent.start_loc[0] - 0.2*self.size > self.size:
                    agent.failed = True
                    agent.done = True
                elif agent.start_loc[1] + 0.2*self.size < 0:
                    agent.failed = True
                    agent.done = True
                elif agent.start_loc[1] - 0.2*self.size > self.size:
                    agent.failed = True
                    agent.done = True

    def step(self, actions):     
        self.i += 1

        for i in range(self.num_moves):
            self.action(actions)
            self.update_positions()
            self.update_done()
            self.outside_environment()
            self.update_collisions()
            if self.show_render==True:
                self.draw_airplanes()
        #pygame.time.delay(100)  # Delay to control frame rate
        rewards = self.reward()
        states = self.state()
        done = self.done or self.i == self.max_episode 
        return rewards, states, done

    def reset(self):
        self.i = 0
        self.done = False
        self.agents = self.create_agents()
        self.set_neighbours()
        if self.show_render==True:
            self.draw_airplanes()
        return self.state()
    
    def update_positions(self):
        for agent in self.agents:
            #agent.act()  # Agent decides on an action
            agent.update_position()  # Update the agent's position

    def draw_airplanes(self):
        self.screen.fill(self.black)
        for agent in self.agents:
            if not agent.done:
                pygame.draw.line(self.screen, (255, 255, 0), agent.start_loc.astype(int), agent.goal_loc.astype(int), 1)
                
                color = (0, 255, 0)  # Default color is green

                # Check for collisions
                for other_agent in self.agents:
                    if other_agent != agent:
                        distance = np.linalg.norm(agent.start_loc - other_agent.start_loc)
                        if distance < self.min_distance:
                            color = (255, 0, 0)  # Change color to red if too close

                pygame.draw.circle(self.screen, color, agent.start_loc.astype(int), 0.5*self.size_ac*self.scale)
                pygame.draw.circle(self.screen, (255, 0, 0), agent.goal_loc.astype(int), 2)
        pygame.display.flip()


        # Ensure the frame lasts one second
        if self.real_time == True:
            self.clock.tick(self.num_moves*2)

    def close(self) -> None:

        """
        Closes the viewer
        :return:
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    



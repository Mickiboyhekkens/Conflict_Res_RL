import numpy as np
import random

class Agent(object):
    def __init__(self, 
                 agent_id, 
                 env, 
                 start_loc, 
                 goal_loc, 
                 timestep, 
                 speed=0.5,
                 min_speed=0.1, 
                 max_speed = 18,
                 min_distance=50,
                 max_heading_change=15*np.pi/180,
                 max_speed_change=5,
                 goal_accuracy=10):
        
        #Goal initialisation
        self.id = agent_id
        self.env = env
        self.timestep = timestep
        self.start_loc = np.array(start_loc)
        self.goal_loc = np.array(goal_loc)
        self.goal_accuracy = goal_accuracy
        self.heading = 0
        self.min_distance = min_distance
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_heading_change = max_heading_change 
        self.max_speed_change = max_speed_change


        # Agent state
        self.done = False
        self.collision = False
        self.failed = False

        # Neighbours
        self.neighbours = []

        #State matrix S
        self.speed = speed  # dim = 1
        self.heading = self.calculate_relative_bearing()  # dim = 1
        self.relative_bearing = self.calculate_relative_bearing() # dim = 1
        self.d_list = [] # dim = n
        self.heading_rel_list = [] # dim = n
        self.t_CPA_list = [] # dim = n
        self.d_CPA_list = [] # dim = n
        self.flat_state_matrix = np.zeros(3 + 4*len(self.neighbours)) # dim = 3 + 4n

    def calculate_relative_bearing(self):
        # Calculate the relative bearing to the goal
        direction_to_goal = self.goal_loc - self.start_loc
        angle_to_goal = np.arctan2(direction_to_goal[1], direction_to_goal[0])
        self.relative_bearing = angle_to_goal - self.heading
        # Normalize relative bearing to the range [-pi, pi]
        self.relative_bearing = (self.relative_bearing + np.pi) % (2 * np.pi) - np.pi
        return self.relative_bearing

    def perform_action(self, action):
        # Perform action based on action index
        if action == 0:
            self.heading_change(-1)
        elif action == 1:
            self.heading_change(1)
        elif action == 2:
            self.speed_change(-1)
        elif action == 3:
            self.speed_change(1)
        elif action == 4:
            pass
        elif action == 5:
            self.heading_change(-1)
            self.speed_change(-1)
        elif action == 6:
            self.heading_change(1)
            self.speed_change(-1)
        elif action == 7:
            self.heading_change(-1)
            self.speed_change(1)
        elif action == 8:
            self.heading_change(1)
            self.speed_change(1)

    def heading_change(self, delta_heading):
        # Change heading by a given amount
        self.heading += delta_heading*self.max_heading_change
    
    def speed_change(self, delta_speed):
        # Change speed by a given amount
        self.speed = np.clip(self.speed + delta_speed*self.max_speed_change, self.min_speed, self.max_speed)

    def update_position(self):
        # Update position based on current heading and speed
        self.start_loc += self.speed * np.array([np.cos(self.heading), np.sin(self.heading)]) * self.timestep
        self.calculate_relative_bearing()  # Update relative bearing after moving
        self.goal_reached()

    def calculate_state(self):
        v_own = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        
        # Initialize lists
        distances = []
        relative_headings = []
        CPA_distances = []
        CPA_times = []
        

        for agent in self.neighbours:

            # Calculate relative velocity and distance
            v_other = agent.speed * np.array([np.cos(agent.heading), np.sin(agent.heading)])
            v_rel = v_own - v_other
            d_rel = self.start_loc - agent.start_loc
            d = np.linalg.norm(d_rel)
            heading_diff = (self.heading - agent.heading + np.pi) % (2 * np.pi) - np.pi
            if np.dot(v_rel, v_rel) != 0:
                t_CPA = np.dot(-d_rel, v_rel) / np.dot(v_rel, v_rel)
            else:
                t_CPA = 1000000000000000
            d_CPA = np.sqrt(abs(np.dot(d_rel, d_rel) - t_CPA**2 * np.dot(v_rel, v_rel)))

            
            # Append to lists
            distances.append(d)
            relative_headings.append(heading_diff)
            CPA_distances.append(d_CPA)
            CPA_times.append(t_CPA)

        # Update state matrix
        self.d_list = distances
        self.heading_rel_list = relative_headings
        self.t_CPA_list = CPA_times
        self.d_CPA_list = CPA_distances
        
        self.flat_state_matrix = np.concatenate([np.array([self.speed]),
                                                    np.array([self.heading]),
                                                    np.array([self.relative_bearing]),
                                                    np.array(distances).flatten(),
                                                    np.array(relative_headings).flatten(),
                                                    np.array(CPA_times).flatten(),
                                                    np.array(CPA_distances).flatten()])
    
    def check_collision(self):
        # Check if the agent is too close to another agent
        self.collision = False
        collisions = []
        for agent in self.neighbours:
            if np.linalg.norm(self.start_loc - agent.start_loc) < self.min_distance:
                self.collision = True
                collisions.append(self.id)

    def goal_reached(self):
        # Check if the goal has been reached
        if np.linalg.norm(self.start_loc - self.goal_loc) < self.goal_accuracy:
            self.done = True
    

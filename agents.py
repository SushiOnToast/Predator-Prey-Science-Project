import random
import math
import pygame
from constants import *  # Assuming constants like ENERGY, PREDATOR_FOV, PREY_FOV, etc., are defined here
from neural_network import NeuralNetwork
import numpy as np

import math

class RayCaster:
    def __init__(self, agent, num_rays, fov_angle, max_range):
        """
        Initialize the RayCaster for the given agent.
        :param agent: The agent for which rays are cast.
        :param num_rays: Number of rays to cast.
        :param fov_angle: Field of view (in degrees).
        :param max_range: Maximum range of rays.
        """
        self.agent = agent
        self.num_rays = num_rays
        self.fov_angle = math.radians(fov_angle)  # Convert FOV to radians
        self.max_range = max_range

    def cast_rays(self, screen, other_agents):
        """
        Cast rays in the agent's field of view and return distances to the closest objects.
        :param screen: Pygame screen for boundary checks.
        :param other_agents: List of other agents in the environment.
        :return: List of distances to the closest object for each ray.
        """
        ray_distances = []
        step_angle = self.fov_angle / max(1, (self.num_rays - 1))

        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * step_angle
            ray_direction = self.agent.direction + angle_offset
            distance = self._cast_single_ray(ray_direction, screen, other_agents)
            ray_distances.append(distance)

            # Draw the ray for visualization
            # self._draw_ray(ray_direction, distance, screen)
        return ray_distances

    def _cast_single_ray(self, angle, screen, other_agents):
        """
        Cast a single ray in a specific direction.
        :param angle: Angle of the ray (in radians).
        :param screen: Pygame screen for boundary checks.
        :param other_agents: List of other agents in the environment.
        :return: Distance to the closest object or max range.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        for t in range(1, int(self.max_range + 1)):
            x = self.agent.x + t * dx
            y = self.agent.y + t * dy

            # Stop at screen borders
            if not (0 <= x < screen.get_width() and 0 <= y < screen.get_height()):
                return t  # Return distance to the boundary

            # Check for collisions with other agents, excluding itself
            for other_agent in other_agents:
                if other_agent != self.agent and other_agent.is_alive and self._check_collision(x, y, other_agent):
                    return t  # Return distance to the agent

        return self.max_range  # No intersection within range

    def _draw_ray(self, angle, distance, screen):
        """
        Draw a ray on the screen from the agent's position in the given direction.
        :param angle: The direction angle of the ray.
        :param distance: The distance the ray travels before hitting an object or max range.
        :param screen: Pygame screen to draw the ray.
        """
        dx = math.cos(angle) * distance
        dy = math.sin(angle) * distance

        # Draw the line from the agent's position to the calculated ray end position
        pygame.draw.line(screen, BLUE, (self.agent.x, self.agent.y), (self.agent.x + dx, self.agent.y + dy), 2)

    def _check_collision(self, x, y, other_agent):
        """
        Check if the point (x, y) collides with an agent.
        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param other_agent: Another agent in the environment.
        :return: True if the point collides with the agent, False otherwise.
        """
        return math.sqrt((x - other_agent.x)**2 + (y - other_agent.y)**2) < other_agent.size



class Agent:
    def __init__(self, x, y, type_, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        self.x = x
        self.y = y
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 1
        self.energy = ENERGY
        self.type = type_
        self.size = 10
        self.energy_depletion_rate = 0.05
        self.is_alive = True
        self.is_stationary = False
        self.is_recovering = False

        self.fov_angle = PREDATOR_FOV if self.type == "predator" else PREY_FOV
        self.num_rays = NUM_RAYS
        self.range = 200 if self.type == "predator" else 100
        self.digestion_cooldown = 0

        self.reproduction_threshold = 50 if type_ == "prey" else 10
        self.time_survived = 0
        self.prey_eaten = 0

        self.nn = NeuralNetwork(input_size, hidden_size, output_size, 2)
        self.ray_caster = RayCaster(self, self.num_rays, self.fov_angle, self.range)

        self.epsilon = 0.1
        self.gamma = 0.99
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0

    def select_action(self, state):
        q_values = self.nn.forward(state)

        # print(q_values)
        
        # Split the Q-values into two parts: one for angular velocity and one for speed
        angular_velocity_q_values = q_values[:len(q_values)//2]
        speed_q_values = q_values[len(q_values)//2:]

        # Select the best action from each part of Q-values
        angular_velocity_action = np.argmax(angular_velocity_q_values)
        speed_action = np.argmax(speed_q_values)
        
        return angular_velocity_action, speed_action


    def learn(self, current_state, reward):
        if self.previous_state is not None:
            # Get current Q-values and next Q-values from the neural network
            current_q_values = self.nn.forward(self.previous_state)
            next_q_values = self.nn.forward(current_state)

            # Calculate the target for angular velocity and speed
            angular_velocity_target = reward + self.gamma * np.max(next_q_values[:len(next_q_values)//2])
            speed_target = reward + self.gamma * np.max(next_q_values[len(next_q_values)//2:])
            
            # Calculate the temporal difference (TD) errors
            angular_velocity_td_error = angular_velocity_target - current_q_values[self.previous_action[0]]
            speed_td_error = speed_target - current_q_values[self.previous_action[1]]
            
            # Clip the TD errors to stay within bounds
            angular_velocity_td_error = np.clip(angular_velocity_td_error, -1.0, 1.0)
            speed_td_error = np.clip(speed_td_error, -1.0, 1.0)

            # Backpropagation to adjust the neural network weights
            self.nn.backward(self.previous_state, angular_velocity_td_error, self.previous_action[0])
            self.nn.backward(self.previous_state, speed_td_error, self.previous_action[1])

        # Update previous state, action, and reward for the next step
        self.previous_state = current_state
        self.previous_action = self.select_action(current_state)
        self.previous_reward = reward

    def move(self, screen, other_agents):
        if self.is_alive:
            self.time_survived += 1
            raw_state = np.array(self.ray_caster.cast_rays(screen, other_agents))
            state = (raw_state - np.mean(raw_state)) / (np.std(raw_state) + 1e-8)  # Normalize state

            # Epsilon-greedy strategy: choose random action with probability epsilon
            if random.uniform(0, 1) < self.epsilon:
                delta_angular_velocity, delta_speed = np.random.uniform(-1, 1, 2)
            else:
                delta_angular_velocity, delta_speed = self.select_action(state)

            # Move the agent if it is not recovering
            if self.energy > 0 and not self.is_recovering:
                self.direction += delta_angular_velocity / 2
                self.speed = self.speed + delta_speed / 5

                # Update position based on speed and direction
                self.x += self.speed * math.cos(self.direction)
                self.y += self.speed * math.sin(self.direction)

                # Ensure the agent stays within screen bounds
                self.x = np.clip(self.x, 0, screen.get_width() - self.size)
                self.y = np.clip(self.y, 0, screen.get_height() - self.size)

                # Decrease energy and check if agent dies due to energy depletion
                self.energy -= self.energy_depletion_rate
                if self.energy <= 0:
                    if self.type == "predator":
                        self.is_alive = False
                    else:
                        self.energy = 0
                        self.is_recovering = True
                        self.is_stationary = True
            elif self.is_recovering:
                self.manage_recovery()

            if self.digestion_cooldown > 0:
                self.digestion_cooldown -= 1

            # Predator behavior: attempt to eat prey
            if self.type == "predator":
                for agent in other_agents:
                    if agent.type == "prey" and agent.is_alive:
                        distance = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                        if distance <= self.size + agent.size:
                            self.eat_prey(agent)
                            break

            # Calculate and clip the reward
            reward = self.calculate_reward()
            reward = np.clip(reward, -5, 5)  # Clip the reward range to avoid large updates

            # Learn from the experience
            self.learn(state, reward)


    def reproduce(self):
        if self.type == "prey" and self.time_survived >= PREY_MIN_SURVIVAL_TIME and self.energy >= self.reproduction_threshold:
            offset_x, offset_y = random.uniform(-20, 20), random.uniform(-20, 20)
            offspring = Agent(self.x + offset_x, self.y + offset_y, self.type)
            self.energy /= 2
            self.time_survived = 0
            return offspring
        elif self.type == "predator" and self.prey_eaten >= PREDATOR_PREY_EATEN_THRESHOLD:
            offset_x, offset_y = random.uniform(-20, 20), random.uniform(-20, 20)
            offspring = Agent(self.x + offset_x, self.y + offset_y, self.type)
            self.prey_eaten = 0
            return offspring
        return None

    def manage_recovery(self):
        if self.type == "prey" and self.is_recovering:
            self.energy = min(ENERGY, self.energy + 0.1)
            if self.energy >= ENERGY:
                self.is_recovering = False
                self.is_stationary = False

    def eat_prey(self, prey):
        if self.digestion_cooldown == 0:
            prey.is_alive = False
            self.energy += 20
            self.digestion_cooldown = DIGESTION_COOLDOWN_TIME
            self.prey_eaten += 1

    def calculate_reward(self):
        if self.type == "predator" and self.energy > ENERGY:
            return 10
        elif self.type == "prey" and self.is_alive:
            return 1 + (self.time_survived * 0.1)
        return -10

    def draw(self, screen):
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

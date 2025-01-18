import random
import math
import pygame
from constants import *  # Assuming constants like ENERGY, PREDATOR_FOV, PREY_FOV, etc., are defined here
from neural_network import NeuralNetwork
import numpy as np

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

    def cast_rays(self, screen, all_agents):
        """
        Cast rays in the agent's field of view and return distances to the closest objects.
        :param screen: Pygame screen for boundary checks.
        :param all_agents: List of all agents in the environment.
        :return: List of distances to the closest object for each ray.
        """
        # Exclude self.agent from other agents for collision checks
        other_agents = [agent for agent in all_agents if agent != self.agent and agent.is_alive]

        ray_distances = []
        step_angle = self.fov_angle / max(1, (self.num_rays - 1))

        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * step_angle
            ray_direction = self.agent.direction + angle_offset
            distance = self._cast_single_ray(ray_direction, screen, other_agents)
            ray_distances.append(distance)
        return ray_distances

    def _cast_single_ray(self, angle, screen, nearby_agents):
        dx = math.cos(angle)
        dy = math.sin(angle)

        for t in range(1, int(self.max_range + 1)):
            x = self.agent.x + t * dx
            y = self.agent.y + t * dy

            # Stop at screen borders
            if not (0 <= x < screen.get_width() and 0 <= y < screen.get_height()):
                return t  # Return distance to the boundary

            # Create the ray's rectangle at (x, y) with a small width for detection purposes
            ray_rect = pygame.Rect(x, y, 2, 2)  # Small rectangle to represent the ray

            for other_agent in nearby_agents:
                if other_agent != self.agent and other_agent.is_alive:
                    # Check for collision using colliderect
                    agent_rect = pygame.Rect(other_agent.x - other_agent.size, 
                                            other_agent.y - other_agent.size, 
                                            other_agent.size * 2, 
                                            other_agent.size * 2)

                    if ray_rect.colliderect(agent_rect):
                        return t  # Return distance when collision occurs

        return self.max_range  # No intersection within range

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

    def move(self, screen, other_agents):
        if self.is_alive:
            self.time_survived += 1
            state = np.array(self.ray_caster.cast_rays(screen, other_agents))/self.range
            

            # No reinforcement learning or Q-learning here, just neural network outputs
            output = self.nn.forward(state)
            print(output)
            delta_angular_velocity, delta_speed = output[0], output[1]

            if self.energy > 0 and not self.is_recovering:
                self.direction += delta_angular_velocity / 50
                self.speed = self.speed + delta_speed / 50

                self.x += self.speed * math.cos(self.direction)
                self.y += self.speed * math.sin(self.direction)

                self.x = np.clip(self.x, 0, screen.get_width() - self.size)
                self.y = np.clip(self.y, 0, screen.get_height() - self.size)

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

            if self.type == "predator":
                for agent in other_agents:
                    if agent.type == "prey" and agent.is_alive:
                        distance = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                        if distance <= self.size + agent.size:
                            self.eat_prey(agent)
                            break

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
            predator_rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)
            prey_rect = pygame.Rect(prey.x - prey.size, prey.y - prey.size, prey.size * 2, prey.size * 2)

            if predator_rect.colliderect(prey_rect):
                prey.is_alive = False
                self.energy += 20
                self.digestion_cooldown = DIGESTION_COOLDOWN_TIME
                self.prey_eaten += 1

    def calculate_reward(self, nearby_agents):
        reward = 0
        
        if self.type == "predator":
            # Reward for catching prey
            reward += 10 * self.prey_eaten
            
            # Small reward for energy efficiency
            if self.energy > 50:
                reward += 0.5
            
            # Penalty for idling or running out of energy
            if self.energy <= 0:
                reward -= 10
            elif self.speed < 0.1:
                reward -= 0.1
            
            # Exploration reward
            for agent in nearby_agents:
                if agent.type == "prey":
                    distance = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                    if distance <= self.range:
                        reward += 0.1

        elif self.type == "prey":
            # Reward for surviving
            reward += 1

            # Bonus for avoiding predators
            for agent in nearby_agents:
                if agent.type == "predator":
                    distance = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                    if distance <= self.size + agent.size + 20:
                        reward -= 5  # Close call
                    else:
                        reward += 0.1  # Safe distance

            # Energy recovery bonus
            if self.is_stationary and not self.is_recovering:
                reward += 0.2
            
            # Penalty for getting caught
            if not self.is_alive:
                reward -= 10
        
        return reward


    def draw(self, screen):
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

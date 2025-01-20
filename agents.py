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
    def __init__(self, x, y, type_, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, nn=None):
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

        self.nn = nn if nn is not None else NeuralNetwork(input_size, hidden_size, output_size)
        self.ray_caster = RayCaster(self, self.num_rays, self.fov_angle, self.range)

        self.fitness = 0

    def move(self, screen, other_agents):
        if self.is_alive:
            self.time_survived += 1
            state = np.array(self.ray_caster.cast_rays(screen, other_agents))/self.range

            # Neural network decision-making
            output = self.nn.forward(state)
            delta_angular_velocity, delta_speed = output[0], output[1]

            if self.energy > 0 and not self.is_recovering:
                self.direction += delta_angular_velocity / 50
                self.speed = self.speed + delta_speed / 50

                self.x += self.speed * math.cos(self.direction)
                self.y += self.speed * math.sin(self.direction)

                self.x = max(0, min(self.x, screen.get_width() - self.size))
                self.y = max(0, min(self.y, screen.get_height() - self.size))

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

    def eat_prey(self, prey):
        """Handle the predation and energy increase for predators."""
        if prey.is_alive:
            prey.is_alive = False
            self.energy += prey.energy  # Predator gains energy from the prey
            self.prey_eaten += 1
            self.digestion_cooldown = 10  # Predator enters digestion cooldown

    def reproduce(self, partner=None):
        """Handle the reproduction process, based on fitness values."""
        offspring = None

        if self.type == "prey":
            # Prey reproduce if their fitness value reaches a certain threshold
            if self.fitness >= PREY_FITNESS_THRESHOLD:
                offspring = self._reproduce_with_crossover(partner)
        
        elif self.type == "predator":
            # Predators reproduce based on their fitness, which increases with prey eaten
            if self.fitness >= PREDATOR_FITNESS_THRESHOLD:
                offspring = self._reproduce_with_crossover(partner)

        if offspring:
            self.energy /= 2  # Split energy between parent and offspring
            self.time_survived = 0  # Reset survival time after reproduction
        return offspring

    def _reproduce_with_crossover(self, partner):
        """Handle crossover between two agents and produce an offspring."""
        if partner is None:
            return None  # No partner, no reproduction

        # Perform crossover on the neural networks
        offspring_nn = self.nn.crossover(partner.nn)
        
        # Optionally, mutate the offspring neural network
        offspring_nn.mutate()

        # Create the offspring agent
        offset_x, offset_y = random.uniform(-20, 20), random.uniform(-20, 20)
        offspring = Agent(self.x + offset_x, self.y + offset_y, self.type)
        offspring.nn = offspring_nn  # Assign the new neural network to the offspring

        return offspring

    def manage_recovery(self):
        if self.type == "prey" and self.is_recovering:
            self.energy += 1  # Regenerate energy for prey when recovering
            if self.energy >= ENERGY:
                self.is_recovering = False
                self.is_stationary = False
                self.energy = ENERGY  # Maximum energy

    def update_fitness(self):
        """Update the fitness of the agent based on its actions."""
        if self.type == "predator":
            # Predator's fitness increases with each prey eaten
            self.fitness += self.prey_eaten
        elif self.type == "prey":
            # Prey's fitness increases with survival time and energy recovery
            self.fitness += self.time_survived * 0.1  # Adjust multiplier as needed
            if self.is_recovering:
                self.fitness += 0.5  # Reward prey for energy recovery

    def draw(self, screen):
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)
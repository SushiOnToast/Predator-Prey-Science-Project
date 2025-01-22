import random
import math
import pygame
from constants import *  # Assuming constants like ENERGY, PREDATOR_FOV, PREY_FOV, etc., are defined here
import numpy as np
import neat

class RayCaster:
    def __init__(self, agent, num_rays, fov_angle, max_range):
        self.agent = agent
        self.num_rays = num_rays
        self.fov_angle = math.radians(fov_angle)  # Convert FOV to radians
        self.max_range = max_range

    def cast_rays(self, screen, all_agents):
        other_agents = [agent for agent in all_agents if agent != self.agent and agent.is_alive]
        ray_data = []

        step_angle = self.fov_angle / max(1, (self.num_rays - 1))

        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * step_angle
            ray_direction = self.agent.direction + angle_offset
            distance, agent_type = self._cast_single_ray(ray_direction, screen, other_agents)
            ray_data.append((distance, agent_type))

        return ray_data

    def _cast_single_ray(self, angle, screen, nearby_agents):
        dx = math.cos(angle)
        dy = math.sin(angle)

        for t in range(1, int(self.max_range + 1)):
            x = self.agent.x + t * dx
            y = self.agent.y + t * dy

            if not (0 <= x < screen.get_width() and 0 <= y < screen.get_height()):
                return t, None

            ray_rect = pygame.Rect(x, y, 2, 2)

            for other_agent in nearby_agents:
                if other_agent != self.agent and other_agent.is_alive:
                    agent_rect = pygame.Rect(other_agent.x - other_agent.size, 
                                            other_agent.y - other_agent.size, 
                                            other_agent.size * 2, 
                                            other_agent.size * 2)

                    if ray_rect.colliderect(agent_rect):
                        return t, other_agent.type  

        return self.max_range, None


class Agent:
    def __init__(self, x, y, type_, nn=None, genome_id=None):
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

        self.nn = nn
        self.ray_caster = RayCaster(self, self.num_rays, self.fov_angle, self.range)

        self.fitness = 0
        self.genome_id = genome_id

        self.distance_traveled = 0  # Track the distance the agent has traveled

    def move(self, screen, other_agents):
        if self.is_alive:
            self.time_survived += 1

            # Track previous position to calculate distance traveled
            previous_x, previous_y = self.x, self.y

            # Get ray distances and their corresponding agent types
            ray_data = self.ray_caster.cast_rays(screen, other_agents)

            # Prepare the input array for the neural network
            state = []
            for distance, agent_type in ray_data:
                # Normalize distance to range [0, 1]
                state.append(distance / self.range)
                # Encode agent type (0 = none, 1 = predator, 2 = prey)
                state.append(0 if agent_type is None else (1 if agent_type == 'predator' else 2))

            # Add the agent's energy level as an input (normalized to [0, 1])
            state.append(self.energy / ENERGY)

            # Calculate a reward/penalty signal
            reward_signal = 0
            if self.type == "predator":
                # Reward predators for eating prey, penalize for not eating
                reward_signal = 1 if self.prey_eaten > 0 else -0.1
            elif self.type == "prey":
                # Reward prey for recovering energy or surviving, penalize for low energy
                if self.is_recovering:
                    reward_signal = 1
                else:
                    reward_signal = -0.1 if self.energy < ENERGY * 0.2 else 0.5

            # Add reward/penalty signal to the state
            state.append(reward_signal)

            # Neural network decision-making
            output = self.nn.activate(np.array(state))
            delta_angular_velocity, delta_speed = output[0], output[1]

            # Apply scaling using tanh activation
            delta_angular_velocity = math.tanh(delta_angular_velocity) * MAX_ANGULAR_VELOCITY
            delta_speed = math.tanh(delta_speed) * MAX_ACCELERATION

            # Apply small random noise to prevent circling and encourage exploration
            delta_angular_velocity += random.uniform(-0.1, 0.1)
            delta_speed += random.uniform(-0.1, 0.1)

            # Apply movement limits and update direction/speed
            self.direction += delta_angular_velocity
            self.speed += delta_speed

            max_speed = 3.0  # Maximum allowed speed
            self.speed = max(0, min(self.speed, max_speed))

            # Update position
            self.x += self.speed * math.cos(self.direction)
            self.y += self.speed * math.sin(self.direction)

            # Clamp position to screen boundaries
            self.x = max(0, min(self.x, screen.get_width() - self.size))
            self.y = max(0, min(self.y, screen.get_height() - self.size))

            # Calculate distance traveled in this step
            step_distance = math.sqrt((self.x - previous_x) ** 2 + (self.y - previous_y) ** 2)
            self.distance_traveled += step_distance

            # Decrease energy over time
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

            # Handle digestion cooldown for predators
            if self.digestion_cooldown > 0:
                self.digestion_cooldown -= 1

            # Predator-prey interaction
            if self.type == "predator":
                for agent in other_agents:
                    if agent.type == "prey" and agent.is_alive:
                        distance = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                        if distance <= self.size + agent.size:
                            self.eat_prey(agent)
                            break

            # # Check for immediate reproduction condition
            # if self.energy >= self.reproduction_threshold:
            #     self.reproduce_immediate(other_agents)

    def eat_prey(self, prey):
        """Handle the predation and energy increase for predators."""
        if prey.is_alive:
            prey.is_alive = False
            self.energy += prey.energy  # Predator gains energy from the prey
            self.prey_eaten += 1
            self.digestion_cooldown = 10  # Predator enters digestion cooldown

    def reproduce_immediate(self, other_agents):
        """Immediately reproduce if the agent has enough energy."""
        partner = self.find_reproduction_partner(other_agents)
        if partner:
            offspring = self.reproduce(partner)
            # Position the offspring close to one of the parents
            offspring.x = self.x + random.uniform(-20, 20)
            offspring.y = self.y + random.uniform(-20, 20)
            offspring.energy = ENERGY // 2  # Start with some initial energy
            return offspring
        return None

    def find_reproduction_partner(self, other_agents):
        """Find another agent of the same type with enough energy to reproduce."""
        for agent in other_agents:
            if agent != self and agent.is_alive and agent.type == self.type and agent.energy >= self.reproduction_threshold:
                return agent
        return None

    def reproduce(self, partner):
        """Handle the reproduction process using NEAT's built-in crossover."""
        if partner is None:
            return None  # No partner, no reproduction

        # Perform crossover using NEAT
        offspring_nn = self.nn.crossover(partner.nn)
        offspring_nn.mutate()

        offspring = Agent(self.x, self.y, self.type)
        offspring.nn = offspring_nn  # Assign the new neural network to the offspring
        return offspring

    def manage_recovery(self):
        if self.type == "prey" and self.is_recovering:
            self.energy += 1  # Regenerate energy for prey when recovering
            if self.energy >= ENERGY:
                self.is_recovering = False
                self.is_stationary = False
                self.energy = ENERGY  # Maximum energy

    def update_fitness(self, other_agents):
        if self.type == "predator":
            self.fitness += self.prey_eaten
            self.fitness -= self.distance_traveled * 0.005  # Penalize excessive movement
        elif self.type == "prey":
            self.fitness += self.time_survived * 0.1
            if self.is_recovering:
                self.fitness += 0.5
            self.fitness += self.distance_traveled * 0.01

            # Reward evasion
            closest_predator_distance = min(
                math.sqrt((self.x - predator.x) ** 2 + (self.y - predator.y) ** 2)
                for predator in other_agents if predator.type == "predator" and predator.is_alive
            )
            self.fitness += closest_predator_distance * 0.001

    def draw(self, screen):
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

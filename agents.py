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
        """
        Cast rays in the agent's field of view and return distances to the closest objects and their types.
        :param screen: Pygame screen for boundary checks.
        :param all_agents: List of all agents in the environment.
        :return: List of tuples (distance, type) for each ray.
        """
        other_agents = [agent for agent in all_agents if agent != self.agent and agent.is_alive]
        ray_data = []

        step_angle = self.fov_angle / max(1, (self.num_rays - 1))

        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * step_angle
            ray_direction = self.agent.direction + angle_offset
            distance, agent_type = self._cast_single_ray(ray_direction, screen, other_agents)
            ray_data.append((distance, agent_type))
            # self._draw_ray(ray_direction, distance, screen)

        return ray_data

    def _cast_single_ray(self, angle, screen, nearby_agents):
        dx = math.cos(angle)
        dy = math.sin(angle)

        for t in range(1, int(self.max_range + 1)):
            x = self.agent.x + t * dx
            y = self.agent.y + t * dy

            # Stop at screen borders
            if not (0 <= x < screen.get_width() and 0 <= y < screen.get_height()):
                return t, None  # Return distance and no agent type

            ray_rect = pygame.Rect(x, y, 2, 2)  # Small rectangle to represent the ray

            for other_agent in nearby_agents:
                if other_agent != self.agent and other_agent.is_alive:
                    agent_rect = pygame.Rect(other_agent.x - other_agent.size, 
                                            other_agent.y - other_agent.size, 
                                            other_agent.size * 2, 
                                            other_agent.size * 2)

                    if ray_rect.colliderect(agent_rect):
                        return t, other_agent.type  # Return distance and the type of the agent

        return self.max_range, None  # No intersection within range
    
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

        # Additional metrics
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
                state.append(distance / self.range)  # Normalize distance to range [0, 1]
                state.append(0 if agent_type is None else (1 if agent_type == 'predator' else 2))  # Encode agent type

            # Neural network decision-making
            output = self.nn.activate(np.array(state))
            delta_angular_velocity, delta_speed = output[0], output[1]

            # Apply scaling using tanh activation
            delta_angular_velocity = math.tanh(delta_angular_velocity) * MAX_ANGULAR_VELOCITY
            delta_speed = math.tanh(delta_speed) * MAX_ACCELERATION

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


    def eat_prey(self, prey):
        """Handle the predation and energy increase for predators."""
        if prey.is_alive:
            prey.is_alive = False
            self.energy += prey.energy  # Predator gains energy from the prey
            self.prey_eaten += 1
            self.digestion_cooldown = 10  # Predator enters digestion cooldown

    def reproduce(self, partner=None):
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
        # Body color
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

        # Eye parameters
        eye_radius = self.size // 4  # Size of the eyes
        pupil_radius = eye_radius // 2  # Size of the pupils
        eye_offset = self.size // 2  # Distance of the eyes from the center
        pupil_offset = eye_radius // 2  # Offset for the pupils based on direction

        # Calculate eye positions relative to the agent's direction
        eye_angle = math.pi / 4  # Angle offset for the eyes
        left_eye_x = self.x + eye_offset * math.cos(self.direction - eye_angle)
        left_eye_y = self.y + eye_offset * math.sin(self.direction - eye_angle)
        right_eye_x = self.x + eye_offset * math.cos(self.direction + eye_angle)
        right_eye_y = self.y + eye_offset * math.sin(self.direction + eye_angle)

        # Draw eyes (white sclera)
        pygame.draw.circle(screen, WHITE, (int(left_eye_x), int(left_eye_y)), eye_radius)
        pygame.draw.circle(screen, WHITE, (int(right_eye_x), int(right_eye_y)), eye_radius)

        # Calculate pupil positions based on direction
        left_pupil_x = left_eye_x + pupil_offset * math.cos(self.direction)
        left_pupil_y = left_eye_y + pupil_offset * math.sin(self.direction)
        right_pupil_x = right_eye_x + pupil_offset * math.cos(self.direction)
        right_pupil_y = right_eye_y + pupil_offset * math.sin(self.direction)

        # Draw pupils (black)
        pygame.draw.circle(screen, BLACK, (int(left_pupil_x), int(left_pupil_y)), pupil_radius)
        pygame.draw.circle(screen, BLACK, (int(right_pupil_x), int(right_pupil_y)), pupil_radius)


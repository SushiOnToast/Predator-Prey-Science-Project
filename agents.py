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

            # Check for collisions with other agents
            for other_agent in other_agents:
                if other_agent.is_alive and self._check_collision(x, y, other_agent):
                    return t  # Return distance to the agent

        return self.max_range  # No intersection within range

    def _check_collision(self, x, y, other_agent):
        """
        Check if the point (x, y) collides with an agent.
        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param other_agent: Another agent in the environment.
        :return: True if the point collides with the agent, False otherwise.
        """
        return math.sqrt((x - other_agent.x)**2 + (y - other_agent.y)**2) < other_agent.size


# Define Agent Class
class Agent:
    def __init__(self, x, y, type_, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        self.x = x
        self.y = y
        self.direction = random.uniform(0, 2 * math.pi)  # Random initial direction
        self.speed = 2  # Constant speed
        self.energy = ENERGY  # Starting energy
        self.type = type_  # Predator or Prey
        self.size = 10
        self.energy_depletion_rate = 0.1
        self.is_alive = True
        self.is_stationary = False  # Track if prey is stationary to regain energy
        self.is_recovering = False  # Track if prey is recovering energy

        self.fov_angle = PREDATOR_FOV if self.type == "predator" else PREY_FOV
        self.num_rays = NUM_RAYS
        self.range = 200 if self.type == "predator" else 50
        self.digestion_cooldown = 0 # digestion cooldown for predators
        
        self.reproduction_threshold = 50 if type_ == "prey" else 10  # Thresholds for reproduction
        self.time_survived = 0  # Track survival time for prey
        self.prey_eaten = 0  # Track number of prey eaten by predators

        # Neural Network for decision making
        self.nn = NeuralNetwork(input_size, hidden_size, output_size)

        self.ray_caster = RayCaster(self, self.num_rays, self.fov_angle, self.range)

    def move(self, screen, other_agents):
        """Move the agent based on its direction, speed, and neural network output."""
        if self.is_alive:
            self.time_survived += 1

            if self.energy > 0 and not self.is_recovering:  # Only move if not recovering
                # Get input from the environment (distances from rays)
                ray_inputs = self.ray_caster.cast_rays(screen, other_agents)
                ray_inputs = np.array(ray_inputs).flatten()  # Flatten the list into a single array for input

                # Perform forward pass through the neural network
                nn_output = self.nn.forward(ray_inputs)

                # Extract the output values (direction and speed)
                # Assuming the network's output has two values: speed and direction change
                self.speed = np.clip(nn_output[0] * 10, 0, 10)  # Speed is between 0 and 10
                self.direction += nn_output[1] * math.pi / 10  # Direction change is scaled by pi

                # Move the agent based on the neural network output
                self.x += self.speed * math.cos(self.direction)
                self.y += self.speed * math.sin(self.direction)

                # Adjust position if agent hits the screen border
                if self.x - self.size < 0:
                    self.x = self.size
                elif self.x + self.size > screen.get_width():
                    self.x = screen.get_width() - self.size
                if self.y - self.size < 0:
                    self.y = self.size
                elif self.y + self.size > screen.get_height():
                    self.y = screen.get_height() - self.size

                # Deplete energy over time while moving
                self.energy -= self.energy_depletion_rate

                # If energy runs out and agent is a predator, it dies
                if self.energy <= 0:
                    if self.type == "predator":
                        self.is_alive = False
                    else:
                        self.energy = 0  # Ensure energy is zero when it starts recovery
                        self.is_recovering = True  # Start recovery if prey is out of energy
                        self.is_stationary = True  # Prey becomes stationary to regain energy

            elif self.is_recovering:  # If recovering, manage recovery process
                self.manage_recovery()

            # Digestion cooldown logic
            if self.digestion_cooldown > 0:
                self.digestion_cooldown -= 1

            # Predator eating logic using FOV and ray-casting
            if self.type == "predator":
                for agent in other_agents:
                    if agent.type == "prey" and agent.is_alive:
                        distance = math.sqrt((self.x - agent.x)**2 + (self.y - agent.y)**2)
                        if distance <= self.size + agent.size:
                            self.eat_prey(agent)
                            break  # Stop checking this agent once prey is eaten
            
            reward = self.calculate_reward()
            self.adjust_weights(reward)
    
    # Apply simple Q-learning-style adjustment after moving
    def adjust_weights(self, reward):
        learning_rate = 0.01
        self.nn.weights_input_hidden += learning_rate * reward * np.random.randn(*self.nn.weights_input_hidden.shape)
        self.nn.bias_hidden += learning_rate * reward * np.random.randn(*self.nn.bias_hidden.shape)
        self.nn.weights_hidden_output += learning_rate * reward * np.random.randn(*self.nn.weights_hidden_output.shape)
        self.nn.bias_output += learning_rate * reward * np.random.randn(*self.nn.bias_output.shape)


    def reproduce(self):
        """Create offspring with slight position offset."""
        if self.type == "prey" and self.time_survived >= PREY_MIN_SURVIVAL_TIME and self.energy >= self.reproduction_threshold:
            # Prey can reproduce if they have survived for enough time and have enough energy
            offset_x = random.uniform(-20, 20)
            offset_y = random.uniform(-20, 20)
            
            offspring = Agent(
                self.x + offset_x, 
                self.y + offset_y, 
                self.type
            )
            
            # Reset the parent's energy and survival time
            self.energy /= 2  # Split energy with offspring
            self.time_survived = 0  # Reset survival time after reproduction

            return offspring
        elif self.type == "predator" and self.prey_eaten >= PREDATOR_PREY_EATEN_THRESHOLD:
            # Predator can reproduce if it has eaten enough prey
            offset_x = random.uniform(-20, 20)
            offset_y = random.uniform(-20, 20)

            offspring = Agent(
                self.x + offset_x, 
                self.y + offset_y, 
                self.type
            )

            # Reset predator's prey-eaten count
            self.prey_eaten = 0  # Reset prey eaten counter

            return offspring
        else:
            return None

    def manage_recovery(self):
        """Prey regains energy if they are stationary and recovering."""
        if self.type == "prey" and self.is_recovering:
            self.energy = min(ENERGY, self.energy + 0.1)  # Cap energy at 100
            if self.energy >= ENERGY:
                self.is_recovering = False  # Exit recovery mode
                self.is_stationary = False  # Start moving again
    
    def eat_prey(self, prey):
        """Predator eats prey and gains energy."""
        if self.digestion_cooldown == 0:
            prey.is_alive = False
            self.energy += 20
            self.digestion_cooldown = DIGESTION_COOLDOWN_TIME
            self.prey_eaten += 1  # Increase prey eaten counter
    
    def calculate_reward(self):
        """calculate reward for predators and prey"""
        if self.type == "predator" and self.energy > ENERGY:
            return 10 # reward for eating prey
        elif self.type == "prey" and self.is_alive:
            return 1 # reward for surviving
        else:
            return 0

    def draw(self, screen):
        """Draw the agent on the screen."""
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

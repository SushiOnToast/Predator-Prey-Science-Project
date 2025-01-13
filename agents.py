import random
import math
import pygame
from constants import *  # Assuming constants like ENERGY, PREDATOR_FOV, PREY_FOV, etc., are defined here

# Define Agent Class
class Agent:
    def __init__(self, x, y, type_, input_size, hidden_size, output_size):
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
        self.num_rays = 10
        self.range = 200 if self.type == "predator" else 50
        self.digestion_cooldown = 0 # digestion cooldown for predators
        
        self.reproduction_threshold = 50 if type_ == "prey" else 10  # Thresholds for reproduction
        self.time_survived = 0  # Track survival time for prey
        self.prey_eaten = 0  # Track number of prey eaten by predators

        # Neural Network for decision making
        self.nn = NeuralNetwork(input_size, hidden_size, output_size)

    def move(self, screen, other_agents):
        """Move the agent based on its direction, speed, and neural network output."""
        if self.is_alive:
            self.time_survived += 1

            if self.energy > 0 and not self.is_recovering:  # Only move if not recovering
                # Get input from the environment (distances from rays)
                ray_inputs = self.cast_rays(screen, other_agents)
                ray_inputs = np.array(ray_inputs).flatten()  # Flatten the list into a single array for input

                # Perform forward pass through the neural network
                nn_output = self.nn.forward(ray_inputs)

                # Extract the output values (direction and speed)
                # Assuming the network's output has two values: speed and direction change
                self.speed = np.clip(nn_output[0] * 10, 0, 10)  # Speed is between 0 and 10
                self.direction += nn_output[1] * math.pi  # Direction change is scaled by pi

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

            # Reproduction logic (same as before)

    def cast_rays(self, screen, other_agents):
        """Cast rays in the agent's FOV and return the distances to the closest agent."""
        ray_intersections = []
        step_angle = self.fov_angle / max(1, (self.num_rays - 1))  # the angle between each of the rays

        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * math.radians(step_angle)  # Distribute rays within the FOV
            ray_direction = self.direction + angle_offset
            intersections = self._cast_ray(ray_direction, screen, other_agents)
            ray_intersections.append(intersections)

        return ray_intersections

    def _cast_ray(self, angle, screen, other_agents):
        """Cast a single ray and return all intersections as (distance, object) tuples."""
        dx = math.cos(angle)
        dy = math.sin(angle)

        intersections = []

        for t in range(1, int(self.range + 1)):
            x = self.x + t * dx
            y = self.y + t * dy

            # Stop at screen borders
            if not (0 <= x < screen.get_width() and 0 <= y < screen.get_height()):
                break

            for agent in other_agents:
                if agent.is_alive and self._check_collision(x, y, agent):
                    intersections.append((t, agent))  # Append distance and agent

        # Return the closest intersection or the max range
        if intersections:
            return intersections
        else:
            return [(self.range, None)]  # No intersection within range

    def _check_collision(self, x, y, agent):
        """Check if the point x, y collides with the agent."""
        return math.sqrt((x - agent.x)**2 + (y - agent.y)**2) < agent.size

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
            self.energy += 20
            self.digestion_cooldown = DIGESTION_COOLDOWN_TIME
            prey.is_alive = False
            self.prey_eaten += 1  # Increase prey eaten counter
            
    def draw(self, screen):
        """Draw the agent on the screen."""
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

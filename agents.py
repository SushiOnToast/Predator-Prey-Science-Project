import random
import math
import pygame
from constants import *  

# Define Agent Class
class Agent:
    def __init__(self, x, y, type_):
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
        self.digestion_cooldown = 0 # digestion cooldown for predators to ensure that they dont gain energy while its active, to prevent overaccumulation of energy

    def move(self, screen, other_agents):
        """Move the agent based on its direction and speed."""
        if self.is_alive:
            if self.energy > 0 and not self.is_recovering:  # Only move if not recovering
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

            if self.digestion_cooldown > 0:
                self.digestion_cooldown -= 1

            # Now handle the predator’s eating logic using FOV and ray-casting
            if self.type == "predator":
                # Cast rays to detect prey within the FOV
                ray_distances = self.cast_rays(screen, other_agents)

                for i, distance in enumerate(ray_distances):
                    angle_offset = (i - (self.num_rays // 2))  # Calculate the angle offset for each ray
                    ray_direction = self.direction + math.radians(angle_offset)

                    # Check if prey is in range of any ray
                    for agent in other_agents:
                        if agent.type == "prey" and agent.is_alive:
                            distance_to_prey = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                            if distance_to_prey <= self.size + agent.size and distance_to_prey <= distance:
                                self.eat_prey(agent)  # Try to eat prey

    def cast_rays(self, screen, other_agents):
        """cast rays in the agent's FOV and return the distances to the closest agent"""
        ray_distances = []
        step_angle = self.fov_angle / max(1, (self.num_rays - 1)) # the angle between each of the rays

        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * math.radians(step_angle) # distrubute rays within the FOV
            ray_direction = self.direction + angle_offset
            distance = self._cast_ray(ray_direction, screen, other_agents)
            ray_distances.append(distance)

        return ray_distances
    
    def _cast_ray(self, angle, screen, other_agents):
        """cast a single ray and return the distance to the closet object"""
        dx = math.cos(angle) 
        dy = math.sin(angle) 

        # loop to chec collisions
        for t in range(1, int(self.range + 1)):
            x = self.x + t * dx 
            y = self.y + t * dy 

            if not (0 <= x < screen.get_width() and 0 <= y < screen.get_height()):  # Stop at screen borders
                return t

            for agent in other_agents:
                if agent.is_alive and self._check_collision(x, y, agent):
                    return t  # Distance to the closest agent

        return self.range  # Max range if no collision
    
    def _check_collision(self, x, y, agent):
        """checs if the point x, y collides with the agent"""
        # its a simple distance ceh, if its within the agent's size
        return math.sqrt((x - agent.x)**2 + (y - agent.y)**2) < agent.size
    
    def draw_rays(self, screen):
        """Visualize rays cast by the agent for debugging."""
        step_angle = self.fov_angle / (self.num_rays - 1)
        for i in range(self.num_rays):
            angle_offset = (i - (self.num_rays // 2)) * math.radians(step_angle)
            ray_direction = self.direction + angle_offset
            end_x = self.x + self.range * math.cos(ray_direction)
            end_y = self.y + self.range * math.sin(ray_direction)
            pygame.draw.line(screen, "gray", (self.x, self.y), (end_x, end_y), 1)


    def draw(self, screen):
        """Draw the agent on the screen."""
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)
        if self.type == "predator":
            self.draw_rays(screen)


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


        
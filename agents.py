import random
import math
import pygame
from constants import *  # Assuming constants like RED, GREEN, WIDTH, HEIGHT, FPS are defined in a separate file

# Define Agent Class
class Agent:
    def __init__(self, x, y, type_):
        self.x = x
        self.y = y
        self.direction = random.uniform(0, 2 * math.pi)  # Random initial direction
        self.speed = 2  # Constant speed
        self.energy = 100  # Starting energy
        self.type = type_  # Predator or Prey
        self.size = 10
        self.energy_depletion_rate = 0.1 if self.type == "prey" else 0.2  # Predators deplete energy faster
        self.is_alive = True
        self.is_stationary = False  # Track if prey is stationary to regain energy
        self.is_recovering = False  # Track if prey is recovering energy

        self.fov_angle = 60 if self.type == "predator" else 270
        self.num_rays = 5 if self.type == "predator" else 7
        self.range = 200 if self.type == "predator" else 50

    def move(self, screen):
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

 

    def draw(self, screen):
        """Draw the agent on the screen."""
        color = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

    def manage_recovery(self):
        """Prey regains energy if they are stationary and recovering."""
        if self.type == "prey" and self.is_recovering:
            if self.energy < 100:
                self.energy += 0.1  # Slowly regain energy if still stationary
            if self.energy >= 100:  # Prey can move again once energy reaches 50
                self.is_recovering = False  # Exit recovery mode
                self.is_stationary = False  # Start moving again
            if self.energy > 100:
                self.energy = 100  # Cap energy at 100

    def can_move(self):
        """Check if prey can start moving again."""
        if self.is_recovering:
            return self.energy >= 100  # Prey can move again when energy reaches 50 during recovery
        return self.energy > 0  # Prey can move when energy is above 0 and it's not recovering

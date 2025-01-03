import random
import math
import pygame
from constants import *

# we want to define an agent class with position, direction, speed and energy
class Agent:
    def __init__(self, x, y, type_): # the underscore after type lets us use the word type (its actually a kwyword)
        self.x = x
        self.y = y
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 2 # we'll start off with constant speed
        self.energy = 100 # starting energy
        self.type = type_ # predator or prey
        self.size = 10

    def move(self, screen):
        """move agent based on its direction and speed"""
        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)

        # Adjust position if agent hits the screen border
        if self.x - self.size < 0:
            self.x = self.size
        elif self.x + self.size   > screen.get_width():
            self.x = screen.get_width() - self.size 
        if self.y - self.size < 0:
            self.y = self.size
        elif self.y + self.size  > screen.get_height():
            self.y = screen.get_height() - self.size 
    
    def draw(self, screen):
        """draw agent on the screen"""
        colour = RED if self.type == "predator" else GREEN
        pygame.draw.circle(screen, colour, (int(self.x), int(self.y)), self.size)
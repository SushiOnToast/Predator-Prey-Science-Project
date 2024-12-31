# this file will contain all the predator prey related simulation details for the project
import math
from constants import *
import pygame
import random

# todo
# improve stamina mechanics
# mae sure chasinh only happens when within the polygon - colliderect?
# momentum while turninh

# Agent classes
class Agent:
    def __init__(self, x, y, colour, radius, speed, angle, vision_distance, vision_angle):
        self.x = x
        self.y = y
        self.colour = colour
        self.speed = speed
        self.angle = angle # to determine which way the agent is "looking" 
        self.radius = radius # the collision radius 
        self.vision_distance = vision_distance # how far the agent can see 
        self.vision_angle = vision_angle # the angle at which the cone of vision spans

        self.dx = math.cos(math.radians(self.angle)) * self.speed
        self.dy = math.sin(math.radians(self.angle)) * self.speed
        self.stamina = 100 # max stamina
        self.resting = False
        self.wander_timer = 0

    def move(self):
        if not self.resting:
            self.x += self.dx
            self.y += self.dy
            self.stamina -= 0.2 # drain stamina while moving
            if self.stamina <= 0:
                self.resting = True
        else:
            self.stamina += 0.5 # recover stamina while resting
            if self.stamina >= 100:
                self.resting = False

        # checing for border collisions and handling
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.dx = -self.dx
            self.angle = 180 - self.angle
            self.update_direction()

        # checing for border collisions and handling
        if self.y - self.radius < 0 or self.y + self.radius > HEIGHT:
            self.dy = -self.dy
            self.angle = -self.angle
            self.update_direction()

        # Update angle based on current direction
        self.angle = math.degrees(math.atan2(self.dy, self.dx)) % 360
    
    def accelerate(self):
        self.speed += ACCELERATION
        self.update_direction()

    def decerlerate(self):
        self.speed -= DECELERATION
        self.update_direction()

    def update_direction(self):
        self.dx = math.cos(math.radians(self.angle)) * self.speed
        self.dy = math.sin(math.radians(self.angle)) * self.speed
    
    def turn(self, angle_change):
        self.angle = (self.angle + angle_change) % 360
        self.update_direction()
    
    def wander(self):
        self.wander_timer += 1
        if self.wander_timer > random.randint(50, 500):
            self.angle = (self.angle + random.randint(-45, 45)) % 360 #slight change in direction
            self.update_direction()
            self.wander_timer = 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.colour, ((int(self.x)), (int(self.y))), self.radius)

        # vision cone
        start_angle = math.radians(self.angle - self.vision_angle/2)
        end_angle = math.radians(self.angle + self.vision_angle/2)
        
        # Define the points for the vision cone as a polygon for better alignment
        vision_points = [
            (self.x, self.y),  # Agent's current position
            (self.x + math.cos(start_angle) * self.vision_distance,
            self.y + math.sin(start_angle) * self.vision_distance),  # Start of vision cone
            (self.x + math.cos(end_angle) * self.vision_distance,
            self.y + math.sin(end_angle) * self.vision_distance),  # End of vision cone
        ]

        pygame.draw.polygon(screen, self.colour, vision_points, 1)

        # stamina bar
        pygame.draw.rect(screen, RED if self.resting else GREEN,
                         (self.x - 10, self.y - 20, self.stamina/2, 5))
    

class Predator(Agent):
    def chase(self, prey_list):
        closest_prey = None
        closest_distance = float('inf')

        for prey in prey_list:
            distance = math.sqrt((self.x - prey.x)**2 + (self.y-prey.y)**2) # evaluating distance to each prey
            angle_to_prey = math.degrees(math.atan2(prey.y - self.y, prey.x - self.x)) % 360 # evaluating difference in angle to prey

            if distance < closest_distance and self.in_vision(angle_to_prey):
                closest_prey = prey
                closest_distance = distance

        if closest_prey:
            self.angle = math.degrees(math.atan2(closest_prey.y - self.y, closest_prey.x - self.x)) # face in that angle
            self.update_direction()
        else:
            self.wander()
    
    def in_vision(self, angle_to_target):
        delta_angle = (angle_to_target - self.angle + 360) % 360 # difference in angle, extra 360 added to determine the shortest angle
        return delta_angle <= self.vision_angle / 2 or delta_angle >= 360 - self.vision_angle / 2 # determining if its in either half of the vision cone


class Prey(Agent):
    def evade(self, predator_list):
        for predator in predator_list:
            distance = math.sqrt((self.x - predator.x)**2 + (self.y-predator.x)**2)
            if distance < self.vision_distance:
                # move away from the predator
                self.angle = (math.degrees(math.atan2(self.y-predator.y, self.x-predator.x)) + 180) % 360 # calculating the opposite angle to go
                self.update_direction()
        self.wander()

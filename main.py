import pygame
import random
import math
from constants import *
from agents import *

# initialise pygame
pygame.init()

# screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Simulation")

# initialise agents
prey_list = [Prey(random.randint(0, WIDTH), random.randint(0, HEIGHT), BLUE, 5, random.uniform(0.5, 2), random.uniform(0, 360), random.randint(130, 150), random.uniform(60, 90)) for _ in range(NUM_PREY)]
predator_list = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT), RED, 8, random.uniform(1.5, 3.5), random.uniform(0, 360), random.randint(130, 150), random.uniform(60, 90)) for _ in range(NUM_PREDATORS)]

# main loop
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # update and draw predators
    for predator in predator_list:
        predator.chase(prey_list)
        predator.move()
        predator.draw(screen)

    # update and draw prey
    for prey in prey_list:
        prey.evade(predator_list)
        prey.move()
        prey.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)
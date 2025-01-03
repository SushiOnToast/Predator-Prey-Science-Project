import pygame
import random
import math
from constants import *
from agents import *

# initialise pygame
pygame.init()

# screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Predator-Prey Simulation")

def toggle_fullscreen():
    global screen, WIDTH, HEIGHT
    is_fullscreen = pygame.display.get_surface().get_flags() & pygame.FULLSCREEN
    if is_fullscreen:
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

# initialise agents
prey_list = [Prey(random.randint(0, WIDTH), random.randint(0, HEIGHT), BLUE, 5, random.uniform(0.5, 2), random.uniform(0, 360), random.randint(260, 300), random.uniform(30, 70), random.randint(100, 150), random.uniform(0.5, 2)) for _ in range(NUM_PREY)]
predator_list = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT), RED, 8, random.uniform(0.75, 2.25), random.uniform(0, 360), random.randint(260, 300), random.uniform(30, 70), random.randint(100, 150), random.uniform(0.5, 2)) for _ in range(NUM_PREDATORS)]

# main loop
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # update and draw predators
    for predator in predator_list:
        predator.chase(prey_list, screen)
        predator.move(screen)
        predator.draw(screen)

    # update and draw prey
    for prey in prey_list:
        prey.evade(predator_list, screen)
        prey.move(screen)
        prey.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)
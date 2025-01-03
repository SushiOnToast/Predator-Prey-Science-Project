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

# main loop
def run_simulation():
    agents = [Agent(random.randint(50, 750), random.randint(50, 550), random.choice(["predator", "prey"])) for _ in range(50)]

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for agent in agents:
            agent.move(screen)
            agent.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

run_simulation()
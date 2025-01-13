import pygame
import random
from constants import *
from agents import *
from debug import debug_text

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

# Main loop to run the simulation
def run_simulation():
    agents = [Agent(random.randint(50, 750), random.randint(50, 550), random.choice(["predator", "prey"])) for _ in range(NUM_AGENTS)]

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Remove dead agents
        agents = [agent for agent in agents if agent.is_alive]
        new_agents = []
        tracked_prey = None
        for agent in agents:
            if agent.type == "predator":
                tracked_prey = agent
                break  # Stop when the first prey is found

        # Move and draw agents
        for agent in agents:
            agent.move(screen, agents)  # Move the agent (with recovery handling within the move method)
            agent.draw(screen)
            # Handle reproduction
            offspring = agent.reproduce()
            if offspring:
                new_agents.append(offspring)
        
        agents.extend(new_agents)

        # Track the prey being followed
        # if tracked_prey.is_alive:
        #     pygame.draw.rect(screen, BLACK, (tracked_prey.x - 10, tracked_prey.y - 10, 20, 20), 2)  # Highlight the tracked prey
        #     debug_text(screen, str(tracked_prey.is_recovering))
        #     debug_text(screen, str(tracked_prey.energy), 0, 15)
        #     debug_text(screen, str(tracked_prey.is_stationary), 0, 30)

        pygame.display.flip()
        clock.tick(FPS)

# Start simulation
run_simulation()
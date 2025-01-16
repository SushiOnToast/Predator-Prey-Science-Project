import pygame
import random
from constants import *
from agents import Agent
from debug import debug_text

class Simulation:
    def __init__(self):
        # Initialize pygame
        pygame.init()

        # Screen setup
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Predator-Prey Simulation")
        self.clock = pygame.time.Clock()

        # List to store agents
        self.agents = []
        self.running = True
        self.load_agents()

    def toggle_fullscreen(self):
        is_fullscreen = pygame.display.get_surface().get_flags() & pygame.FULLSCREEN
        if is_fullscreen:
            pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def load_agents(self):
        """Initialize agents (predators and prey)"""
        self.agents = [Agent(random.randint(50, 750), random.randint(50, 550), random.choice(["predator", "prey"])) for _ in range(NUM_AGENTS)]

    def remove_dead_agents(self):
        """Remove dead agents from the simulation"""
        self.agents = [agent for agent in self.agents if agent.is_alive]

    def handle_reproduction(self):
        """Handle agent reproduction"""
        new_agents = []
        for agent in self.agents:
            offspring = agent.reproduce()
            if offspring:
                new_agents.append(offspring)
        self.agents.extend(new_agents)

    def handle_movement(self):
        """Move and draw agents on the screen"""
        for agent in self.agents:
            agent.move(self.screen, self.agents)  # Move the agent (with recovery handling within the move method)
            agent.draw(self.screen)

    def update_display(self):
        """Update the display every frame"""
        pygame.display.flip()

    def run(self):
        """Main loop to run the simulation"""
        while self.running:
            self.screen.fill(WHITE)
            self.handle_events()
            self.remove_dead_agents()
            self.handle_movement()
            self.handle_reproduction()
            self.update_display()
            self.clock.tick(FPS)

    def handle_events(self):
        """Handle user inputs and events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


# Run the simulation
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()

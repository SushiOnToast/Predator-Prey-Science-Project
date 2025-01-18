import pygame
import random
import math
from constants import *
from agents import Agent
from debug import debug_text


class SpatialGrid:
    def __init__(self, screen_width, screen_height, cell_size):
        self.cell_size = cell_size
        self.grid_width = math.ceil(screen_width / cell_size)
        self.grid_height = math.ceil(screen_height / cell_size)
        self.cells = [[[] for _ in range(self.grid_width)] for _ in range(self.grid_height)]

    def clear(self):
        """Clear all cells before recalculating positions."""
        for row in self.cells:
            for cell in row:
                cell.clear()

    def add_agent(self, agent):
        """Add an agent to the grid based on its position."""
        grid_x = int(agent.x // self.cell_size)
        grid_y = int(agent.y // self.cell_size)
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.cells[grid_y][grid_x].append(agent)

    def get_nearby_agents(self, agent):
        """Get agents in the same and neighboring cells."""
        grid_x = int(agent.x // self.cell_size)
        grid_y = int(agent.y // self.cell_size)
        nearby_agents = []

        for dx in range(-1, 2):  # Check neighboring cells (-1, 0, +1)
            for dy in range(-1, 2):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    nearby_agents.extend(self.cells[ny][nx])

        return nearby_agents

    def draw(self, screen):
        """Visualize the grid on the screen."""
        for x in range(0, self.grid_width * self.cell_size, self.cell_size):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, screen.get_height()))  # Vertical lines
        for y in range(0, self.grid_height * self.cell_size, self.cell_size):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (screen.get_width(), y))  # Horizontal lines


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

        # Initialize spatial grid
        self.cell_size = 50  # Adjust based on agent density and environment size
        self.spatial_grid = SpatialGrid(WIDTH, HEIGHT, self.cell_size)

        # Load agents
        self.load_agents()

    def toggle_fullscreen(self):
        is_fullscreen = pygame.display.get_surface().get_flags() & pygame.FULLSCREEN
        if is_fullscreen:
            pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def load_agents(self):
        """Initialize agents (predators and prey)."""
        self.agents = [Agent(random.randint(50, 750), random.randint(50, 550), random.choice(["predator", "prey"])) for _ in range(NUM_AGENTS)]

    def remove_dead_agents(self):
        """Remove dead agents from the simulation."""
        self.agents = [agent for agent in self.agents if agent.is_alive]

    def handle_reproduction(self):
        """Handle agent reproduction."""
        new_agents = []
        for agent in self.agents:
            offspring = agent.reproduce()
            if offspring:
                new_agents.append(offspring)
        self.agents.extend(new_agents)

    def handle_movement(self):
        """Move and draw agents on the screen."""
        # Clear the spatial grid
        self.spatial_grid.clear()

        # Add agents to the grid
        for agent in self.agents:
            self.spatial_grid.add_agent(agent)

        # Move agents and only consider nearby agents
        for agent in self.agents:
            nearby_agents = self.spatial_grid.get_nearby_agents(agent)
            agent.move(self.screen, nearby_agents)  # Pass only nearby agents to the agent's move logic
            agent.draw(self.screen)

    def update_display(self):
        """Update the display every frame."""
        pygame.display.flip()

    def run(self):
        """Main loop to run the simulation."""
        while self.running:
            self.screen.fill(WHITE)

            self.spatial_grid.draw(self.screen)

            debug_text(self.screen, str(self.agents[0].speed))
            debug_text(self.screen, str(self.agents[0].direction), 0, 20)

            self.handle_events()
            self.remove_dead_agents()
            self.handle_movement()
            self.handle_reproduction()
            self.update_display()
            self.clock.tick(FPS)

    def handle_events(self):
        """Handle user inputs and events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


# Run the simulation
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()

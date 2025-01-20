import pygame
import random
import math
from constants import *
from agents import Agent
from debug import debug_text
from fitness_tracker import FitnessTracker


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
                cell[:] = []  # Clears the list of agents in the cell

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
        self.generation = 0
        self.steps_since_last_generation = 0

        self.fitness_tracker = FitnessTracker()

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
        self.agents = [
            Agent(
                random.randint(50, 750),
                random.randint(50, 550),
                random.choice(["predator", "prey"]),
            )
            for _ in range(NUM_AGENTS)
        ]

    def evaluate_fitness(self):
        """Evaluate the fitness of each agent less frequently."""
        if self.steps_since_last_generation % 10 == 0:  # Evaluate fitness every 10 frames
            for agent in self.agents:
                agent.update_fitness()

    def rank_based_selection(self):
        """Select parents using rank-based selection."""
        self.agents.sort(key=lambda agent: agent.fitness, reverse=True)
        total_ranks = sum(range(1, len(self.agents) + 1))
        probabilities = [(len(self.agents) - rank) / total_ranks for rank in range(len(self.agents))]
        num_parents = len(self.agents) // 2
        parents = random.choices(self.agents, weights=probabilities, k=num_parents)
        parents = list(set(parents))
        parents.extend(self.agents[:2])  # Add top 2 agents for elitism
        parents = parents[:num_parents]
        return parents

    def crossover(self, parents):
        """Perform crossover to generate new offspring."""
        offspring = []
        for _ in range(len(self.agents) - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child_nn = parent1.nn.crossover(parent2.nn)
            parent = random.choice([parent1, parent2])
            offset_x = random.randint(-self.cell_size, self.cell_size)
            offset_y = random.randint(-self.cell_size, self.cell_size)
            child_x = max(0, min(parent.x + offset_x, WIDTH))
            child_y = max(0, min(parent.y + offset_y, HEIGHT))
            child_type = parent.type
            child = Agent(child_x, child_y, child_type, nn=child_nn)
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        """Apply mutations to the offspring's neural networks."""
        for child in offspring:
            child.nn.mutate()

    def remove_dead_agents(self):
        """Remove dead agents from the simulation."""
        self.agents = [agent for agent in self.agents if agent.is_alive]

    def handle_movement(self):
        """Move and draw agents on the screen."""
        self.spatial_grid.clear()
        for agent in self.agents:
            self.spatial_grid.add_agent(agent)
        for agent in self.agents:
            nearby_agents = self.spatial_grid.get_nearby_agents(agent)
            agent.move(self.screen, nearby_agents)
            agent.draw(self.screen)

    def update_display(self):
        """Update the display every frame."""
        pygame.display.flip()

    def run(self):
        """Main loop to run the simulation."""
        while self.running:
            self.screen.fill(WHITE)
            self.spatial_grid.draw(self.screen)

            if self.steps_since_last_generation >= STEPS_PER_GENERATION:
                self.generation += 1
                self.steps_since_last_generation = 0
                self.evaluate_fitness()
                parents = self.rank_based_selection()
                offspring = self.crossover(parents)
                self.mutate(offspring)
                self.agents = parents + offspring

                # log fitness at end of generation
                predator_fitness = [agent.fitness for agent in self.agents if agent.type == "predator"]
                prey_fitness = [agent.fitness for agent in self.agents if agent.type == "prey"]
                self.fitness_tracker.log_fitness(self.generation, predator_fitness, prey_fitness)

            debug_text(self.screen, f"Generation: {self.generation}", 0, 40)
            self.steps_since_last_generation += 1
            self.handle_events()
            self.handle_movement()
            self.remove_dead_agents()
            self.update_display()
            self.clock.tick(FPS)
            
        self.fitness_tracker.plot_fitness()

    def handle_events(self):
        """Handle user inputs and events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


# Run the simulation
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()

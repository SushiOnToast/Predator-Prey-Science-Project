import pygame
import random
import math
from constants import *
from agents import Agent
from debug import debug_text
from fitness_tracker import FitnessTracker
import neat


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
    def __init__(self, neat_config, generation_duration=30):
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

        # Control time per generation (in seconds)
        self.generation_duration = generation_duration  # Time per generation in seconds
        self.start_time = pygame.time.get_ticks()  # Track the time when the generation starts

        self.fitness_tracker = FitnessTracker()

        # Load NEAT config
        self.neat_config = neat_config
        # Create the populations for predators and prey
        self.predator_population = neat.Population(self.neat_config)
        self.prey_population = neat.Population(self.neat_config)

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
        self.agents = []
        for genome_id, genome in self.predator_population.population.items():
            nn = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            agent = Agent(
                random.randint(50, 750),
                random.randint(50, 550),
                "predator",
                nn=nn,
                genome_id=genome_id
            )
            self.agents.append(agent)

        for genome_id, genome in self.prey_population.population.items():
            nn = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            agent = Agent(
                random.randint(50, 750),
                random.randint(50, 550),
                "prey",
                nn=nn,
                genome_id=genome_id
            )
            self.agents.append(agent)

    def evaluate_fitness(self, genomes, config):
        """Evaluate the fitness of each agent less frequently."""
        if self.steps_since_last_generation % 10 == 0:  # Evaluate fitness every 10 frames
            for genome_id, genome in genomes:
                agent = self.get_agent_by_genome_id(genome_id)
                if agent is not None:
                    agent.update_fitness()
                    # Ensure that fitness is set for the genome
                    if genome.fitness is None:  # If fitness is None, set it to the agent's fitness
                        genome.fitness = agent.fitness
                    # Additional check in case agent's fitness is None
                    if genome.fitness is None:
                        genome.fitness = 0  # Set a default fitness value
                else:
                    print(f"Warning: Agent with genome_id {genome_id} not found.")
                    genome.fitness = 0
                    # When initializing genomes in NEAT:
        for genome_id, genome in genomes:
            if genome.fitness is None:
                genome.fitness = 0  # Set a default fitness value for all genomes



    def get_agent_by_genome_id(self, genome_id):
        """Get the agent corresponding to a specific genome ID."""
        for agent in self.agents:
            if agent.genome_id == genome_id:
                return agent
        print(f"Agent with genome_id {genome_id} not found.")
        return None

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
        font = pygame.font.Font(None, 36)  # Create a font object with the desired size
        while self.running:
            self.screen.fill(WHITE)
            self.spatial_grid.draw(self.screen)

            current_time = pygame.time.get_ticks()
            if current_time - self.start_time >= self.generation_duration * 1000:  # Check if generation time has passed
                self.generation += 1
                self.start_time = current_time  # Reset the start time for the next generation

                # Perform NEAT operations for both populations (predators and prey)
                self.evaluate_fitness(self.predator_population.population.items(), self.neat_config)
                self.predator_population.run(self.evaluate_fitness, 3)  # Run one generation for predators

                self.evaluate_fitness(self.prey_population.population.items(), self.neat_config)
                self.prey_population.run(self.evaluate_fitness, 3)  # Run one generation for prey

                # Retrieve the new generation of agents
                self.agents.extend(self.get_new_agents())

                # Log fitness and handle generation-related tasks
                predators_exist = any(agent.type == "predator" for agent in self.agents)
                prey_exist = any(agent.type == "prey" for agent in self.agents)

                if not predators_exist or not prey_exist:
                    message = "Predators extinct!" if not predators_exist else "Prey extinct!"
                    debug_text(self.screen, message, WIDTH // 2 - 50, HEIGHT // 2, 20)
                    break
                
                # Extract only the genome objects as lists
                predator_genomes = [genome for _, genome in self.predator_population.population.items()]
                prey_genomes = [genome for _, genome in self.prey_population.population.items()]

                # Gather fitness lists (if needed)
                predator_fitness = [genome.fitness for genome in predator_genomes]
                prey_fitness = [genome.fitness for genome in prey_genomes]

                # Pass genome lists directly to log_fitness
                self.fitness_tracker.log_fitness(
                    generation=self.generation,
                    predator_fitness=predator_fitness,
                    prey_fitness=prey_fitness,
                    predators=[agent for agent in self.agents if agent.type == "predator"],
                    preys=[agent for agent in self.agents if agent.type == "prey"]
                )

                self.steps_since_last_generation = 0
            else:
                self.steps_since_last_generation += 1

            self.handle_movement()
            self.update_display()

            # Display the generation count on the screen
            generation_text = font.render(f"Generation: {self.generation}", True, (0, 0, 0))  # Black text
            self.screen.blit(generation_text, (10, 10))  # Position it at the top left

            self.remove_dead_agents()
            self.check_for_exit_events()
        self.fitness_tracker.plot_fitness()


    def get_new_agents(self):
        """Retrieve the new generation of agents after NEAT has run."""
        new_agents = []
        for genome_id, genome in self.predator_population.population.items():
            nn = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            agent = Agent(
                random.randint(0, self.screen.get_width()-10),
                random.randint(0, self.screen.get_height()-10),
                "predator",
                nn=nn,
                genome_id=genome_id
            )
            new_agents.append(agent)

        for genome_id, genome in self.prey_population.population.items():
            nn = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            agent = Agent(
                random.randint(0, self.screen.get_width()-10),
                random.randint(0, self.screen.get_height()-10),
                "prey",
                nn=nn,
                genome_id=genome_id
            )
            new_agents.append(agent)

        return new_agents

    def check_for_exit_events(self):
        """Check for any quit events to exit the simulation."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:  # Toggle fullscreen on 'F' key press
                    self.toggle_fullscreen()
                elif event.key == pygame.K_q:  # Quit the simulation on 'Q' key press
                    self.running = False


if __name__ == "__main__":
    config_path = "config-feedforward.txt"
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, config_path)
    simulation = Simulation(config)
    simulation.run()

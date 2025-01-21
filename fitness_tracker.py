import pandas as pd
import matplotlib.pyplot as plt

class FitnessTracker:
    def __init__(self):
        # Initialize the dictionary to store fitness data and additional metrics
        self.fitness_data = {
            'generation': [],
            'top_predator_fitness': [],
            'top_prey_fitness': [],
            'average_predator_fitness': [],
            'average_prey_fitness': [],
            'average_predator_energy': [],  # Changed to average
            'average_prey_energy': [],  # Changed to average
            'average_predator_prey_eaten': [],  # Changed to average
            'average_prey_survival_time': [],  # Changed to average
            'average_predator_distance': [],  # Changed to average
            'average_prey_distance': []  # Changed to average
        }

    def log_fitness(self, generation, predator_fitness, prey_fitness, predators, preys):
        """Log fitness data for the current generation."""
        
        # Calculate metrics for the generation
        top_predator_fitness = max((fitness for fitness in predator_fitness if fitness is not None), default=0)
        top_prey_fitness = max((fitness for fitness in prey_fitness if fitness is not None), default=0)
        avg_predator_fitness = sum((fitness for fitness in predator_fitness if fitness is not None)) / len(predator_fitness) if predator_fitness else 0
        avg_prey_fitness = sum((fitness for fitness in prey_fitness if fitness is not None)) / len(prey_fitness) if prey_fitness else 0
        
        # Additional metrics (changed to average)
        avg_predator_energy = sum(agent.energy for agent in predators) / len(predators) if predators else 0
        avg_prey_energy = sum(agent.energy for agent in preys) / len(preys) if preys else 0
        avg_predator_prey_eaten = sum(agent.prey_eaten for agent in predators) / len(predators) if predators else 0
        avg_prey_survival_time = sum(agent.time_survived for agent in preys) / len(preys) if preys else 0
        
        # Track average distance traveled by predators and prey
        avg_predator_distance = sum(agent.distance_traveled for agent in predators) / len(predators) if predators else 0
        avg_prey_distance = sum(agent.distance_traveled for agent in preys) / len(preys) if preys else 0
        
        # Store the data
        self.fitness_data['generation'].append(generation)
        self.fitness_data['top_predator_fitness'].append(top_predator_fitness)
        self.fitness_data['top_prey_fitness'].append(top_prey_fitness)
        self.fitness_data['average_predator_fitness'].append(avg_predator_fitness)
        self.fitness_data['average_prey_fitness'].append(avg_prey_fitness)
        self.fitness_data['average_predator_energy'].append(avg_predator_energy)
        self.fitness_data['average_prey_energy'].append(avg_prey_energy)
        self.fitness_data['average_predator_prey_eaten'].append(avg_predator_prey_eaten)
        self.fitness_data['average_prey_survival_time'].append(avg_prey_survival_time)
        self.fitness_data['average_predator_distance'].append(avg_predator_distance)
        self.fitness_data['average_prey_distance'].append(avg_prey_distance)

    def plot_fitness(self):
        """Plot the fitness data and additional metrics."""
        
        # Convert the fitness data to a DataFrame for easier plotting
        df = pd.DataFrame(self.fitness_data)

        # Plot fitness data
        plt.figure(figsize=(14, 10))
        plt.style.use('seaborn-v0_8')

        # Plot top fitness values
        plt.subplot(3, 2, 1)
        plt.plot(df['generation'], df['top_predator_fitness'], label='Top Predator Fitness', color='red', marker='o')
        plt.plot(df['generation'], df['top_prey_fitness'], label='Top Prey Fitness', color='green', marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()

        # Plot energy levels
        plt.subplot(3, 2, 2)
        plt.plot(df['generation'], df['average_predator_energy'], label='Average Predator Energy', color='orange', marker='o')
        plt.plot(df['generation'], df['average_prey_energy'], label='Average Prey Energy', color='blue', marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Energy')
        plt.title('Energy Levels')
        plt.legend()

        # Plot number of prey eaten by predators
        plt.subplot(3, 2, 3)
        plt.plot(df['generation'], df['average_predator_prey_eaten'], label='Average Prey Eaten', color='purple', marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Prey Eaten')
        plt.title('Predator Prey Eaten Count')
        plt.legend()

        # Plot prey survival time
        plt.subplot(3, 2, 4)
        plt.plot(df['generation'], df['average_prey_survival_time'], label='Average Prey Survival Time', color='cyan', marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Survival Time')
        plt.title('Prey Survival Time')
        plt.legend()

        # Plot distance traveled by predators
        plt.subplot(3, 2, 5)
        plt.plot(df['generation'], df['average_predator_distance'], label='Average Predator Distance', color='brown', marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Distance Traveled')
        plt.title('Predator Distance Traveled')
        plt.legend()

        # Plot distance traveled by prey
        plt.subplot(3, 2, 6)
        plt.plot(df['generation'], df['average_prey_distance'], label='Average Prey Distance', color='yellow', marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Distance Traveled')
        plt.title('Prey Distance Traveled')
        plt.legend()

        plt.tight_layout()
        plt.show()

"""File for tracking and visualising agent fitness data 
for debugging and general observation"""

import pandas as pd
import matplotlib.pyplot as plt

class FitnessTracker:
    def __init__(self):
        # Initialize the dictionary to store fitness data
        self.fitness_data = {
            'generation': [],
            'top_predator_fitness': [],
            'top_prey_fitness': [],
            'average_predator_fitness': [],
            'average_prey_fitness': []
        }

    def log_fitness(self, generation, predator_fitness, prey_fitness):
        # Log the fitness data for the current generation
        top_predator_fitness = max(predator_fitness)
        top_prey_fitness = max(prey_fitness)
        avg_predator_fitness = sum(predator_fitness) / len(predator_fitness)
        avg_prey_fitness = sum(prey_fitness) / len(prey_fitness)

        # Store the data
        self.fitness_data['generation'].append(generation)
        self.fitness_data['top_predator_fitness'].append(top_predator_fitness)
        self.fitness_data['top_prey_fitness'].append(top_prey_fitness)
        self.fitness_data['average_predator_fitness'].append(avg_predator_fitness)
        self.fitness_data['average_prey_fitness'].append(avg_prey_fitness)

    def plot_fitness(self):
        # Convert the fitness data to a DataFrame for easier plotting
        df = pd.DataFrame(self.fitness_data)

        # Plot top fitness values
        plt.figure(figsize=(10, 6))
        plt.plot(df['generation'], df['top_predator_fitness'], label='Top Predator Fitness', color='red', marker='o')
        plt.plot(df['generation'], df['top_prey_fitness'], label='Top Prey Fitness', color='green', marker='o')

        # Plot average fitness values
        plt.plot(df['generation'], df['average_predator_fitness'], label='Average Predator Fitness', color='orange', linestyle='--')
        plt.plot(df['generation'], df['average_prey_fitness'], label='Average Prey Fitness', color='blue', linestyle='--')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()

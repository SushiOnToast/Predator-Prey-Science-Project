import numpy as np
import random  # Make sure to import random for the mutation method
from constants import MAX_SPEED, MAX_ANGULAR_VELOCITY, LEARNING_RATE  # Ensure these constants are defined

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He initialization for weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias_hidden = np.random.randn(hidden_size) * 0.01  # Small random biases
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias_output = np.random.randn(output_size) * 0.01

    def forward(self, inputs):
        """Forward pass to compute the output actions (speed and angular velocity)."""
        self.hidden = relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output  # Raw output
        
        # Compute speed and angular velocity
        speed = np.clip(sigmoid(self.output[0]) * MAX_SPEED, 0, MAX_SPEED)
        angular_velocity = np.clip(tanh(self.output[1]) * MAX_ANGULAR_VELOCITY, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        return speed, angular_velocity

    def backward(self, input_data, td_error, action_taken):
        """Backpropagation to update the neural network based on TD error."""
        output_error = np.zeros_like(self.output)
        output_error[action_taken] = td_error  # Apply TD error only to the chosen action
        hidden_error = output_error.dot(self.weights_hidden_output.T) * relu_derivative(self.hidden)

        input_data = input_data.reshape(1, -1)
        output_error = output_error.reshape(1, -1)
        hidden_error = hidden_error.reshape(1, -1)
        self.hidden = self.hidden.reshape(1, -1)

        self.weights_hidden_output += LEARNING_RATE * self.hidden.T.dot(output_error)
        self.bias_output += LEARNING_RATE * output_error.flatten()

        self.weights_input_hidden += LEARNING_RATE * input_data.T.dot(hidden_error)
        self.bias_hidden += LEARNING_RATE * hidden_error.flatten()

    def crossover(self, other, crossover_rate=0.5):
        """Perform crossover between two neural networks."""
        child_nn = NeuralNetwork(self.weights_input_hidden.shape[0], self.weights_input_hidden.shape[1], self.weights_hidden_output.shape[1])

        for matrix_name in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
            parent1_matrix = getattr(self, matrix_name)
            parent2_matrix = getattr(other, matrix_name)

            crossover_weight = random.uniform(0, 1)
            new_matrix = crossover_weight * parent1_matrix + (1 - crossover_weight) * parent2_matrix
            setattr(child_nn, matrix_name, new_matrix)

        return child_nn

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2, structural_mutation_rate=0.05):
        """Apply small random mutations to the network's weights and structure."""
        # Weight Perturbation with Gaussian Noise
        if random.random() < mutation_rate:
            self.weights_input_hidden += np.random.randn(*self.weights_input_hidden.shape) * mutation_strength
            self.weights_hidden_output += np.random.randn(*self.weights_hidden_output.shape) * mutation_strength
            self.bias_hidden += np.random.randn(*self.bias_hidden.shape) * mutation_strength
            self.bias_output += np.random.randn(*self.bias_output.shape) * mutation_strength

        # Structural Mutation (Add/Remove Neurons or Connections)
        if random.random() < structural_mutation_rate:
            # Example: Add a neuron to the hidden layer
            if random.random() < 0.5:
                new_neuron = np.random.randn(self.weights_input_hidden.shape[0])  # New weight from input to the new neuron
                self.weights_input_hidden = np.hstack((self.weights_input_hidden, new_neuron.reshape(-1, 1)))  # Add new weight
                self.bias_hidden = np.append(self.bias_hidden, np.random.randn())  # Add bias for new neuron

                # Also update the output weights to match the new hidden size
                self.weights_hidden_output = np.hstack((self.weights_hidden_output, np.random.randn(self.weights_hidden_output.shape[0], 1)))  # Add new output connection

            # Example: Remove a neuron from the hidden layer
            else:
                if self.weights_input_hidden.shape[1] > 1:  # Ensure at least one neuron remains
                    idx_to_remove = random.randint(0, self.weights_input_hidden.shape[1] - 1)
                    self.weights_input_hidden = np.delete(self.weights_input_hidden, idx_to_remove, axis=1)  # Remove column
                    self.bias_hidden = np.delete(self.bias_hidden, idx_to_remove)  # Remove corresponding bias

                    # Also update the output weights to match the new hidden size
                    self.weights_hidden_output = np.delete(self.weights_hidden_output, idx_to_remove, axis=0)  # Remove corresponding output connection

                    # If no hidden neurons remain, reset the output weights to prevent errors
                    if self.weights_input_hidden.shape[1] == 0:
                        self.weights_hidden_output = np.zeros((0, self.weights_hidden_output.shape[1]))

                    # Update the bias
                    self.bias_hidden = np.zeros(self.weights_input_hidden.shape[1])

                # Example: Add/remove a connection between hidden and output layer
                else:
                    connection_to_remove = random.randint(0, self.weights_hidden_output.shape[0] - 1)
                    self.weights_hidden_output = np.delete(self.weights_hidden_output, connection_to_remove, axis=0)  # Remove weight

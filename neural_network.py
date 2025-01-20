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
        """
        Forward pass to compute the output actions (speed and angular velocity).

        Parameters:
        - inputs: The input state (e.g., ray distances).

        Returns:
        - speed: The computed speed (scaled and clipped).
        - angular_velocity: The computed angular velocity (scaled and clipped).
        """
        self.hidden = relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output  # Raw output
        
        # Compute speed and angular velocity
        speed = np.clip(sigmoid(self.output[0]) * MAX_SPEED, 0, MAX_SPEED)
        angular_velocity = np.clip(tanh(self.output[1]) * MAX_ANGULAR_VELOCITY, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        return speed, angular_velocity

    def backward(self, input_data, td_error, action_taken):
        """
        Backpropagation to update the neural network based on TD error.

        Parameters:
        - input_data: The input state (e.g., ray distances).
        - td_error: Temporal Difference (TD) error for the chosen action.
        - action_taken: Index of the action that was taken (0 for speed, 1 for angular velocity).
        """
        # Compute the output error
        output_error = np.zeros_like(self.output)  # Zero gradient for all actions
        output_error[action_taken] = td_error  # Apply TD error only to the chosen action

        # Compute the hidden layer error
        hidden_error = output_error.dot(self.weights_hidden_output.T) * relu_derivative(self.hidden)

        # Reshape for matrix operations
        input_data = input_data.reshape(1, -1)
        output_error = output_error.reshape(1, -1)
        hidden_error = hidden_error.reshape(1, -1)
        self.hidden = self.hidden.reshape(1, -1)

        # Update weights and biases using gradient descent
        self.weights_hidden_output += LEARNING_RATE * self.hidden.T.dot(output_error)
        self.bias_output += LEARNING_RATE * output_error.flatten()

        self.weights_input_hidden += LEARNING_RATE * input_data.T.dot(hidden_error)
        self.bias_hidden += LEARNING_RATE * hidden_error.flatten()

    def crossover(self, other):
        """Perform crossover between two neural networks."""
        # Create a child neural network with the same structure
        child_nn = NeuralNetwork(self.weights_input_hidden.shape[0], self.weights_input_hidden.shape[1], self.weights_hidden_output.shape[1])
        
        # Combine the weights of both networks
        child_nn.weights_input_hidden = (self.weights_input_hidden + other.weights_input_hidden) / 2
        child_nn.weights_hidden_output = (self.weights_hidden_output + other.weights_hidden_output) / 2
        child_nn.bias_hidden = (self.bias_hidden + other.bias_hidden) / 2
        child_nn.bias_output = (self.bias_output + other.bias_output) / 2
        
        return child_nn
    
    def mutate(self):
        """Apply small random mutations to the network's weights."""
        mutation_rate = 0.1  # Percentage of weights to mutate
        mutation_strength = 0.2  # How much to change the weights by
        
        if random.random() < mutation_rate:
            self.weights_input_hidden += np.random.randn(*self.weights_input_hidden.shape) * mutation_strength
            self.weights_hidden_output += np.random.randn(*self.weights_hidden_output.shape) * mutation_strength
            self.bias_hidden += np.random.randn(*self.bias_hidden.shape) * mutation_strength
            self.bias_output += np.random.randn(*self.bias_output.shape) * mutation_strength

# Example usage
if __name__ == "__main__":
    # Define network dimensions
    input_size = 10  # Example input size (e.g., 10 rays)
    hidden_size = 8  # Number of hidden neurons
    output_size = 2  # Outputs: speed and angular velocity

    # Instantiate the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Dummy inputs and test
    dummy_input = np.random.rand(input_size)  # Random ray values between 0 and 1
    speed, angular_velocity = nn.forward(dummy_input)
    print(f"Forward Propagation Output:\nSpeed: {speed}, Angular Velocity: {angular_velocity}")

    # Example backpropagation
    td_error = 0.5  # Example TD error
    action_taken = 0  # Update for speed
    nn.backward(dummy_input, td_error, action_taken)
    print("Backward propagation complete. Weights updated.")

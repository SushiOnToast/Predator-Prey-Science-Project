import numpy as np
from constants import *  # Make sure constants like MAX_SPEED, LEARNING_RATE are defined

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
    def __init__(self, input_size, hidden_size, output_size, num_actions):
        # He initialization for weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias_output = np.zeros(output_size)
        self.Q_values = np.zeros((num_actions,))  # Q-values for actions

    def forward(self, inputs):
        self.hidden = relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output  # Save raw output
        
        # Use sigmoid and tanh for specific outputs
        speed = np.clip(sigmoid(self.output[0]) * MAX_SPEED, 0, MAX_SPEED)
        angular_velocity = np.clip(tanh(self.output[1]) * MAX_ANGULAR_VELOCITY, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        # print(speed, angular_velocity)
        return speed, angular_velocity

    def backward(self, input_data, td_error, action_taken):
        """
        Backpropagation for Q-learning.

        Parameters:
        - input_data: The input state (e.g., ray distances).
        - td_error: Temporal Difference (TD) error for the chosen action.
        - action_taken: Index of the action that was taken (e.g., 0 for speed, 1 for angular velocity).
        """
        # Compute the output error
        output_error = np.zeros_like(self.output)  # Zero gradient for all actions
        output_error[action_taken] = td_error  # Apply TD error only to the chosen action
        
        # Compute the error for the hidden layer
        hidden_error = output_error.dot(self.weights_hidden_output.T) * relu_derivative(self.hidden)

        # Reshape for matrix operations
        output_error = output_error.reshape(1, -1)
        self.hidden = self.hidden.reshape(1, -1)
        input_data = input_data.reshape(1, -1)
        hidden_error = hidden_error.reshape(1, -1)

        # Update weights and biases (gradient descent)
        self.weights_hidden_output += LEARNING_RATE * self.hidden.T.dot(output_error)
        self.bias_output += LEARNING_RATE * np.sum(output_error, axis=0)

        self.weights_input_hidden += LEARNING_RATE * input_data.T.dot(hidden_error)
        self.bias_hidden += LEARNING_RATE * np.sum(hidden_error, axis=0)


# Create the neural network
input_size = 10  # Example input size (10 rays)
hidden_size = 8  # Example number of hidden neurons
output_size = 2  # Speed and angular velocity
num_actions = 2  # Only speed and angular velocity

nn = NeuralNetwork(input_size, hidden_size, output_size, num_actions)

# Dummy inputs
dummy_input = np.random.rand(input_size)  # Random ray values between 0 and 1

# Test forward propagation
speed, angular_velocity = nn.forward(dummy_input)
print(f"Forward Propagation Output:\nSpeed: {speed}, Angular Velocity: {angular_velocity}")


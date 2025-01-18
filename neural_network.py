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

        return speed, angular_velocity

    def backward(self, input_data, td_error, action_taken):
        # Compute the output error
        output_error = np.zeros_like(self.output)
        output_error[action_taken] = td_error  # Apply TD error to the action taken
        
        # Compute gradients for hidden and input layers
        hidden_error = output_error.dot(self.weights_hidden_output.T) * relu_derivative(self.hidden)
        
        output_error = output_error.reshape(1, -1)
        self.hidden = self.hidden.reshape(1, -1)
        input_data = input_data.reshape(1, -1)
        hidden_error = hidden_error.reshape(1, -1)

        # Update weights and biases
        self.weights_hidden_output += self.hidden.T.dot(output_error) * LEARNING_RATE
        self.bias_output += np.sum(output_error, axis=0) * LEARNING_RATE

        self.weights_input_hidden += input_data.T.dot(hidden_error) * LEARNING_RATE
        self.bias_hidden += np.sum(hidden_error, axis=0) * LEARNING_RATE

# # Parameters for training
# input_size = 10  # Number of rays (example)
# hidden_size = 5  # Number of neurons in hidden layer
# output_size = 2  # Speed and angular velocity outputs
# num_actions = 2  # Assume 2 actions for simplicity (e.g., speed and direction)
# epochs = 5000  # Number of epochs to train
# epsilon = 0.1  # Exploration rate for epsilon-greedy
# LEARNING_RATE = 0.01  # Learning rate

# # Create the neural network
# nn = NeuralNetwork(input_size, hidden_size, output_size, num_actions)

# def reward_function(agent_position, target_position, speed, angular_velocity):
#     # Compute the distance to the target
#     distance_to_target = np.linalg.norm(target_position - agent_position)

#     # Reward for getting closer to the target (negative distance means closer)
#     reward = -distance_to_target

#     # If speed is very low (inefficient movement), penalize
#     if speed < 0.1:  # Small speed threshold (adjustable)
#         reward -= 1  # Penalty for low speed

#     # Penalize for moving in the wrong direction (not toward the target)
#     # Use dot product to measure alignment with the target direction
#     target_direction = target_position - agent_position
#     target_direction = target_direction / np.linalg.norm(target_direction)  # Normalize
#     movement_direction = np.array([np.cos(angular_velocity), np.sin(angular_velocity)])

#     alignment = np.dot(target_direction, movement_direction)
#     if alignment < 0:  # If alignment is negative, we are moving away
#         reward -= 0.5  # Penalize for moving away

#     # If the agent moves efficiently, reward it
#     if alignment > 0.8:  # Adjustable threshold for efficient movement
#         reward += 0.5  # Reward for efficient movement

#     return reward


# # Training loop
# for epoch in range(epochs):
#     # Generate random input (sensor data)
#     input_data = np.random.rand(input_size)

#     # Forward pass to get speed and angular velocity
#     speed, angular_velocity = nn.forward(input_data)

#     # Simulate agent's current position (you may use a real position in a more complex setup)
#     agent_position = np.array([np.random.rand(), np.random.rand()])  # Random position
#     target_position = np.array([1.0, 1.0])  # Target position

#     # Calculate the reward using the new reward function
#     reward = reward_function(agent_position, target_position, speed, angular_velocity)

#     # Select action (for simplicity, we assume we're just updating one action at a time)
#     action_taken = np.random.choice(num_actions)

#     # Temporal difference error (TD error)
#     td_error = reward - nn.Q_values[action_taken]  # Simplified TD error calculation

#     # Update Q-values based on TD error (Q-learning)
#     nn.Q_values[action_taken] += LEARNING_RATE * td_error

#     # Perform backpropagation
#     nn.backward(input_data, td_error, action_taken)

#     # Print the progress every 100 epochs to check error decreasing
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: TD Error = {td_error:.4f}, Reward = {reward:.4f}")

# # After training, print the final Q-values
# print(f"Final Q-values: {nn.Q_values}")

import numpy as np

# Activation function and derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability improvement
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He initialization for weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, inputs):
        """
        Perform a forward pass through the network.
        :param inputs: Input data (NumPy array).
        :return: Network outputs.
        """
        self.hidden = relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = softmax(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def mutate(self, mutation_rate=0.1):
        """
        Apply random mutations to weights and biases.
        :param mutation_rate: Probability of mutation.
        """
        self.weights_input_hidden += mutation_rate * np.random.randn(*self.weights_input_hidden.shape)
        self.bias_hidden += mutation_rate * np.random.randn(*self.bias_hidden.shape)
        self.weights_hidden_output += mutation_rate * np.random.randn(*self.weights_hidden_output.shape)
        self.bias_output += mutation_rate * np.random.randn(*self.bias_output.shape)

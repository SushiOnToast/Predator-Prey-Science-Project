import numpy as np

# define activation function (relU)
def relu(x):
    return np.maximum(0, x)

# define the derivative of reLU for bacpropogation
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.bias_hidden = np.random.rand(hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_output = np.random.rand(output_size) - 0.5

    def forward(self, inputs):
        """
        perform a forward pass through thte networ
        :param inputs: Input data (Id NumPy array)
        :return: netowr outputs
        """
        self.hidden = relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        return self.output

    def mutate(self, mutation_rate=0.1):
        """
        apply random mutations to weights and biases
        :param mutation_Rate: probability of mutation
        """
        self.weights_input_hidden += mutation_rate * np.random.randn(*self.weights_input_hidden.shape)
        self.bias_hidden += mutation_rate * np.random.randn(*self.bias_hidden.shape)
        self.weights_hidden_output += mutation_rate * np.random.randn(*self.weights_hidden_output.shape)
        self.bias_output += mutation_rate * np.random.randn(*self.bias_output.shape)
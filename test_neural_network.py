import unittest
import numpy as np
from neural_network import NeuralNetwork  # Replace with your actual module name

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """Set up the neural network before each test."""
        self.input_size = 10
        self.hidden_size = 8
        self.output_size = 2
        self.num_actions = 2
        self.network = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, self.num_actions)
        self.max_speed = 10  # Assuming MAX_SPEED is 10
        self.max_angular_velocity = 5  # Assuming MAX_ANGULAR_VELOCITY is 5

    def test_forward_zeros(self):
        """Test the forward pass with all zeros as input."""
        inputs = np.zeros(self.input_size)
        speed, angular_velocity = self.network.forward(inputs)
        self.assertGreaterEqual(speed, 0)
        self.assertLessEqual(speed, self.max_speed)
        self.assertGreaterEqual(angular_velocity, -self.max_angular_velocity)
        self.assertLessEqual(angular_velocity, self.max_angular_velocity)

    def test_forward_random(self):
        """Test the forward pass with random inputs."""
        inputs = np.random.uniform(-1, 1, self.input_size)
        speed, angular_velocity = self.network.forward(inputs)
        self.assertGreaterEqual(speed, 0)
        self.assertLessEqual(speed, self.max_speed)
        self.assertGreaterEqual(angular_velocity, -self.max_angular_velocity)
        self.assertLessEqual(angular_velocity, self.max_angular_velocity)

    def test_softmax_output_sum(self):
        """Ensure the softmax output sums to 1."""
        inputs = np.random.uniform(-1, 1, self.input_size)
        self.network.forward(inputs)
        softmax_output_sum = np.sum(self.network.output)
        self.assertAlmostEqual(softmax_output_sum, 1.0, places=5)

    def test_hidden_layer_activation(self):
        """Test the hidden layer activations are non-negative (ReLU)."""
        inputs = np.random.uniform(-1, 1, self.input_size)
        self.network.forward(inputs)
        hidden_layer_output = self.network.hidden
        self.assertTrue(np.all(hidden_layer_output >= 0))

if __name__ == "__main__":
    unittest.main()

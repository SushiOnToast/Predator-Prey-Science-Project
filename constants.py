# window screen dimensions
WIDTH, HEIGHT = 800, 600

# colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Simulation parameters
FPS = 60
NUM_AGENTS = 10
DIGESTION_COOLDOWN_TIME = 50
ENERGY = 100
MAX_ANGULAR_VELOCITY = 0.2
MAX_SPEED = 5.0

PREDATOR_FOV = 60
PREY_FOV = 270
NUM_RAYS = 10

PREY_FITNESS_THRESHOLD = 10000
PREDATOR_FITNESS_THRESHOLD = 3
INPUT_SIZE = NUM_RAYS  # Number of rays (input features)
HIDDEN_SIZE = 8  # Number of neurons in the hidden layer (can experiment with this)
OUTPUT_SIZE = 2  # Speed and direction (outputs)

LEARNING_RATE = 0.001
STEPS_PER_GENERATION = 1000  # Number of steps before incrementing generation

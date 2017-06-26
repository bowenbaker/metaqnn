output_states = 10                                                                          # Number of Classes
image_size = 28                                                                             # Size of images before they enter network (or smallest dimension of image)

layer_limit = 12                                                                            # Max number of layers

# Transition Options
possible_conv_depths = [64, 128, 256, 512]                                                  # Choices for number of filters in a convolutional layer
possible_conv_sizes = [1,3,5]                                                               # Choices for kernel size (square)
possible_pool_sizes = [[5,3], [3,2], [2,2]]                                                 # Choices for [kernel size, stride] for a max pooling layer
max_fc = 2                                                                                  # Maximum number of fully connected layers (excluding final FC layer for softmax output)
possible_fc_sizes = [i for i in [512, 256, 128] if i >= output_states]                      # Possible number of neurons in a fully connected layer

allow_initial_pooling = False                                                               # Allow pooling as the first layer
init_utility = 0.5                                                                          # Set this to around the performance of an average model. It is better to undershoot this
allow_consecutive_pooling = False                                                           # Allow a pooling layer to follow a pooling layer

conv_padding = 'SAME'                                                                       # set to 'SAME' (recommended) to pad convolutions so input and output dimension are the same
                                                                                            # set to 'VALID' to not pad convolutions

batch_norm = False                                                                          # Add batchnorm after convolution before activation


# Epislon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
epsilon_schedule = [[1.0, 1500],
                    [0.9, 100],
                    [0.8, 100],
                    [0.7, 100],
                    [0.6, 150],
                    [0.5, 150],
                    [0.4, 150],
                    [0.3, 150],
                    [0.2, 150],
                    [0.1, 150]]

# Q-Learning Hyper parameters
learning_rate = 0.01                                                                        # Q Learning learning rate (alpha from Equation 3)
discount_factor = 1.0                                                                       # Q Learning discount factor (gamma from Equation 3)
replay_number = 128                                                                         # Number trajectories to sample for replay at each iteration

# Set up the representation size buckets (see paper Appendix Section B)
def image_size_bucket(image_size):
    if image_size > 7:
        return 8
    elif image_size > 3:
        return 4
    else:
        return 1

# Condition to allow a transition to fully connected layer based on the current representation size
def allow_fully_connected(representation_size):
    return representation_size <= 4




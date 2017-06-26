import state_space_parameters as ssp

MODEL_NAME = 'svhn_auto'

# Number of output neurons
NUM_CLASSES = 10

#Final Image Size
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

#Batch Queue parameters
TRAIN_BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 68257
NUM_ITER_PER_EPOCH_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 26032
MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = .1

TEST_INTERVAL_EPOCHS = 1
MAX_STEPS = 20 * NUM_ITER_PER_EPOCH_TRAIN # Max number of batches


#Training Parameters
OPTIMIZER = 'Adam'
MOMENTUM = 0.9                                                # Optimizer to use
MOVING_AVERAGE_DECAY = 0.9999                                       # The decay to use for the moving average.
WEIGHT_DECAY_RATE = 0.0005                                          # Weight decay factor   


# Learning Rate
#INITIAL_LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001]               # Initial learning rate.
INITIAL_LEARNING_RATES = [0.001 * (0.4**i) for i in range(5)]
ACC_THRESHOLD = 0.27
LEARNING_RATE_DECAY_FACTOR = 0.2                                    # Learning rate decay factor.
NUM_EPOCHS_PER_DECAY = 5                                            # Epochs after which learning rate decays.
LR_POLICY = "step"

# Print every?
DISPLAY_ITER = 100 # Number of batches to print between
SAVE_EPOCHS = 1 # Number of epochs between snapshots

ENABLE_IMAGE_DATA = True
#Train files
TRAIN_FILE = '/home/bowen/work/deepbio/datasets/svhn/train.csv.shuff'
VAL_FILE = '/home/bowen/work/deepbio/datasets/svhn/val.csv'
TEST_FILE = ''
# For training logs and checkpoints
LOG_DIR = 'logs/' + MODEL_NAME
CHECKPOINT_DIR = 'trained_models/' + MODEL_NAME

output_states = 10 # Number of Classes
image_size = 32 # Size of images before they enter network (or smallest dimension of image)

layer_limit = 12 # Max number of layers
recursion_depth_limit = 0 # Depreciated (was for splits)
possible_split_widths = range(2,6) # Depreciated (was for splits)

# Transition Options
possible_conv_depths = [64, 128, 256, 512]
possible_conv_sizes = [1,3,5]
possible_pool_sizes = [[5,3], [3,2], [2,2]]
max_fc = 2
possible_fc_sizes = [i for i in [512, 256, 128] if i >= output_states]

number_image_size_buckets = 4

conv_padding = 'SAME'

# FILES TO GENERATE
files = ['rogue_nets.csv',
         'herm_nets.csv',
         'luna_nets.csv',
         'luna_nets2.csv',
         'phoenix_nets.csv',
         'phoenix_nets2.csv',
         'uhura_nets.csv',
         'raven_nets.csv',
         'davi_nets.csv']


def image_size_bucket(image_size):
    if image_size > 7:
        return 8
    elif image_size > 3:
        return 4
    else:
        return 1

def allow_fully_connected(image_bucket):
    return image_bucket <= 4

def get_transition_space_size():
    n_conv = len(possible_conv_sizes) * len(possible_conv_depths)
    n_pool = len(possible_pool_sizes)
    n_fc = len(possible_fc_sizes) * max_fc
    n_nin = len(possible_conv_depths)
    n_gap = 1
    n_term = 1

    n_conv_trans = n_conv * (n_conv + n_pool + n_fc + n_nin + n_gap + n_term)
    n_pool_trans = n_pool * (n_conv + n_fc + n_nin + n_gap + n_term)
    n_nin_trans = n_nin * (n_conv + n_pool + n_fc + n_nin + n_gap + n_term)
    n_fc_trans = n_fc * (n_fc + n_term)
    n_gap_trans = n_term


    total_trans = (n_conv_trans + n_pool_trans + n_nin_trans + n_fc_trans + n_gap_trans) * layer_limit * number_image_size_buckets

    return total_trans

print get_transition_space_size()





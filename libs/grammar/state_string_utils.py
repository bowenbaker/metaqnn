import math
import numpy as np
from operator import itemgetter
import cnn
import state_enumerator as se

class StateStringUtils:
    ''' Contains all functions dealing with converting nets to net strings
        and net strings to state lists.
    '''
    def __init__(self, state_space_parameters):
        self.image_size = state_space_parameters.image_size
        self.output_number = state_space_parameters.output_states
        self.enum = se.StateEnumerator(state_space_parameters)

    def add_drop_out_states(self, state_list):
        ''' Add drop out every 2 layers and after each fully connected layer
        Sets dropout rate to be between 0 and 0.5 at a linear rate
        '''
        new_state_list = []
        number_fc = len([state for state in state_list if state.layer_type == 'fc'])
        number_gap = len([state for state in state_list if state.layer_type == 'gap'])
        number_drop_layers = (len(state_list) - number_gap - number_fc)/2 + number_fc
        drop_number = 1
        for i in range(len(state_list)):
            new_state_list.append(state_list[i])
            if ((((i+1) % 2 == 0 and i != 0) or state_list[i].layer_type == 'fc')
                and state_list[i].terminate != 1
                and state_list[i].layer_type != 'gap'
                and drop_number <= number_drop_layers):
                drop_state = state_list[i].copy()
                drop_state.filter_depth = drop_number
                drop_state.fc_size = number_drop_layers
                drop_state.layer_type = 'dropout'
                drop_number += 1
                new_state_list.append(drop_state)

        return new_state_list

    def remove_drop_out_states(self, state_list):
        new_state_list = []
        for state in state_list:
            if state.layer_type != 'dropout':
                new_state_list.append(state)
        return new_state_list


    def state_list_to_string(self, state_list):
        '''Convert the list of strings to a string we can train from according to the grammar'''
        out_string = ''
        strings = []
        i = 0
        while i < len(state_list):
            state = state_list[i]
            if self.state_to_string(state):
                strings.append(self.state_to_string(state))
            i += 1
        return str('[' + ', '.join(strings) + ']')

    def state_to_string(self, state):
        ''' Returns the string asociated with state.
        '''
        if state.terminate == 1:
            return 'SM(%i)' % (self.output_number)
        elif state.layer_type == 'conv':
            return 'C(%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride)
        elif state.layer_type == 'gap':
            return 'GAP(%i)' % (self.output_number)
        elif state.layer_type == 'pool':
            return 'P(%i,%i)' % (state.filter_size, state.stride)
        elif state.layer_type == 'fc':
            return 'FC(%i)' % (state.fc_size)
        elif state.layer_type == 'dropout':
            return 'D(%i,%i)' % (state.filter_depth, state.fc_size) ##SUPER BAD i am using fc_size and filter depth -- should fix later
        return None

    def convert_model_string_to_states(self, parsed_list, start_state=None):
        '''Takes a parsed model string and returns a recursive list of states.'''

        states = [start_state] if start_state else [se.State('start', 0, 1, 0, 0, self.image_size, 0, 0)]

        for layer in parsed_list:
            if layer[0] == 'conv':
                states.append(se.State(layer_type='conv',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=layer[1],
                                    filter_size=layer[2],
                                    stride=layer[3],
                                    image_size=states[-1].image_size,
                                    fc_size=0,
                                    terminate=0))
            elif layer[0] == 'gap':
                states.append(se.State(layer_type='gap',
                                        layer_depth=states[-1].layer_depth + 1,
                                        filter_depth=0,
                                        filter_size=0,
                                        stride=0,
                                        image_size=1,
                                        fc_size=0,
                                        terminate=0))
            elif layer[0] == 'pool':
                states.append(se.State(layer_type='pool',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=0,
                                    filter_size=layer[1],
                                    stride=layer[2],
                                    image_size=self.enum._calc_new_image_size(states[-1].image_size, layer[1], layer[2]),
                                    fc_size=0,
                                    terminate=0))
            elif layer[0] == 'fc':
                states.append(se.State(layer_type='fc',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=len([state for state in states if state.layer_type == 'fc']),
                                    filter_size=0,
                                    stride=0,
                                    image_size=0,
                                    fc_size=layer[1],
                                    terminate=0))
            elif layer[0] == 'dropout':
                states.append(se.State(layer_type='dropout',
                                        layer_depth=states[-1].layer_depth,
                                        filter_depth=layer[1],
                                        filter_size=0,
                                        stride=0,
                                        image_size=states[-1].image_size,
                                        fc_size=layer[2],
                                        terminate=0))
            elif layer[0] == 'softmax':
                termination_state = states[-1].copy() if states[-1].layer_type != 'dropout' else states[-2].copy()
                termination_state.terminate=1
                termination_state.layer_depth += 1
                states.append(termination_state)

        return states




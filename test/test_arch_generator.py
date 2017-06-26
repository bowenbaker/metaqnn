from __future__ import absolute_import
import os
import sys
sys.path.append("../")
sys.path.append("../../../")
import q_learner as ag
from state_enumerator import State, StateEnumerator
from state_string_utils import StateStringUtils
from generate_net_csvs import GenerationDistributor


#################################################################
# SHOULD BE RUN FROM TEST DIR
#################################################################


import cnn

import hyper_parameters
import state_space_parameters

gen = ag.QLearner(state_space_parameters, 1)
enum = StateEnumerator(state_space_parameters)

#Test State Copy creates 2 versions
state1 = State('start', 0, 1, 0, 0, 32, 1, 0, 0, 0)
state2 = state1.copy()
state2.layer_type='conv'
state1.image_size=25
assert(state1.layer_type=='start')
assert(state2.image_size==32)

#Test updating q-values
q_gen = ag.QLearner(state_space_parameters, 1)
start_state = State('start', 0, 1, 0, 0, 32, 0, 0, 0, 0)
gen.qstore.q = enum.enumerate_state(start_state, gen.qstore.q)
to_state = State(state_list=gen.qstore.q[start_state.as_tuple()]['to_states'][0])
gen.qstore.q = enum.enumerate_state(to_state, gen.qstore.q)
gen.qstore.q[to_state.as_tuple()]['utilities'][0] = 0.6
gen._update_q_value(start_state,to_state,0.4)
assert(gen.qstore.q[start_state.as_tuple()]['utilities'][0] == 0.5 + gen.learning_rate *(0.4 + 0.6 - 0.5))


for i in range(100):
    previous_action_values = gen.qstore.q.copy()
    net, acc_best_val, acc_last_val, acc_best_test, acc_last_test = gen.generate_net()
    stringutils = StateStringUtils(state_space_parameters)

    print net
    net_lists = cnn.parse('net', net)
    print '---------------------'
    print 'Net String'
    print net
    print '----------------------'
    assert(net == stringutils.state_list_to_string(stringutils.convert_model_string_to_states(net_lists)))

    for key in previous_action_values.keys():
        old_to_state = [state for state in previous_action_values[key]['to_states']]
        new_to_state = [state for state in gen.qstore.q[key]['to_states']]

        for i in range(len(old_to_state)):
            assert(old_to_state[i] in new_to_state)
        assert(previous_action_values[key]['utilities'] == gen.qstore.q[key]['utilities'])
    

print 'Transition probabilities arent being mutated'

print os.listdir('../../../string_models/mnist')
gd = GenerationDistributor('../../../string_models/cifar10_3', state_space_parameters, 1)
ql = gd.load_qlearner()
ql._reset_for_new_walk()
#print ql.qstore.q.keys()
#print [state for state in ql.qstore.q[State('start', 0, 1, 0, 0, 28, 0, 0, 0, 0).as_tuple()][0]]
#print ql.qstore.q[State('start', 0, 1, 0, 0, 28, 0, 0, 0, 0).as_tuple()][1]
for key in ql.qstore.q.keys():
    if key[0] == 'start':
        print key
max_v = max(ql.qstore.q[State('start', 0, 1, 0, 0, 32, 0, 0, 0, 0).as_tuple()]['utilities'])
print max_v
#print ql.qstore.q[State('start', 0, 1, 0, 0, 28, 0, 0, 0, 0).as_tuple()][0][ql.qstore.q[State('start', 0, 1, 0, 0, 28, 0, 0, 0, 0).as_tuple()][1].index(max_v)].as_tuple()

# for depth in range(15):
#     print ql.generate_optimal_net_fixed_depth(depth)


# State space size
states = 0
layer_multiplier = enum.layer_limit * len(enum.possible_split_widths) * (enum.recursion_depth_limit+1)
conv_states = len(enum.possible_conv_depths) * len(enum.possible_conv_sizes) * 32
pad_states = len(enum.possible_pool_sizes) * 32
fc_states = len(enum.possible_fc_sizes) * 3
gap_states = 1
nin_states = len(enum.possible_conv_depths) * 32
bn_states = 1
total_states = (conv_states + pad_states + fc_states + gap_states + nin_states + bn_states)
total_states *= layer_multiplier
total_states *= 2 # for termination states


print 'There are %i conv states' % (conv_states * layer_multiplier)
print 'There are %i pad states' % (pad_states * layer_multiplier)
print 'There are %i total states' % total_states

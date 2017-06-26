from libs.grammar.q_learner import QLearner, QValues
from libs.grammar import state_enumerator as se
import needed_for_testing.hyper_parameters as hp
import needed_for_testing.state_space_parameters as ssp

from test_base import Test

import pandas as pd
import numpy as np

class TestQLearner(Test):
    def __init__(self):
        self.start_state = se.State('start', 0, 1, 0, 0, ssp.image_size, 0, 0, 0, 0)
        self.q_path = 'needed_for_testing/q_values.csv'
        self.se = se.StateEnumerator(ssp)


    def test_epsilon0_generation(self):
        test = True

        qstore = QValues()
        qstore.load_q_values(self.q_path)
        optimal_states = [self.start_state.copy()]
        bucketed_state = self.se.bucket_state(self.start_state)
        while bucketed_state.as_tuple() in qstore.q and  len(qstore.q[bucketed_state.as_tuple()]['utilities']):
            next_action_index = np.random.randint(len(qstore.q[bucketed_state.as_tuple()]['utilities']))
            qstore.q[bucketed_state.as_tuple()]['utilities'][next_action_index] = 100000000000.0
            next_state = self.se.state_action_transition(optimal_states[-1], 
                                                         se.State(state_list=qstore.q[bucketed_state.as_tuple()]['actions'][next_action_index]))
            optimal_states.append(next_state)
            bucketed_state = self.se.bucket_state(optimal_states[-1])

        ql = QLearner(ssp, 0.0, qstore=qstore)
        states = ql._run_agent()

        states = [state.as_tuple() for state in states]
        optimal_states = [self.se.bucket_state(state).as_tuple() for state in optimal_states[1:]]

        if len(states) != len(optimal_states):
            print states
            print optimal_states
            print 'Wrong Length'
            return False

        for i in range(len(states)):
            test = test and states[i] == optimal_states[i]

        return test

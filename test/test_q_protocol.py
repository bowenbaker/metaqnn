from libs.grammar.q_learner import QLearner, QValues
from libs.grammar import state_enumerator as se
import libs.grammar.q_protocol as q_protocol
import needed_for_testing.hyper_parameters as hp
import needed_for_testing.state_space_parameters as ssp

from test_base import Test

import pandas as pd
import numpy as np

class TestQProtocol(Test):
    def __init__(self):
        pass

    def test_construct_login(self):
        msg = q_protocol.construct_login_message('luna')
        out = q_protocol.parse_message(msg)

        return (out['sender'] == 'luna'
                and out['type'] == 'login')

    def test_construct_new_net(self):
        test = True
        msg = q_protocol.construct_new_net_message('luna', '[C(120,1,1), P(5,1), GAP(10), SM(10)]', 0.7, 2000)
        out = q_protocol.parse_message(msg)

        test = (out['sender'] == 'luna'
                and out['type'] == 'new_net'
                and out['net_string'] == '[C(120,1,1), P(5,1), GAP(10), SM(10)]'
                and float(out['epsilon']) == 0.7
                and int(out['iteration_number']) == 2000)

        if not test:
            print msg
            print out

        return test

    def test_construct_net_trained(self):
        test = True

        msg = q_protocol.construct_net_trained_message('luna', '[C(120,1,1), P(5,1), GAP(10), SM(10)]', 0.1, 1000, 0.2, 2000, 0.7, 3000)
        out = q_protocol.parse_message(msg)

        test = (out['sender'] == 'luna'
                and out['type'] == 'net_trained'
                and out['net_string'] == '[C(120,1,1), P(5,1), GAP(10), SM(10)]'
                and float(out['acc_best_val']) == 0.1
                and int(out['iter_best_val']) == 1000
                and float(out['acc_last_val']) == 0.2
                and int(out['iter_last_val']) == 2000
                and float(out['epsilon']) == 0.7
                and int(out['iteration_number']) == 3000)


        if not test:
            print msg
            print out

        return test







from libs.grammar import state_enumerator as se
import needed_for_testing.hyper_parameters as hp
import needed_for_testing.state_space_parameters as ssp

from test_base import Test

import pandas as pd
import numpy as np

class TestStateEnumerator(Test):
    def __init__(self):
        self.se = se.StateEnumerator(ssp)


    # Test it properly buckets and doesn't mutate
    def test_bucket_state(self):
        test = True

        cases = [(se.State('start', 0, 1, 0, 0, 30, 0, 0, 0, 0), 8),
                 (se.State('start', 0, 1, 0, 0, 7, 0, 0, 0, 0), 4),
                 (se.State('start', 0, 1, 0, 0, 3, 0, 0, 0, 0), 1)
                ]
        for case in cases:
            test = test and self.se.bucket_state(case[0]).image_size == case[1]
            test = test and case[0].image_size != case[1]

        return test

    def test_bucket_state_tuple(self):
        test = True

        cases = [(se.State('start', 0, 1, 0, 0, 30, 0, 0, 0, 0).as_tuple(), 8),
                 (se.State('start', 0, 1, 0, 0, 7, 0, 0, 0, 0).as_tuple(), 4),
                 (se.State('start', 0, 1, 0, 0, 3, 0, 0, 0, 0).as_tuple(), 1)
                ]
        for case in cases:
            test = test and self.se.bucket_state_tuple(case[0])[5] == case[1]
            test = test and case[0][5] != case[1]

        return test


    # Test it properly distinguishes different state types
    def test_transition_to_action(self):
        test = True
        cases = [(se.State('start', 0, 1, 0, 0, 30, 0, 0, 0, 0), se.State('conv', 1, 1, 0, 0, 7, 0, 0, 0, 0), 30),
                 (se.State('conv', 0, 1, 0, 0, 7, 0, 0, 0, 0), se.State('fc', 1, 1, 0, 0, 0, 0, 0, 512, 0), 0),
                 (se.State('conv', 0, 1, 0, 0, 7, 0, 0, 0, 0), se.State('gap', 1, 1, 0, 0, 0, 0, 0, 0, 0), 0),
                 (se.State('start', 0, 1, 0, 0, 30, 0, 0, 0, 0), se.State('pool', 1, 1, 0, 0, 7, 0, 0, 0, 0), 30)
                ]

        for case in cases:
            test = test and self.se.transition_to_action(case[0], case[1]).image_size == case[2]

        return test



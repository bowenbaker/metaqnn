from test_q_values import TestQValues
from test_state_enumerator import TestStateEnumerator
from test_q_learner import TestQLearner
from test_q_protocol import TestQProtocol


test_classes = [TestQProtocol(), TestQValues(), TestStateEnumerator(), TestQLearner()]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def test_all():
    for test_class in test_classes:
        print bcolors.HEADER + 'Testing: ' + test_class.__class__.__name__ + bcolors.ENDC
        for fn in test_class.get_all_time_fns():
            succeeded = fn()
            status = bcolors.OKGREEN + 'SUCCEEDED' + bcolors.ENDC if succeeded else bcolors.FAIL + 'FAILED' + bcolors.ENDC
            print bcolors.OKBLUE + fn.__name__ + ': ' + bcolors.ENDC + status



test_all()
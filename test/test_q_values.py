from libs.grammar.q_learner import QValues
from test_base import Test

import pandas as pd
import numpy as np

class TestQValues(Test):
    def __init__(self):
        self.q_csv_path = 'needed_for_testing/q_values.csv'

    def test_load_unload(self):
        qvals = QValues()
        qvals.load_q_values(self.q_csv_path)
        qvals.save_to_csv(self.q_csv_path + '.test')

        qcsv_before = pd.read_csv(self.q_csv_path)
        qcsv_after = pd.read_csv(self.q_csv_path + '.test')

        sort_val = [col for col in qcsv_before.columns.values]

        qcsv_before = qcsv_before.sort_values(sort_val)[sort_val]
        qcsv_after = qcsv_after.sort_values(sort_val)[sort_val]

        test = True

        for col in qcsv_before.columns.values:
            test = test and np.all(qcsv_before[col].values == qcsv_after[col].values)

        return test

    def test_load_unload2(self):
        qvals = QValues()
        qvals.load_q_values(self.q_csv_path)
        qvals.save_to_csv(self.q_csv_path + '.test')

        qcsv_before = pd.read_csv(self.q_csv_path)
        qcsv_after = pd.read_csv(self.q_csv_path + '.test')

        concat = pd.concat([qcsv_before, qcsv_after])

        return len(concat.drop_duplicates()) == len(qcsv_before)
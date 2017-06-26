import math
import numpy as np
import os
import sys

from model_gen import ModelGen
from run_caffe_command_line import check_out_of_memory
from run_caffe_command_line import get_all_accuracies
from run_caffe_command_line import get_last_epoch_snapshot
from run_caffe_command_line import get_last_test_epoch
from run_caffe_command_line import get_test_accuracies_dict
from run_caffe_command_line import run_caffe_from_snapshot
from run_caffe_command_line import run_caffe_return_accuracy

import caffe

from caffe import layers as cl
from caffe import params as P
from caffe import to_proto

class ModelExec:
    def __init__(self, model_dir, hyper_parameters, state_space_parameters):
        self.model_dir = model_dir
        self.hp = hyper_parameters
        self.ssp = state_space_parameters

    # Runs the model based on description.
    def run_one_model(self, model_descr, gpu_to_use=None):
        # Iterate through all learning rates until we get passed accuracy threshhold
        for learning_rate in self.hp.INITIAL_LEARNING_RATES:
            model_dir, solver_path = ModelGen(self.model_dir, self.hp, self.ssp).save_models(
                model_descr, learning_rate, self.hp.NUM_ITER_TO_TRY_LR)

            # Execute.
            print "Running [%s]" % solver_path
            log_file = self.get_log_fname(
                model_dir, learning_rate, self.hp.NUM_ITER_TO_TRY_LR)
            # Check log file for existence.
            acc = None
            if os.path.exists(log_file):
                acc_dict = get_test_accuracies_dict(log_file)
                # Check if we got the test accuracy for one epoch.
                if self.hp.NUM_ITER_TO_TRY_LR in acc_dict:
                    acc = acc_dict[self.hp.NUM_ITER_TO_TRY_LR]
            if not acc:
                acc, acc_dict = run_caffe_return_accuracy(solver_path, log_file, self.hp.CAFFE_ROOT, gpu_to_use=gpu_to_use)

            if check_out_of_memory(log_file):
                return {'learning_rate': learning_rate,
                        'status': 'OUT_OF_MEMORY',
                        'test_accs': {}}

            if self.hp.NUM_ITER_TO_TRY_LR not in acc_dict:
                raise Exception("Model training interrupted during first epoch. Crashing!")
            snapshot_file = self.get_snapshot_epoch_fname(model_dir, self.hp.NUM_ITER_TO_TRY_LR)

            print "Got accuracy [%f]" % acc
            if acc > self.hp.ACC_THRESHOLD:
                model_dir, solver_path = ModelGen(self.model_dir, self.hp, self.ssp).save_models(model_descr,
                                                                                       learning_rate,
                                                                                       self.hp.MAX_STEPS)
                log_file = self.get_log_fname_complete(model_dir, learning_rate)

                # Check if log file exists.
                if os.path.exists(log_file):
                    last_iter, last_epoch = get_last_test_epoch(log_file)
                    if last_iter > 0:
                        snapshot_file = self.get_snapshot_epoch_fname(model_dir, last_iter)
                        if last_iter == self.hp.MAX_STEPS:
                            test_acc_dict = get_test_accuracies_dict(log_file)
                            return {'solver_path': solver_path,
                                    'accuracy': acc,
                                    'learning_rate': learning_rate,
                                    'status': 'OLD_MODEL',
                                    'test_accs': test_acc_dict}
                        print "Will resume from [%d] using [%s]" % (last_iter, snapshot_file)

                test_acc_list = run_caffe_from_snapshot(solver_path, log_file, snapshot_file, self.hp.CAFFE_ROOT, gpu_to_use=gpu_to_use)
                test_acc_dict = get_test_accuracies_dict(log_file)
                if self.hp.MAX_STEPS in test_acc_dict:
                    return {'learning_rate': learning_rate,
                            'status': 'SUCCESS',
                            'test_accs': test_acc_dict}
                else:
                    if check_out_of_memory(log_file):
                        return {'learning_rate': learning_rate,
                                'status': 'OUT_OF_MEMORY',
                                'test_accs': {}}
                    raise Exception("Model training interrupted. Crashing!")

        return {'learning_rate': learning_rate,
                'status': 'FAIL',
                'test_accs': {}}

        # Returns log file name for given model dir when polling for various
    # learning rates.
    def get_log_fname(self, model_dir, learning_rate, max_iter):
        return '%s/%s_%f_%d.txt' % (model_dir, 'log', learning_rate, max_iter)


    # Returns log file when training for maximum iterations
    def get_log_fname_complete(self, model_dir, learning_rate):
        return '%s/%s_%f_%d.txt' % (model_dir, 'log_complete', learning_rate, self.hp.NUM_ITER_PER_EPOCH_TRAIN)


    # Returns the snapshot file name given iteration number. By default
    # returns snapshot after first epoch
    def get_snapshot_epoch_fname(self, model_dir, max_iter):
        return '%s/%s_iter_%d.solverstate' % (model_dir, 'modelsave', max_iter)

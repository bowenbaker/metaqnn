import math
import numpy as np
import os
import sys

from string_to_model import Parser

class ModelGen:

    def __init__(self, model_dir, hyper_parameters, state_space_parameters):
        self.model_dir = model_dir
        self.ssp = state_space_parameters
        self.hp = hyper_parameters

    def model_paths(self, model_descr):
        solver_path = os.path.join(self.model_dir, "solver.prototxt")
        netspec_path = os.path.join(self.model_dir, "train_net.prototxt")
        return self.model_dir, solver_path, netspec_path

  # Saves caffe specs including solver to given directories.
    def save_models(self, model_descr, learning_rate, max_iter):
        print "Creating Caffe Configs for %s" % model_descr
        model_dir, solver_path, netspec_path = self.model_paths(model_descr)
        p = Parser(self.hp, self.ssp)
        p.create_caffe_spec(model_descr, netspec_path)
        self.save_solver(solver_path, netspec_path, learning_rate, max_iter)
        return model_dir, solver_path

    # Saves the solver from hyper parameters.
    def save_solver(self, solver_path, netspec_path, learning_rate=-1, max_iter=-1):
        if learning_rate == -1:
            learning_rate = self.hp.INITIAL_LEARNING_RATES[0]
        if max_iter == -1:
            max_iter = self.hp.MAX_STEPS
        solver_proto =  'net: "%s"' % netspec_path + \
                        '\ntest_iter: %d' % (self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/self.hp.EVAL_BATCH_SIZE) + \
                        '\ntest_interval: %d' % (self.hp.TEST_INTERVAL_EPOCHS*self.hp.NUM_ITER_PER_EPOCH_TRAIN,) + \
                        '\nbase_lr: %f' % learning_rate + \
                        '\nmomentum: %f' % self.hp.MOMENTUM + \
                        '\nweight_decay: %f' % self.hp.WEIGHT_DECAY_RATE + \
                        '\ndisplay: %d' % self.hp.DISPLAY_ITER + \
                        '\nmax_iter: %d' % max_iter + \
                        '\nsnapshot: %d' % (self.hp.SAVE_EPOCHS*self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/self.hp.TRAIN_BATCH_SIZE) + \
                        '\nsnapshot_prefix: "%s/%s"' % (self.model_dir, 'modelsave') + \
                        '\nsolver_mode: %s' % 'GPU' + \
                        '\ntype: "%s"' % self.hp.OPTIMIZER + \
                        '\nlr_policy: "%s"' % self.hp.LR_POLICY + \
                        '\ngamma: %f' % self.hp.LEARNING_RATE_DECAY_FACTOR
        if self.hp.LR_POLICY == 'step':
            solver_proto += '\nstepsize: %i' % (self.hp.NUM_EPOCHS_PER_DECAY *self.hp.NUM_ITER_PER_EPOCH_TRAIN)

        else:
            for epoch, number_decays in self.hp.STEP_LIST:
                step = epoch * self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.hp.TRAIN_BATCH_SIZE
                for j in range(number_decays):
                    solver_proto += '\nstepvalue: %i' % (step + j)
            
        with open(solver_path, "w") as solver:
            solver.write(solver_proto)

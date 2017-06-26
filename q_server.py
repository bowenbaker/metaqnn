from twisted.internet import reactor, protocol
from twisted.internet.defer import DeferredLock

import libs.grammar.q_protocol as q_protocol
import libs.grammar.q_learner as q_learner

import pandas as pd
import numpy as np

import argparse
import traceback
import os
import socket
import time

class bcolors:
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class QServer(protocol.ServerFactory):
    def __init__(self,
                 list_path,
                 state_space_parameters,
                 hyper_parameters,
                 epsilon=None,
                 number_models=None):

        self.protocol = QConnection
        self.new_net_lock = DeferredLock()
        self.clients = {} # name of connection is key, each value is dict with {'connection', 'net', 'iters_sampled'}

        self.replay_columns = ['net',                   #Net String
                               'accuracy_best_val',     
                               'iter_best_val',
                               'accuracy_last_val',
                               'iter_last_val',
                               'accuracy_best_test',
                               'accuracy_last_test',
                               'ix_q_value_update',     #Iteration for q value update
                               'epsilon',               # For epsilon greedy
                               'time_finished',         # UNIX time
                               'machine_run_on']


        self.list_path = list_path

        self.replay_dictionary_path = os.path.join(list_path, 'replay_database.csv')
        self.replay_dictionary, self.q_training_step = self.load_replay()

        self.schedule_or_single = False if epsilon else True
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]
            self.number_models = state_space_parameters.epsilon_schedule[0][1]
        else:
            self.epsilon = epsilon
            self.number_models = number_models if number_models else 10000000000
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters

        self.number_q_updates_per_train = 100

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.check_reached_limit()


    def load_replay(self):
        if os.path.isfile(self.replay_dictionary_path):
            print 'Found replay dictionary'
            replay_dic = pd.read_csv(self.replay_dictionary_path)
            q_training_step = max(replay_dic.ix_q_value_update)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic, q_training_step

    def load_qlearner(self):
        # Load previous q_values
        if os.path.isfile(os.path.join(self.list_path, 'q_values.csv')):
            print 'Found q values'
            qstore = q_learner.QValues()
            qstore.load_q_values(os.path.join(self.list_path, 'q_values.csv'))
        else:
            qstore = None


        ql = q_learner.QLearner(self.state_space_parameters,
                                    self.epsilon,
                                    qstore=qstore,
                                    replay_dictionary=self.replay_dictionary)

        return ql

    def filter_replay_for_first_run(self, replay):
        ''' Order replay by iteration, then remove duplicate nets keeping the first'''
        temp = replay.sort_values(['ix_q_value_update']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['net'])

    def number_trained_unique(self, epsilon=None):
        '''Epsilon defaults to the minimum'''
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)

    def check_reached_limit(self):
        ''' Returns True if the experiment is complete
        '''
        if len(self.replay_dictionary):
            completed_current = self.number_trained_unique(self.epsilon) >= self.number_models

            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_trained_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False

                            break

                else:
                    completed_experiment = True

                return completed_experiment

            else:
                return False

    def generate_new_netork(self):
        try:
            (net,
             acc_best_val,
             iter_best_val,
             acc_last_val,
             iter_last_val,
             acc_best_test,
             acc_last_test,
             machine_run_on) = self.qlearner.generate_net()

            # We have already trained this net
            if net in self.replay_dictionary.net.values:
                self.q_training_step += 1
                self.incorporate_trained_net(net,
                                             acc_best_val,
                                             iter_best_val,
                                             acc_last_val,
                                             iter_last_val,
                                             self.epsilon,
                                             [self.q_training_step],
                                             machine_run_on)
                return self.generate_new_netork()

            # Sampled net is currently being trained on another machine
            elif net in [self.clients[key]['net'] for key in self.clients.keys()]:
                self.q_training_step += 1
                for key, value in self.clients.iteritems():
                    if value['net'] == net:
                        value['iters_sampled'].append(self.q_training_step)

                return self.generate_new_netork()

            else:
                self.q_training_step += 1
                return net, self.q_training_step

        except Exception:
            print traceback.print_exc()

    def incorporate_trained_net(self,
                                net_string,
                                acc_best_val,
                                iter_best_val,
                                acc_last_val,
                                iter_last_val,
                                epsilon,
                                iters,
                                machine_run_on):

        try:
            # If we sampled the same net many times, we should add them each into the replay database
            for train_iter in iters:
                self.replay_dictionary = pd.concat([self.replay_dictionary, pd.DataFrame({'net':[net_string],
                                                                                          'accuracy_best_val':[acc_best_val],
                                                                                          'iter_best_val': [iter_best_val],
                                                                                          'accuracy_last_val': [acc_last_val],
                                                                                          'iter_last_val': [iter_last_val],
                                                                                          'accuracy_best_test':[-1.0],
                                                                                          'accuracy_last_test': [-1.0],
                                                                                          'ix_q_value_update': [train_iter],
                                                                                          'epsilon': [epsilon],
                                                                                          'time_finished': [time.time()],
                                                                                          'machine_run_on': [machine_run_on]})])
                self.replay_dictionary.to_csv(self.replay_dictionary_path, index=False, columns=self.replay_columns)

            self.qlearner.update_replay_database(self.replay_dictionary)
            for i in range(len(iters)):
                self.qlearner.sample_replay_for_update()
            self.qlearner.save_q(self.list_path)
            print bcolors.YELLOW + 'Incorporated net from %s, acc: %f, net: %s' % (machine_run_on, acc_best_val, net_string) + bcolors.ENDC
        except Exception:
            print traceback.print_exc()


class QConnection(protocol.Protocol):
    #def generate_new_net(self):
    def __init__(self):
        pass

    def connectionLost(self, reason):
        hostname_leaving = [k for k, v in self.factory.clients.iteritems() if v['connection'] is self][0]
        print bcolors.FAIL + hostname_leaving + ' is disconnecting' + bcolors.ENDC
        self.factory.clients.pop(hostname_leaving)

    def send_new_net(self, client_name):
        completed_experiment = self.factory.new_net_lock.run(self.factory.check_reached_limit).result
        if not completed_experiment:
            net_to_run, iteration = self.factory.new_net_lock.run(self.factory.generate_new_netork).result
            print bcolors.OKBLUE + ('Sending net to %s:\n%s\nIteration %i, Epsilon %f' % (client_name, net_to_run, iteration, self.factory.epsilon)) + bcolors.ENDC       
            self.factory.clients[client_name] = {'connection': self, 'net': net_to_run, 'iters_sampled': [iteration]}
            self.transport.write(q_protocol.construct_new_net_message(socket.gethostname(), net_to_run, self.factory.epsilon, iteration))
        else:
            print 'EXPERIMENT COMPLETE!'


    def dataReceived(self, data):
        msg = q_protocol.parse_message(data)
        if msg['type'] == 'login':

            # Redundant connection
            if msg['sender'] in self.factory.clients:
                self.transport.write(q_protocol.construct_redundant_connection_message(socket.gethostname()))
                print bcolors.FAIL + msg['sender'] + ' tried to connect again. Killing second connection.' + bcolors.ENDC
                self.transport.loseConnection()

            # New connection
            else:
                print bcolors.OKGREEN + msg['sender'] + ' has connected.' + bcolors.ENDC
                self.send_new_net(msg['sender'])
                
        elif msg['type'] == 'net_trained':
            iters = self.factory.clients[msg['sender']]['iters_sampled']
            self.factory.new_net_lock.run(self.factory.incorporate_trained_net, msg['net_string'],
                                                                                float(msg['acc_best_val']),
                                                                                int(msg['iter_best_val']),
                                                                                float(msg['acc_last_val']),
                                                                                int(msg['iter_last_val']),
                                                                                float(msg['epsilon']),
                                                                                iters,
                                                                                msg['sender'])
            self.send_new_net(msg['sender'])
        elif msg['type'] == 'net_too_large':
            self.send_new_net(msg['sender'])


def main():
    parser = argparse.ArgumentParser()
    
    model_pkgpath = os.path.join(os.path.dirname(__file__),'models')
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument('model',
                        help='model package name package should have a model.py,' + 
                             'file, hyper_parameters.py file, and a log folder',
                        choices=model_choices)

    parser.add_argument('list_path')
    parser.add_argument('-eps', '--epsilon', help='For Epsilon Greedy Strategy', type=float)
    parser.add_argument('-nmt', '--number_models_to_train', type=int, 
                            help='How many models for this epsilon do you want to train.')

    args = parser.parse_args()

    _model = __import__('models.' + args.model,
                        globals(),
                        locals(),
                        ['state_space_parameters', 'hyper_parameters'], 
                        -1)

    factory = QServer(args.list_path,
                      _model.state_space_parameters,
                      _model.hyper_parameters,
                      args.epsilon,
                      args.number_models_to_train)


    reactor.listenTCP(8000,factory)
    reactor.run()

# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()    

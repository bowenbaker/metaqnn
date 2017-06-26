from twisted.internet import reactor, protocol
import libs.grammar.q_protocol as q_protocol
import time
import socket
import argparse
import os
import shutil


import pandas as pd
from libs.caffe.model_exec import ModelExec
from libs.misc.clear_trained_models import clear_redundant_logs_caffe

def rm_model_dir(base_ckpt_dir, net):
    ckpt_file_map_file = os.path.join(base_ckpt_dir, 'file_map.csv')
    if os.path.exists(ckpt_file_map_file):
        ckpt_file_map = pd.read_csv(ckpt_file_map_file)

        # Check if net exists
        if len(ckpt_file_map.net == net):
            model_dir = os.path.join(base_ckpt_dir, str(ckpt_file_map[ckpt_file_map.net == net].file_number.values[0]))
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                ckpt_file_map = ckpt_file_map[ckpt_file_map.net != net]
                ckpt_file_map.to_csv(ckpt_file_map_file, index=False)

def get_model_dir(base_ckpt_dir, net):
    if not os.path.exists(base_ckpt_dir):
        os.makedirs(base_ckpt_dir)

    ckpt_file_map_file = os.path.join(base_ckpt_dir, 'file_map.csv')
    if not os.path.exists(ckpt_file_map_file):
        pd.DataFrame(columns=['net', 'file_number']).to_csv(ckpt_file_map_file, index=False)
    
    ckpt_file_map = pd.read_csv(ckpt_file_map_file)

    # check if we already have a folder
    if sum(ckpt_file_map['net'] == net) == 0:
        next_ckpt = 1 if len(ckpt_file_map) == 0 else max(ckpt_file_map['file_number']) + 1
        ckpt_file_map = pd.concat([ckpt_file_map, pd.DataFrame({'net':[net], 'file_number':[next_ckpt]})])
        ckpt_file_map.to_csv(ckpt_file_map_file, index=False)
    else:
        next_ckpt = ckpt_file_map[ckpt_file_map['net'] == net]['file_number'].values[0]

    return os.path.join(base_ckpt_dir, str(int(next_ckpt)))

# a client protocol

class QClient(protocol.Protocol):
    """Once connected, send a message, then print the result."""
    
    def connectionMade(self):
        self.transport.write(q_protocol.construct_login_message(self.factory.clientname))
    
    def dataReceived(self, data):
        out = q_protocol.parse_message(data)
        if out['type'] == 'redundant_connection':
            print 'Redundancy in connect name'

        if out['type'] == 'new_net':
            print 'Ready to train ' + out['net_string']

            if self.factory.debug:
                time.sleep(5)
                self.transport.write(q_protocol.construct_net_trained_message(self.factory.clientname,
                                                                              out['net_string'],
                                                                              86.0,
                                                                              100,
                                                                              85.5,
                                                                              10000,
                                                                              float(out['epsilon']),
                 
                                                                              int(out['iteration_number'])))
            else:
                model_dir = get_model_dir(self.factory.hyper_parameters.CHECKPOINT_DIR, out['net_string'])

                trainer = ModelExec(model_dir, self.factory.hyper_parameters, self.factory.state_space_parameters)

                train_out = trainer.run_one_model(out['net_string'], gpu_to_use=self.factory.gpu_to_use)
                print 'OUT', train_out

                # If OUT OF MEMORY or FAIL, delete files
                if train_out['status'] in ['OUT_OF_MEMORY', 'FAIL']:
                    rm_model_dir(self.factory.hyper_parameters.CHECKPOINT_DIR, out['net_string'])

                if train_out['status'] == 'OUT_OF_MEMORY':
                    self.transport.write(q_protocol.construct_net_too_large_message(self.factory.clientname))
                else:

                    (iter_best, acc_best) = max(train_out['test_accs'].items(), key=lambda x: x[1]) if train_out['status'] != 'FAIL' \
                                                                                                else (0, 1.0/self.factory.hyper_parameters.NUM_CLASSES)
                    (iter_last, acc_last) = max(train_out['test_accs'].items(), key=lambda x: x[0]) if train_out['status'] != 'FAIL' \
                                                                                            else (0, 1.0/self.factory.hyper_parameters.NUM_CLASSES)

                    # Clear out model files
                    clear_redundant_logs_caffe(self.factory.hyper_parameters.CHECKPOINT_DIR, pd.DataFrame({'net': [out['net_string']],
                                                                                                           'iter_best_val': [iter_best],
                                                                                                           'iter_last_val': [iter_last]}))


                    self.transport.write(q_protocol.construct_net_trained_message(self.factory.clientname,
                                                                                  out['net_string'],
                                                                                  acc_best,
                                                                                  iter_best,
                                                                                  acc_last,
                                                                                  iter_last,
                                                                                  float(out['epsilon']),
                                                                                  int(out['iteration_number'])))


    
    def connectionLost(self, reason):
        print "connection lost"

class QFactory(protocol.ClientFactory):
    def __init__(self, clientname, hyper_parameters, state_space_parameters, gpu_to_use, debug):
        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters
        self.protocol = QClient
        self.clientname = clientname
        self.gpu_to_use = gpu_to_use
        self.debug = debug

    def clientConnectionFailed(self, connector, reason):
        print "Connection failed - goodbye!"
        reactor.stop()
    
    def clientConnectionLost(self, connector, reason):
        print "Connection lost - goodbye!"
        reactor.stop()

def start_reactor(clientname, hostname, model, gpu_to_use, debug):
    _model = __import__('models.' + model, globals(), locals(), ['hyper_parameters', 'state_space_parameters'], -1)

    if gpu_to_use is not None:
        print 'GPU TO USE', gpu_to_use
        _model.hyper_parameters.CHECKPOINT_DIR = _model.hyper_parameters.CHECKPOINT_DIR + str(gpu_to_use)

    f = QFactory(clientname, _model.hyper_parameters, _model.state_space_parameters, gpu_to_use, debug)
    reactor.connectTCP(hostname, 8000, f)
    reactor.run()

# this connects the protocol to a server running on port 8000
def main():
    parser = argparse.ArgumentParser()
    
    model_pkgpath = os.path.join(os.path.dirname(__file__),'models')
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument('model',
                        help='model package name package should have a model.py,' + 
                             'file, hyper_parameters.py file, and a log folder',
                        choices=model_choices)

    parser.add_argument('clientname')
    parser.add_argument('hostname')
    parser.add_argument('-gpu', '--gpu_to_use', help="GPU number to use", type=int)
    parser.add_argument('--debug', type=bool, help="True if you don't want to actually run networks and return bs", default=False)

    args = parser.parse_args()

    start_reactor(args.clientname, args.hostname, args.model, args.gpu_to_use, args.debug)

# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()

import os
import re
import sys

from collections import OrderedDict

def check_out_of_memory(log_file):
    check_str = "Check failed: error == cudaSuccess"
    check_str2 = "SIGSEGV"
    with open(log_file, 'r') as f:
        for line in f:
            if check_str in line or check_str2 in line:
                print "Caffe out of memory detected!"
                return True
    return False

# helper function to parse log file.
def parse_line_for_net_output(regex_obj, row, row_dict_list, line, iteration):
    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one.
            if row:
                row_dict_list.append(row)
            row = {'NumIters':  iteration}

        # Get the key value pairs from a line.
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)
    # Check if this row is the last for this dictionary.
    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        row_dict_list.append(row)
        row = None
    return row_dict_list, row


# MAIN FUNCTION: parses log file.
def parse_caffe_log_file(log_file):
    print "Parsing [%s]" % log_file
    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile(
            'Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile(
            'Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    iteration = -1
    train_dict_list = []
    test_dict_list = []
    snapshot_list = []
    train_row = None
    test_row = None

    with open(log_file, 'r') as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                continue
            # Parse for test/train accuracies and loss.
            train_dict_list, train_row = parse_line_for_net_output(
                    regex_train_output, train_row, train_dict_list, line, iteration)
            test_dict_list, test_row = parse_line_for_net_output(
                    regex_test_output, test_row, test_dict_list, line, iteration)
 
    if test_dict_list == [] and test_row:
        test_dict_list = [test_row]

    return train_dict_list, test_dict_list


# Gets all accuracies as a list.
def get_all_accuracies(log_file):
    test_acc_list = []
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    for test_acc in test_dict_list:
        test_acc_list.append(test_acc['Accuracy1'])
    return test_acc_list


# Helper function to get accuracies as dict.
def get_test_accuracies_dict(log_file):
    test_acc_dict = OrderedDict()
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    for test_acc in test_dict_list:
        test_acc_dict[int(test_acc['NumIters'])] = test_acc['Accuracy1']
    return test_acc_dict


def get_snapshot_list(log_file):
    print "Parsing [%s] for snapshots" % log_file
    regex_iteration = re.compile('Iteration (\d+)')
    regex_snapshot = re.compile('Snapshotting solver state to binary proto file (\S+)')
    iteration = -1
    snapshot_list = []

    with open(log_file, 'r') as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue
            snapshot_match = regex_snapshot.search(line)
            if snapshot_match:
                snapshot_file = snapshot_match.group(1)
                iter_no = int(snapshot_file.split(".")[0].split("_")[-1])
                snapshot_list.append((iter_no, snapshot_file))
    
    return snapshot_list

# Helper function to get epoch.
def get_last_epoch_snapshot(log_file):
    snapshot_list = get_snapshot_list(log_file)
    print snapshot_list
    last_iter, snapshot_file = snapshot_list[-1]
    return last_iter, snapshot_file

# Helper function to get epoch.
def get_last_test_epoch(log_file):
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    if len(test_dict_list) == 0:
        return -1, []
    last_test_epoch = test_dict_list[-1]
    return last_test_epoch['NumIters'], last_test_epoch

# Run caffe command line and return accuracies.
def run_caffe_return_accuracy(solver_fname, log_file, caffe_root, num_iter=-1, gpu_to_use=None):
    cmd_suffix = ''
    if num_iter > 0:
        cmd_suffix = ' --iterations %d ' % num_iter

    if gpu_to_use is not None:
        run_cmd = '%s train --solver %s --gpu %i %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, gpu_to_use, cmd_suffix, log_file)
    else:
        run_cmd = '%s train --solver %s %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, cmd_suffix, log_file)

    # Run the caffe code.
    print "Running [%s]" % run_cmd
    os.system(run_cmd)

    # Get the accuracy values.
    if check_out_of_memory(log_file):
        return None, None
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    acc = test_dict_list[-1]['Accuracy1']
    acc_dict = {test_dict_list[-1]['NumIters']: test_dict_list[-1]['Accuracy1']}
    return acc, acc_dict

def run_caffe_from_snapshot(solver_fname, log_file, snapshot_file, caffe_root, gpu_to_use=None):
    if gpu_to_use is not None:
        run_cmd = '%s train --solver %s --gpu %i --snapshot %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, gpu_to_use, snapshot_file, log_file)
    else:
        run_cmd = '%s train --solver %s --snapshot %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, snapshot_file, log_file)

    # Run the caffe code.
    print "Running [%s]" % run_cmd
    os.system(run_cmd)

    test_acc_list = get_all_accuracies(log_file)
    return test_acc_list
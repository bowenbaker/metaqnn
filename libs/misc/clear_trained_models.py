import pandas as pd
import numpy as np
import os
import argparse
import re
import csv

def clear_redundant_logs_caffe(ckpt_dir, replay):
    ''' Deletes uneeded log files and model saves:
        args:
            replay - with standard replay dic columns. Deletes only model saves that aren't for the first iteration,
                        the best iteration, and the last iteration
            ckpt_dir - where the models are saved, it must have a filemap
    '''
    file_map = pd.read_csv(os.path.join(ckpt_dir, 'file_map.csv'))

    for i in range(len(file_map)):
        net = file_map.net.values[i]

        # First check that the model was run on this computer
        if net not in replay.net.values:
            continue
        else:
            folder_path = os.path.join(ckpt_dir, str(int(file_map[file_map.net == net].file_number.values[0])))
            if not os.path.isdir(folder_path):
                continue
            best_iter = replay[replay.net == net].iter_best_val.values[0]
            last_iter = replay[replay.net == net].iter_last_val.values[0]
            model_saves = [f for f in os.listdir(folder_path) if f.find('modelsave_iter') >= 0]

            #make sure that the model actual ran
            if not model_saves:
                continue
            first_iter = min([int(re.split('_|\.', f)[2]) for f in model_saves])
            model_saves_to_keep = ['modelsave_iter_%i.solverstate' % iteration for iteration in [best_iter, last_iter, first_iter]] + \
                                  ['modelsave_iter_%i.caffemodel' % iteration for iteration in [best_iter, last_iter, first_iter]]

            # make sure all of the files are there
            if not np.all([os.path.isfile(os.path.join(folder_path, savefile)) for savefile in model_saves_to_keep]):
                continue

            # Delete extraneous files
            for f in model_saves:
                if f not in model_saves_to_keep:
                    os.remove(os.path.join(folder_path, f))

def main():
    parser = argparse.ArgumentParser()
    #get current supported model choices

    parser.add_argument('ckpt_dir')
    parser.add_argument('replay_dictionary_path')
    args = parser.parse_args()

    if not os.path.isfile(args.replay_dictionary_path):
        print 'NOT VALID DICTIONARY'
        return
    if not os.path.isdir(args.ckpt_dir):
        print 'NOT VALID CHECKPOINT DIR'
        return
    

    replay = pd.read_csv(args.replay_dictionary_path)

    clear_redundant_logs_caffe(args.ckpt_dir, replay)



    



if __name__ == '__main__':
    main()

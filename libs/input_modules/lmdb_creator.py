import argparse
import caffe
import lmdb
import numpy as np
import os
import pandas as pd
import shutil

from scipy import misc

import get_datasets
import preprocessing

def add_padding(data, pad, pad_value):
    if pad <= 0:
        return data
    print "Adding pads [%d]" % pad
    shp = data.shape
    padded = np.zeros((shp[0], shp[1], shp[2] + 2*pad, shp[3] + 2*pad))
    padded[:,:,:,:] = pad_value
    padded[:,:,pad:-pad,pad:-pad] = data
    print padded.shape
    return padded

def create_record(X, y, path, save_as_float=False):
    '''Creates a single LMDB file. Path is the full path of filename to store'''

    assert(X.shape[0] == y.shape[0])
    assert(y.min() == 0)
    if os.path.isdir(path):
        print 'removing ' + path
        shutil.rmtree(path)

    N = X.shape[0]

    map_size = X.nbytes * 50
  
    env = lmdb.open(path, map_size=map_size)
    
    print 'creating ' + path

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            if save_as_float:
                datum.float_data.extend(X[i].astype(float).flat)
            else:
                datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)
    
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def shuffle(X, y):
    ''' X and y must be the same length vectors '''
    assert X.shape[0] == y.shape[0]
    assert len(X.shape) == 4
    new_index = np.random.permutation(np.arange(X.shape[0]))
    return X[new_index, :, :, :], y[new_index]


def create_records(Xtr,
                   Ytr,
                   Xte,
                   Yte,
                   root_path,
                   number_val=0,
                   per_image_fn=None,
                   gcn=False,
                   mean_subtraction=False,
                   save_as_float=False,
                   pad=0,
                   Xval=None,
                   Yval=None):
    ''' Splits Xtr in validation and train sets. Also saves a full train lmdb.
        full train and split train are both first shuffled before being saved

        If both Xval is not None AND number_val > 0, we create a new validation set from Xtr combined with Xval
    '''
    print 'Labels train', np.unique(Ytr)
    print 'Labels test', np.unique(Yte)

    if save_as_float:
        print 'Converting to Float'
        Xtr = Xtr.astype(float)
        Ytr = Ytr.astype(float)
        Xte = Xte.astype(float)
        Yte = Yte.astype(float)
        if Xval is not None:
            Xval = Xval.astype(float)
            Yval = Yval.astype(float)

    if per_image_fn is not None:
        print 'Applying ' + per_image_fn.__name__ + ' to training set'
        for i in range(Xtr.shape[0]):
            Xtr[i] = per_image_fn(Xtr[i].T).T
        print 'Applying ' + per_image_fn.__name__ + ' to testing set'
        for i in range(Xte.shape[0]):
            Xte[i] = per_image_fn(Xte[i].T).T
        if Xval is not None:
            print 'Applying ' + per_image_fn.__name__ + ' to validation set'
            for i in range(Xval.shape[0]):
                Xval[i] = per_image_fn(Xval[i].T).T

    if Xval is not None:
        train_x = Xtr.copy()
        train_y = Ytr.copy()
        val_x = Xval.copy()
        val_y = Yval.copy()
        Xtr = np.concatenate([Xtr, Xval])
        Ytr = np.concatenate([Ytr, Yval])

    if number_val:
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        for i in np.unique(Ytr):
            X_label = Xtr[Ytr == i].copy()
            proportion = float(X_label.shape[0]) / Xtr.shape[0]
            divide = int(round(proportion * number_val))

            # Deal with wierd rounding error
            if val_x and np.concatenate(val_x).shape[0] + divide > number_val:
                divide = number_val - np.concatenate(val_x).shape[0]

            val_x.append(X_label[:divide])
            val_y.append([i]*divide)
            train_x.append(X_label[divide:])
            train_y.append([i]*(X_label.shape[0] - divide))

        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        val_x = np.concatenate(val_x)
        val_y = np.concatenate(val_y)
        assert(val_x.shape[0] == number_val)

    if Xval is not None or number_val > 0:

        train_x, train_y = shuffle(train_x, train_y)


        if gcn:
            print 'Train Small Before GCN Mean, std ', np.mean(train_x), np.std(train_x)
            print 'Validation Before GCN mean, std', np.mean(val_x), np.std(val_x)
            train_x, val_x = preprocessing.gcn_whiten(train_x, val_x)
            print 'Train Small Mean, std: ', np.mean(train_x), np.std(train_x)
            print 'Validation Mean, std: ', np.mean(val_x), np.std(val_x)

        if mean_subtraction:
            train_x, val_x = preprocessing.mean_subtraction(train_x, val_x)
            print 'Train Small Mean: ', np.mean(train_x)
            print 'Validation Mean: ', np.mean(train_x)

        if pad:
            pad_value = 0 if gcn or mean_subtraction else 128
            train_x = add_padding(train_x, pad, pad_value)

        print 'Train Small x shape', train_x.shape, train_x.dtype
        print 'Train Small y shape', train_y.shape, train_y.dtype
        print 'Validation x shape', val_x.shape, val_x.dtype
        print 'Validation y shape', val_y.shape, val_y.dtype
        print 'Biggest Class is %f of training set' % (np.unique(train_y, return_counts=True)[1].max()  / float(len(train_y)))
        print 'Biggest Class is %f of validation set' % (np.unique(val_y, return_counts=True)[1].max() / float(len(val_y)))

        create_record(train_x, train_y, os.path.join(root_path, 'train.lmdb'), save_as_float=save_as_float)
        create_record(val_x, val_y, os.path.join(root_path, 'val.lmdb'), save_as_float=save_as_float)
        del train_x, train_y, val_x, val_y

    Xtr, Ytr = shuffle(Xtr, Ytr)

    if gcn:
        print 'Train Small Before GCN Mean, std ', np.mean(Xtr), np.std(Xtr)
        print 'Test Before GCN mean, std', np.mean(Xte), np.std(Xte)
        Xtr, Xte = preprocessing.gcn_whiten(Xtr, Xte)
        print 'Train Mean, std: ', np.mean(Xtr), np.std(Xtr)
        print 'Test Mean, std: ', np.mean(Xte), np.std(Xte)

    if mean_subtraction:
        Xtr, Xte = preprocessing.mean_subtraction(Xtr, Xte)
        print 'Train Mean: ', np.mean(Xtr)
        print 'Test Mean: ', np.mean(Xte)

    if pad:
        pad_value = 0 if gcn or mean_subtraction else 128
        Xtr = add_padding(Xtr, pad, pad_value)

    print 'Train x shape', Xtr.shape, Xtr.dtype
    print 'Train y shape', Ytr.shape, Ytr.dtype
    print 'Test x shape', Xte.shape, Xte.dtype
    print 'Test y shape', Yte.shape, Yte.dtype

    create_record(Xtr, Ytr, os.path.join(root_path, 'train_full.lmdb'), save_as_float=save_as_float)
    create_record(Xte, Yte, os.path.join(root_path, 'test.lmdb'), save_as_float=save_as_float)


def main():
    parser = argparse.ArgumentParser()
    dataset_options = ['cifar10', 'cifar100', 'svhn', 'svhn_full', 'svhn_small', 'mnist', 'caltech101']
    parser.add_argument('dataset', choices=dataset_options, help='Which data set')
    parser.add_argument('root_save_dir', help='Where to save lmdb')
    parser.add_argument('-v','--number_val', help='How many validation images', type=int, default=0)
    parser.add_argument('-prep', '--preprocessing', help='Which per image prepocessing function to use', default=None, choices=['lcn', 'standard_whiten'])
    parser.add_argument('-gcn', '--gcn', help='Whether to use global contrast normalization or not. Default is false', default=False, type=bool)
    parser.add_argument('-ms', '--mean_subtraction', help='Do global mean subtraction?', default=False, type=bool)
    parser.add_argument('-odd', '--original_data_dir', help='Folder where original data is stored')
    parser.add_argument('-pad', '--padding', help='Padding value on each side.', type=int, default=0)

    args = parser.parse_args()
    if not os.path.isdir(args.root_save_dir):
        os.makedirs(args.root_save_dir)

    if args.original_data_dir and not os.path.isdir(args.original_data_dir):
        print 'ERROR: original data dir not real directory'
        return

    # Should we save as float?
    save_as_float = args.preprocessing is not 'none' or args.gcn or args.mean_subtraction

    # Get preprocessing function
    if args.preprocessing == 'lcn':
        per_image_fn = preprocessing.lcn_whiten
    elif args.preprocessing == 'standard_whiten':
        per_image_fn = preprocessing.standard_whiten
    else:
        per_image_fn = None

    #Convert to absolute paths
    root_save_dir = os.path.abspath(args.root_save_dir)

    save_dir=root_save_dir if not args.original_data_dir else None
    root_path=os.path.abspath(args.original_data_dir) if args.original_data_dir else None
    padding = args.padding if args.padding else 0 

    if args.dataset == 'cifar10':
        Xtr, Ytr, Xte, Yte = get_datasets.get_cifar10(save_dir=save_dir,
                                                      root_path=root_path)
        Xval, Yval = None, None
    elif args.dataset == 'cifar100':
        Xtr, Ytr, Xte, Yte = get_datasets.get_cifar100(save_dir=save_dir,
                                                       root_path=root_path)
        Xval, Yval = None, None
    elif args.dataset == 'svhn':
        Xtr, Ytr, Xte, Yte = get_datasets.get_svhn(save_dir=save_dir,
                                                   root_path=root_path)
        Xval, Yval = None, None
    elif args.dataset == 'svhn_full':
        Xtr, Ytr, Xval, Yval, Xte, Yte = get_datasets.get_svhn_full(save_dir=save_dir,
                                                                    root_path=root_path)
    elif args.dataset == 'svhn_small':
        Xtr, Ytr, Xval, Yval, Xte, Yte = get_datasets.get_svhn_small(save_dir=save_dir,
                                                            root_path=root_path)
    elif args.dataset == 'mnist':
        Xtr, Ytr, Xte, Yte = get_datasets.get_mnist(save_dir=save_dir,
                                                   root_path=root_path)
        Xval, Yval = None, None

    elif args.dataset == 'caltech101':
        Xtr, Ytr, Xval, Yval = get_datasets.get_caltech101(save_dir=save_dir,
                                                            root_path=root_path)
        Xte = Xval.copy()
        Yte = Yval.copy()

    create_records(Xtr=Xtr,
                   Ytr=Ytr,
                   Xte=Xte,
                   Yte=Yte,
                   root_path=root_save_dir,
                   number_val=args.number_val,
                   per_image_fn=per_image_fn,
                   gcn=args.gcn,
                   mean_subtraction=args.mean_subtraction,
                   save_as_float=save_as_float,
                   pad=args.padding,
                   Xval=Xval,
                   Yval=Yval)

if __name__ == "__main__": main()

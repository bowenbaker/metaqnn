MetaQNN Codebase
========

MetaQNN is a codebase used for automatically designing convolutional neural network architectures outlined in the paper: 

**[Designing Neural Network Architectures Using Reinforcement Learning](https://arxiv.org/pdf/1611.02167.pdf)**   
Bowen Baker, Otkrist Gupta, Nikhil Naik, Ramesh Raskar  
*International Conference on Learning Representations*, 2017

If our software or paper helps your research or project, please cite us using:

    @article{baker2017designing,
      title={Designing Neural Network Architectures using Reinforcement Learning},
      author={Baker, Bowen and Gupta, Otkrist and Naik, Nikhil and Raskar, Ramesh},
      journal={International Conference on Learning Representations},
      year={2017}
    }

# Installation
All code was only tested on ubuntu 16.04, python 2.7, with caffe at commit [d208b71](https://github.com/BVLC/caffe/tree/d208b714abb8425f1b96793e04508ad21724ae3f)

1. Install caffe using these [instructions](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide) with CUDA 8 and cuDNN 5.1.
2. ```pip install -r requirements.txt```

# Quick Example (CIFAR-10)
1. Create CIFAR-10 LMDB's on each server you plan to use for training  

    ``` bash
    python libs/input_modules/lmdb_creator.py cifar10 /path/to/data/directory/cifar10 -gcn True -v 5000
    ```
    
2. Modify `models/cifar10/hyper_parameters.py`  
  2a. set `TRAIN_FILE = '/path/to/data/directory/cifar10/train.lmdb'`  
  2b. set `VAL_FILE = '/path/to/data/directory/cifar10/val.lmdb'`  
  2c. set `CAFFE_ROOT = '/path/to/caffe/installation/directory'`  
  2d. (optional) set `CHECKPOINT_DIR = '/path/to/model/snapshot/directory/'`  
3. Create directory `cifar10_logs` to store Q-values and replay database
4. Start Q-Learning Server

    ```bash 
    python q_server.py cifar10 cifar10_logs
    ```
    
5. On each server you want to use for training start a Q-Learning Client

    ```bash
    python caffe_client.py cifar10 unique_client_identifier server_ip_addr
    ```
    If you want to use a specific gpu, for example GPU 4
    ```bash
    python caffe_client.py cifar10 unique_client_identifier server_ip_addr -gpu 4
    ```
    If you are using a multi-gpu server and want to run 4 clients that use GPUs 0 1 3 5 (This command requires you to have tmux installed)
    ```bash
    ./caffe_multiclient.sh cifar10 unique_client_identifier server_ip_addr 0 1 3 5
    ```

# MetaQNN Code Description

Experiment configurations are stored in the `models` folder. Each experiment contains a `hyper_parameters.py` file that contains optimization hyperparameters, data paths, etc., and a `state_space_parameters.py` file that contains state space specifications. The sample experiments are densely commented so that you may easily change around the experiment configurations.

We implemented the Q-Learning algorithm in a distributed server-client framework. One server runs the Q-Learning algorithm and sends out jobs to train CNNs on client servers. We currently only have published a client that uses Caffe for CNN training. If there is enough interest I will publish a client that uses MXNet as well.

## Dataset Creation
We provide easy-to-use helper functions to download and preprocess the CIFAR-10, CIFAR-100, MNIST, and SVHN datasets. It supports standard whitening, local contrast normalization, global contrast normalization, mean subtraction, and padding. The module will save both training and validation lmdbs as well as the full training set and test set lmdbs to the specified location. To see all options run
```bash
python libs/input_modules/lmdb_creator.py -h
```
#### Examples
1. Create CIFAR-10 dataset with global contrast normalization and 5000 validation images run

    ```bash
    python libs/input_modules/lmdb_creator.py cifar10 /path/to/data/directory/cifar10 -gcn True -v 5000
    ```
    
2. Create MNIST dataset with mean subtraction and 10000 validation images

    ```bash
    python libs/input_modules/lmdb_creator.py mnist /path/to/data/directory/mnist -ms True -v 10000
    ```

3. Create the SVHN dataset with the extra 531131 training images and local contrast normalization and standard validation set

    ```bash
    python libs/input_modules/lmdb_creator.py svhn_full /path/to/data/directory/svhn_full -prep lcn 
    ```
    
4. Create the 10% SVHN dataset with standard whitening

    ```bash
    python libs/input_modules/lmdb_creator.py svhn_small /path/to/data/directory/svhn -prep standard_whiten
    ```

# Speeding Up Meta-Modeling with Performance Prediction

If you have limited hardware or just want to run through a larger number of networks, we highly recommend implementing early stopping with simple performance prediction models as outlined in our recent paper

**[Practical Neural Network Performance Prediction for Early Stopping](https://arxiv.org/pdf/1705.10823.pdf)**   
Bowen Baker\*, Otkrist Gupta\*, Nikhil Naik, Ramesh Raskar  
*Under Submission*

We will be releasing code for this before the end of the summer, but the method is extremely simple so you shouldn't have any trouble implementing it yourself if you need to use it before our release.

import pandas as pd
import numpy as np
from scipy import misc
from shutil import copyfile, rmtree
import argparse
import os
import cv2

def standard_whiten(image):
    flattened_image = np.reshape(image.astype(float), [image.shape[0] * image.shape[1], image.shape[2]])
    mean = np.mean(flattened_image, axis=0)
    std = np.std(flattened_image, axis=0)
    std_alt = [1 / np.sqrt(image.shape[0] * image.shape[1])] * image.shape[2]
    std = np.maximum(std, std_alt)
    return (image - mean) / std

def lcn_whiten(image, kernel_size=7, threshhold=0.001):
    ''' Uses opencv guassian blur to estimate the local mean and std
        Assumes that image comes in in (channel, x, y) format

        returns a type float image
    '''
    im = image.astype(float)
    local_mean = cv2.GaussianBlur(im, (kernel_size, kernel_size), 1.0)
    local_var  = cv2.GaussianBlur(np.multiply(im,im), (kernel_size, kernel_size), 1.0)
    local_std = np.sqrt(local_var) 
    local_std = np.maximum(threshhold, local_std)

    lcn_image = (im - local_mean) / local_std

    return lcn_image

def gcn_whiten(Xtr, Xte, threshhold=0.001):
    ''' Xtr and Xte should be in format (image index, chanel, x, y)
        Returns Xtr and Xte as type float
    '''
    global_mean = np.mean(Xtr, axis=0)
    global_std = np.std(Xtr, axis=0)
    global_std = np.maximum(threshhold, global_std)

    Xtr = (Xtr - global_mean) / global_std
    Xte = (Xte - global_mean) / global_std

    return Xtr, Xte

def mean_subtraction(Xtr, Xte):
    global_mean = np.mean(Xtr, axis=0)
    Xtr = Xtr - global_mean
    Xte = Xte - global_mean

    return Xtr, Xte


def preprocess(image_dir, new_image_dir, preprocess_fn):
    
    image_paths = []
    labels = []

    if os.path.isdir(new_image_dir):
        rmtree(new_image_dir)
    os.makedirs(new_image_dir)

    classes = os.listdir(image_dir)

    for clas in classes:
        class_dir = os.path.join(image_dir, str(clas))
        new_class_dir = os.path.join(new_image_dir, str(clas))
        os.makedirs(new_class_dir)
        
        for image_name in os.listdir(class_dir):
            image = misc.imread(os.path.join(class_dir, image_name))
            image = preprocess_fn(image)
            misc.imsave(os.path.join(new_class_dir, image_name), image)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='path to subject directories of images')
    parser.add_argument('new_image_dir', help='path to directory to store record')
    parser.add_argument('preprocess_fn', choices=['standard_whiten'])

    args = parser.parse_args()
    if not os.path.isdir(args.image_dir):
        print 'ERROR: image_dir not real directory'
        return

    #Convert to absolute paths
    image_dir = os.path.abspath(args.image_dir)
    new_image_dir = os.path.abspath(args.new_image_dir)

    if args.preprocess_fn == 'standard_whiten':
        preprocess_fn = standard_whiten

    preprocess(image_dir, new_image_dir, preprocess_fn)

if __name__ == "__main__": main()

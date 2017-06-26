import math
import numpy as np
import os
import sys

import libs.grammar.cnn as cnn

from libs.grammar.state_string_utils import StateStringUtils

import caffe

from caffe import layers as cl
from caffe import params as P
from caffe import to_proto

class Parser:
    def __init__(self, hyper_parameters, state_space_parameters):
        self.hp = hyper_parameters
        self.ssp = state_space_parameters


    def replace_top_names(self, cc, dbg_out=False, simplify_naming=False):
        layer_out_replacement_map = {cc.layer[0].top[0]: 'data', cc.layer[1].top[
            0]: 'data', cc.layer[0].top[1]: 'label', cc.layer[1].top[1]: 'label'}

        if simplify_naming:
            for i in range(len(cc.layer) - 1, 1, -1):
                ll = cc.layer[i]
                print(ll.name, ll.top)
                layer_out_replacement_map[ll.top[0]] = ll.name

        for i in range(len(cc.layer)):
            ll = cc.layer[i]
            for j in range(len(ll.top)):
                if ll.top[j] in layer_out_replacement_map:
                    ll.top[j] = layer_out_replacement_map[ll.top[j]]
            for j in range(len(ll.bottom)):
                if ll.bottom[j] in layer_out_replacement_map:
                    ll.bottom[j] = layer_out_replacement_map[ll.bottom[j]]
            if dbg_out:
                print(ll.name)
                print(ll.top)
                print(ll.bottom)
        return cc

    def add_batchnorm(self, bottom):
        bn = cl.BatchNorm(bottom, in_place=True)
        bn = cl.Scale(bn, bias_term=True, bias_filler=dict(value=0), in_place=True)
        return bn

    # Template for activation function.
    def add_activate(self, bottom):
        return cl.ReLU(bottom, in_place=True)

    # Creating directory with error checking.
    def create_dir(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                print exc 
                raise

    # Creates caffe prototxt specification.
    def create_caffe_spec(self, net_string, caffe_prototxt):
        self.create_dir(caffe_prototxt)
        cc = self.convert(net_string)
        with open(caffe_prototxt, "w") as f:
            f.write(str(cc))

    # Creates the top most layer (data layer) for caffe netspec.
    def create_top_layer(self, phase=caffe.TRAIN, input_file="", train=True):
        if self.hp.GCN_APPROX:
            transform_param = {'scale': 0.0078125,'mean_value': 128}
        else:
            transform_param = {}
        if train:
            transform_param['mirror'] = self.hp.MIRROR
            if self.hp.CROP:
                # Adds random crops.
                transform_param['crop_size'] = self.hp.IMAGE_HEIGHT
        
        data, label = cl.Data(
            batch_size=self.hp.TRAIN_BATCH_SIZE if train else self.hp.EVAL_BATCH_SIZE, backend=P.Data.LMDB, name="data",
            source=input_file, ntop=2, include={'phase': phase}, transform_param=transform_param)
        return data, label

    # MAIN FUNCTION: Converts a net string to a caffe netspec.
    # Adds the data layer and accuracy layer for test/train.
    def convert(self, net_string):
        net_list = cnn.parse('net', net_string)
        net_list = StateStringUtils(self.ssp).convert_model_string_to_states(net_list)[1:]
        data, label = self.create_top_layer(caffe.TRAIN, self.hp.TRAIN_FILE, train=True)
        data1, label1 = self.create_top_layer(caffe.TEST, self.hp.VAL_FILE, train=False)
        loss, acc = self.unpack_list(net_list, data, label)
        lls = [data, data1, acc, loss]

        cc = to_proto(*lls)
        cc = self.replace_top_names(cc)
        return cc

    # Iterate over token list from parser.
    def unpack_list(self, net_list, data, label):
        bottom = data
        for layer_number in range(len(net_list)):
            layer_spec = net_list[layer_number]
            bottom = self.unpack_item(layer_spec,
                                      net_list[layer_number - 1].image_size if layer_number else self.hp.IMAGE_HEIGHT,
                                      layer_number,
                                      bottom,
                                      label)
        # Add Softmax and Accuracy layers.
        loss = cl.SoftmaxWithLoss(bottom, label)
        acc = cl.Accuracy(bottom, label)
        return loss, acc

    def get_pad(self, kernel_size):
        return int(kernel_size / 2) if self.ssp.conv_padding == 'SAME' else 0

    # Parse a single token for topology.
    def unpack_item(self, layer, previous_image_size, layer_number, bottom, label=None):

        if layer.terminate == 1:
            # Softmax Accuracy/Loss
            # loss = cl.SoftmaxWithLoss(bottom, label)
            bottom = cl.InnerProduct(bottom, num_output=self.hp.NUM_CLASSES,
                                 weight_filler=dict(type='xavier'))
            return bottom

        if layer.layer_type == 'conv':
            out_depth = layer.filter_depth
            kernel_size = layer.filter_size
            stride = layer.stride
            pad = self.get_pad(kernel_size)
            bottom = cl.Convolution(bottom,
                                      kernel_size=kernel_size,
                                      num_output=out_depth,
                                      stride=stride,
                                      pad=pad,
                                      weight_filler=dict(type='xavier'))
            if self.ssp.batch_norm:
                bottom = self.add_batchnorm(bottom)
            return self.add_activate(bottom)

        if layer.layer_type == 'nin':
            out_depth = layer.filter_depth
            bottom = cl.Convolution(bottom,
                                    kernel_size=1,
                                    num_output=out_depth,
                                    weight_filler=dict(type='xavier'))
            bottom = self.add_activate(bottom)

            bottom = cl.Convolution(bottom,
                                    kernel_size=1,
                                    num_output=out_depth,
                                    weight_filler=dict(type='xavier'))
            bottom = self.add_activate(bottom)
            return bottom

        if layer.layer_type == 'gap':
            out_depth = self.hp.NUM_CLASSES
            bottom = cl.Convolution(bottom,
                                    kernel_size=1,
                                    num_output=out_depth,
                                    weight_filler=dict(type='xavier'))
            bottom = self.add_activate(bottom)
            bottom = cl.Pooling(
                bottom, kernel_size=previous_image_size, pool=P.Pooling.AVE)
            return bottom

        if layer.layer_type == 'fc':
            num_output = layer.fc_size
            bottom = cl.InnerProduct(bottom, num_output=num_output,
                                     weight_filler=dict(type='xavier'))
            bottom = self.add_activate(bottom)
            return bottom

        if layer.layer_type == 'dropout':
            dropout_ratio = 0.5 * float(layer.filter_depth) / layer.fc_size
            return cl.Dropout(bottom, dropout_ratio=dropout_ratio)

        if layer.layer_type == 'pool':
            kernel_size = layer.filter_size
            stride = layer.stride
            if self.ssp.batch_norm:
                bottom = self.add_batchnorm(bottom)
            return cl.Pooling(bottom, kernel_size=kernel_size, stride=stride, pool=P.Pooling.MAX)


def main(argv):
  parser = Parser()
  parser.create_caffe_spec(argv[1], argv[2])


if __name__ == "__main__":
  main(sys.argv)

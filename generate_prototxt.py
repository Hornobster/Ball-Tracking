#!/usr/bin/python
import os
import sys

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

from caffe import layers as L, params as P

def lenet(batch_size, phase):
    n = caffe.NetSpec()

    # empty layers as placeholders
    # the resulting prototxt must be edited manually
    n.data = L.Input()
    n.label = L.Input()

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1   = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=2, weight_filler=dict(type='xavier'))

    if (phase == 'TRAIN'):
        n.loss  = L.SoftmaxWithLoss(n.score, n.label)
    else if (phase == 'TEST'):
        n.prob = L.Softmax(n.score)
    
    return n.to_proto()

with open('lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet(50, 'TRAIN')))

with open('lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet(50, 'TEST')))


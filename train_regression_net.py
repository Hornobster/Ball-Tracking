#!/usr/bin/python
import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

TRAIN_DATASET_DIR       = os.path.join(os.getcwd(), './train_dataset')
TEST_DATASET_DIR        = os.path.join(os.getcwd(), './test_dataset')
BATCH_FILENAME_FORMAT	= 'dataset_batch%d.hdf5'
DATASET_MEAN_FILENAME	= 'dataset_mean.hdf5'
SOLVER_PROTO_FILENAME	= 'lenet_regression_solver.prototxt'
MODEL_FILENAME		= 'balltracker_regression.caffemodel'
BATCH_SIZE		= 100
NUM_TRAINING_ITERATIONS	= 500
TEST_INTERVAL		= 25
TEST_NUM_SAMPLES	= 1000
FIRST_LAYER		= 'conv1'

def loadBatch(datasetDir, batch_size, n, mean = None):
    data_arr = np.zeros((batch_size, 1, 100, 100))
    label_arr = np.zeros((batch_size, 3))

    hdf5 = os.path.join(datasetDir, BATCH_FILENAME_FORMAT % n) 
    f = h5py.File(hdf5, "r")
    
    images = f.keys()
    
    for idx, i in enumerate(images):
        data_arr[idx, 0, ...] = f[i][...]
        label_arr[idx, ...] = [f[i].attrs['CENTER_X'], f[i].attrs['CENTER_Y'] , f[i].attrs['RADIUS']]
    
    data_arr /= 256.0 # transform to [0, 1)
    label_arr /= 100.0 # transform to [0, 1]
    
    # subtract mean
    if mean is not None:
        data_arr[:, 0, ...] -= mean

    f.close()

    return data_arr, label_arr

# setup solver
caffe.set_device(0)
caffe.set_mode_gpu()

solver = None
solver = caffe.SGDSolver(SOLVER_PROTO_FILENAME)

train_loss = np.zeros(NUM_TRAINING_ITERATIONS)

testData, testLabel = loadBatch(TEST_DATASET_DIR, BATCH_SIZE, 0)

# test the trained net over test_dataset
def testNet(i, test_dataset_dir, solver, numTestSamples):
    print ('Iteration', i, 'testing...')
    correct = 0
    
    for test_it in range(numTestSamples / BATCH_SIZE):
        # load new test batch
        d, l = loadBatch(test_dataset_dir, BATCH_SIZE, test_it)
        solver.test_nets[0].blobs['data'].data[...] = d
        solver.test_nets[0].blobs['label'].data[...] = l

        solver.test_nets[0].forward(start=FIRST_LAYER)
        print('Test Iteration: %d Euclidean loss: %f' % (test_it, solver.test_nets[0].blobs['loss'].data))

# load mean
meanHdf = h5py.File(os.path.join(TRAIN_DATASET_DIR, DATASET_MEAN_FILENAME), 'r')
mean = np.zeros(meanHdf['mean'][...].shape, meanHdf['mean'][...].dtype)
mean[...] = meanHdf['mean'][...]
meanHdf.close()

# main training loop
for it in range(NUM_TRAINING_ITERATIONS):
    # load batch
    trainData, trainLabel = loadBatch(TRAIN_DATASET_DIR, BATCH_SIZE, it, mean)

    solver.net.blobs['data'].data[...] = trainData
    solver.net.blobs['label'].data[...] = trainLabel

    solver.step(1)

    train_loss[it] = solver.net.blobs['loss'].data
    
    if it % TEST_INTERVAL == 0:
        testNet(it, TEST_DATASET_DIR, solver, TEST_NUM_SAMPLES)

# last test
testNet(NUM_TRAINING_ITERATIONS, TEST_DATASET_DIR, solver, TEST_NUM_SAMPLES)
        
# save model
solver.net.save(MODEL_FILENAME)

# plot trainloss over iterations
plt.plot(np.arange(NUM_TRAINING_ITERATIONS), train_loss)
plt.xlabel('iteration')
plt.ylabel('train loss')
plt.show()


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
        if idx < batch_size:
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
train_loss_averaged = []
test_rmse = np.zeros((int(np.ceil(NUM_TRAINING_ITERATIONS / TEST_INTERVAL) + 1), 3))
test_rmse_dummy = np.zeros((2, 3))
rmse_dummy_computed = False

testData, testLabel = loadBatch(TEST_DATASET_DIR, BATCH_SIZE, 0)

# test the trained net over test_dataset
def testNet(i, test_dataset_dir, solver, numTestSamples):
    print ('Iteration', i, 'testing...')

    mean_squared_error = np.zeros((3), dtype='float')
    mean_squared_error_dummy = np.zeros((3), dtype='float')
    
    global rmse_dummy_computed
    for test_it in range(numTestSamples / BATCH_SIZE):
        # load new test batch
        d, l = loadBatch(test_dataset_dir, BATCH_SIZE, test_it)
        solver.test_nets[0].blobs['data'].data[...] = d
        solver.test_nets[0].blobs['label'].data[...] = l

        solver.test_nets[0].forward(start=FIRST_LAYER)

        mean_squared_error += np.sum(np.square(solver.test_nets[0].blobs['score'].data[:] - l), axis = 0)

        if not rmse_dummy_computed:
            mean_squared_error_dummy += np.sum(np.square([0.5, 0.5, 0.275] - l), axis = 0)

    rmse = np.sqrt(mean_squared_error / numTestSamples)

    test_rmse[i // TEST_INTERVAL] = rmse
    print ('RMSE: Center_x: %f, Center_y: %f, Radius: %f' % (rmse[0], rmse[1], rmse[2]))

    if not rmse_dummy_computed:
        rmse_dummy = np.sqrt(mean_squared_error_dummy / numTestSamples)
        test_rmse_dummy[0] = rmse_dummy
        test_rmse_dummy[1] = rmse_dummy

        rmse_dummy_computed = True
    print ('RMSE Dummy: Center_x: %f, Center_y: %f, Radius: %f' % (test_rmse_dummy[0, 0], test_rmse_dummy[0, 1], test_rmse_dummy[0, 2]))

# load mean
meanHdf = h5py.File(os.path.join(TRAIN_DATASET_DIR, DATASET_MEAN_FILENAME), 'r')
mean = np.zeros(meanHdf['mean'][...].shape, meanHdf['mean'][...].dtype)
mean[...] = meanHdf['mean'][...]
meanHdf.close()

moving_window = 100
tmp_loss_average = 0.0

# main training loop
for it in range(NUM_TRAINING_ITERATIONS):
    # load batch
    trainData, trainLabel = loadBatch(TRAIN_DATASET_DIR, BATCH_SIZE, it, mean)

    solver.net.blobs['data'].data[...] = trainData
    solver.net.blobs['label'].data[...] = trainLabel

    solver.step(1)

    train_loss[it] = solver.net.blobs['loss'].data

    # compute moving average of training loss
    tmp_loss_average += train_loss[it] / moving_window
    if (it + 1) % moving_window == 0:
        train_loss_averaged.append(tmp_loss_average)
        tmp_loss_average = 0.0
    
    if it % TEST_INTERVAL == 0:
        testNet(it, TEST_DATASET_DIR, solver, TEST_NUM_SAMPLES)

# last test
testNet(NUM_TRAINING_ITERATIONS, TEST_DATASET_DIR, solver, TEST_NUM_SAMPLES)
        
# save model
solver.net.save(MODEL_FILENAME)

# plot trainloss and test rmse over iterations
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(NUM_TRAINING_ITERATIONS), train_loss)
ax1.plot(np.arange(moving_window / 2, NUM_TRAINING_ITERATIONS + moving_window / 2, moving_window), train_loss_averaged, 'y', linewidth=2)
ax2.plot(TEST_INTERVAL * np.arange(len(test_rmse)), test_rmse[:, 0], 'r', label='C_x', linewidth = 2)
ax2.plot([0, NUM_TRAINING_ITERATIONS], test_rmse_dummy[:, 0], 'r--', label='C_x Dummy')
ax2.plot(TEST_INTERVAL * np.arange(len(test_rmse)), test_rmse[:, 1], 'g', label='C_y', linewidth = 2)
ax2.plot([0, NUM_TRAINING_ITERATIONS], test_rmse_dummy[:, 1], 'g--', label='C_y Dummy')
ax2.plot(TEST_INTERVAL * np.arange(len(test_rmse)), test_rmse[:, 2], 'm', label='R', linewidth = 2)
ax2.plot([0, NUM_TRAINING_ITERATIONS], test_rmse_dummy[:, 2], 'm--', label='R Dummy')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test rmse')
ax2.set_title('Test RMSE - C_x: %f C_y: %f R: %f' % (test_rmse[-1, 0], test_rmse[-1, 1], test_rmse[-1, 2]))
ax2.legend()
plt.show()


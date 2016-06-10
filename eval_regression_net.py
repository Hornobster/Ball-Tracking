#!/usr/bin/python
import numpy as np
import os
import sys
import h5py

# check console arguments
if len(sys.argv) != 6:
    print('Usage: %s descriptor model datasetDir batchSize numSamples' % sys.argv[0])
    sys.exit(1)

# get console arguments
regressorDescriptor, regressorModel, datasetDir, batchSize, numSamples = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

BATCH_FILENAME_FORMAT	= 'dataset_batch%d.hdf5'
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

# setup caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# load model
net = caffe.Net(regressorDescriptor, regressorModel, caffe.TEST)

mean_squared_error = np.zeros((3), dtype='float')
mean_squared_error_dummy = np.zeros((3), dtype='float')
mean_absolute_error = np.zeros((3), dtype='float')
mean_absolute_error_dummy = np.zeros((3), dtype='float')

averages = np.zeros((3), dtype='float')

# main loop
numBatches = numSamples // batchSize
for i in range(numBatches):
    if i % 10 == 0:
        print ('Testing batch %d / %d... %f%%' % (i, numBatches, float(i) / numBatches * 100.0))

    # load new test batch
    d, l = loadBatch(datasetDir, batchSize, i)
    net.blobs['data'].data[...] = d

    net.forward(start=FIRST_LAYER)

    mean_squared_error += np.sum(np.square(net.blobs['score'].data[:] - l), axis = 0)
    mean_squared_error_dummy += np.sum(np.square([0.5, 0.5, 0.275] - l), axis = 0)
    mean_absolute_error += np.sum(np.abs(net.blobs['score'].data[:] - l), axis = 0)
    mean_absolute_error_dummy += np.sum(np.abs([0.5, 0.5, 0.275] - l), axis = 0)

    averages += np.sum(l, axis = 0)

rmse = np.sqrt(mean_squared_error / numSamples)
rmse_dummy = np.sqrt(mean_squared_error_dummy / numSamples)

mean_absolute_error /= numSamples
mean_absolute_error_dummy /= numSamples
averages /= numSamples

print ('RMSE: Center_x: %f, Center_y: %f, Radius: %f' % (rmse[0], rmse[1], rmse[2]))
print ('RMSE Dummy: Center_x: %f, Center_y: %f, Radius: %f' % (rmse_dummy[0], rmse_dummy[1], rmse_dummy[2]))
print ('MAE: Center_x: %f, Center_y: %f, Radius: %f' % (mean_absolute_error[0], mean_absolute_error[1], mean_absolute_error[2]))
print ('MAE Dummy: Center_x: %f, Center_y: %f, Radius: %f' % (mean_absolute_error_dummy[0], mean_absolute_error_dummy[1], mean_absolute_error_dummy[2]))
print ('Average: Center_x: %f, Center_y: %f, Radius: %f' % (averages[0], averages[1], averages[2]))


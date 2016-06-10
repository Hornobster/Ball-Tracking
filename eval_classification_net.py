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
classifierDescriptor, classifierModel, datasetDir, batchSize, numSamples = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

BATCH_FILENAME_FORMAT	= 'dataset_batch%d.hdf5'
FIRST_LAYER		= 'conv1'

def loadBatch(datasetDir, batch_size, n):
    data_arr = np.zeros((batch_size, 1, 100, 100))
    label_arr = np.zeros((batch_size))

    hdf5 = os.path.join(datasetDir, BATCH_FILENAME_FORMAT % n) 
    f = h5py.File(hdf5, "r")
    
    images = f.keys()
    
    for idx, i in enumerate(images):
        if idx < batch_size:
            data_arr[idx, 0, ...] = f[i][...]
            label_arr[idx] = np.int32(f[i].attrs['HAS_SPHERE'])
    
    data_arr /= 256.0 # transform to [0, 1)
    
    f.close()

    return data_arr, label_arr

# setup caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# load model
net = caffe.Net(classifierDescriptor, classifierModel, caffe.TEST)

correct = 0
confusion_matrix = np.zeros((4), dtype='uint32') # [ TP, FN, FP, TN]

# main loop
numBatches = numSamples // batchSize
for i in range(numBatches):
    if i % 10 == 0:
        print ('Testing batch %d / %d... %f%%' % (i, numBatches, float(i) / numBatches * 100.0))

    # load new test batch
    d, l = loadBatch(datasetDir, batchSize, i)
    net.blobs['data'].data[...] = d
    net.blobs['label'].data[...] = l

    net.forward(start=FIRST_LAYER)

    correct += sum(net.blobs['prob'].data.argmax(1) == net.blobs['label'].data)
    
    predicted = net.blobs['prob'].data.argmax(1)
    for p in range(batchSize):
        label = int(net.blobs['label'].data[p])
        if label == 1 and predicted[p] == 1: # true positive
            confusion_matrix[0] += 1
        elif label == 1 and predicted[p] == 0: # false negative
            confusion_matrix[1] += 1
        elif label == 0 and predicted[p] == 1: # false positive
            confusion_matrix[2] += 1
        elif label == 0 and predicted[p] == 0: # true negative
            confusion_matrix[3] += 1

print ('Correct: %d / %d Accuracy: %f' % (correct, numSamples, float(correct) / numSamples * 100.0))
print ('True Positives: %d False Negatives: %d False Positives: %d True Negatives: %d' % (confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]))

with np.errstate(divide='ignore', invalid='ignore'):
    print ('TPR (Recall): %f FPR (Fall-Out): %f' % (float(confusion_matrix[0]) / (confusion_matrix[0] + confusion_matrix[1]), float(confusion_matrix[2]) / (confusion_matrix[2] + confusion_matrix[3])))


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
SOLVER_PROTO_FILENAME	= 'lenet_classification_solver.prototxt'
MODEL_FILENAME		= 'balltracker.caffemodel'
BATCH_SIZE		= 100
NUM_TRAINING_ITERATIONS	= 100
TEST_INTERVAL		= 25
TEST_NUM_SAMPLES	= 1000
FIRST_LAYER		= 'conv1'

def loadBatch(datasetDir, batch_size, n, mean = None):
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
test_acc = np.zeros(int(np.ceil(NUM_TRAINING_ITERATIONS / TEST_INTERVAL) + 1))
output = np.zeros((NUM_TRAINING_ITERATIONS, 8, 2))

testData, testLabel = loadBatch(TEST_DATASET_DIR, BATCH_SIZE, 0)

# auroc statistics
auroc_thresholds = np.linspace(0, 1, BATCH_SIZE)
auroc_stats = np.zeros((len(auroc_thresholds), 4), dtype='uint32') # [ TP, FN, FP, TN ]

# test the trained net over test_dataset
def testNet(i, test_dataset_dir, solver, numTestSamples, auroc = False):
    print ('Iteration', i, 'testing...')
    correct = 0
    
    global auroc_thresholds
    global auroc_stats
    auroc_stats = np.zeros((len(auroc_thresholds), 4), dtype='uint32')

    for test_it in range(numTestSamples / BATCH_SIZE):
        # load new test batch
        d, l = loadBatch(test_dataset_dir, BATCH_SIZE, test_it)
        solver.test_nets[0].blobs['data'].data[...] = d
        solver.test_nets[0].blobs['label'].data[...] = l

        solver.test_nets[0].forward(start=FIRST_LAYER)
        correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
        test_acc[i // TEST_INTERVAL] = float(correct) / numTestSamples * 100.0
        
        if auroc:
            for p in range(BATCH_SIZE):
                for idx, threshold in enumerate(auroc_thresholds):
                    label = int(solver.test_nets[0].blobs['label'].data[p])
                    predicted = solver.test_nets[0].blobs['prob'].data[p][1]
                    if label == 1 and (predicted > threshold): # true positive
                        auroc_stats[idx][0] += 1
                    elif label == 1 and (predicted < threshold): # false negative
                        auroc_stats[idx][1] += 1
                    elif label == 0 and (predicted > threshold): # false positive
                        auroc_stats[idx][2] += 1
                    elif label == 0 and (predicted < threshold): # true negative
                        auroc_stats[idx][3] += 1

    print(test_acc[i // TEST_INTERVAL])

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
    
    # small test for first 8 images
    solver.test_nets[0].blobs['data'].data[...] = testData
    solver.test_nets[0].blobs['label'].data[...] = testLabel
    solver.test_nets[0].forward(start=FIRST_LAYER)
    output[it] = solver.test_nets[0].blobs['prob'].data[:8]
    
    if it % TEST_INTERVAL == 0:
        testNet(it, TEST_DATASET_DIR, solver, TEST_NUM_SAMPLES)

# last test
testNet(NUM_TRAINING_ITERATIONS, TEST_DATASET_DIR, solver, TEST_NUM_SAMPLES, True)
        
# save model
solver.net.save(MODEL_FILENAME)

# plot auroc
fpr = ((auroc_stats[:, 2]).astype(float) / (auroc_stats[:, 2] + auroc_stats[:, 3]))
tpr = ((auroc_stats[:, 0]).astype(float) / (auroc_stats[:, 0] + auroc_stats[:, 1]))
z = np.linspace(min(fpr), max(fpr))
plt.plot(z, z, '--')
plt.plot(fpr, tpr, 'r')
plt.fill_between(fpr, tpr, 0, color='blue', alpha=0.3)
plt.show()

# reload first test batch for plotting
solver.test_nets[0].blobs['data'].data[...] = testData
solver.test_nets[0].blobs['label'].data[...] = testLabel
solver.test_nets[0].forward(start=FIRST_LAYER)

'''
# plot scores over iterations for 8 test images
for i in range(8):
    plt.figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    plt.figure(figsize=(10, 2))
    plt.imshow(output[:NUM_TRAINING_ITERATIONS, i].T, interpolation='nearest', cmap='gray')
    plt.xlabel('iteration')
    plt.ylabel('label')
plt.show()

# plot conv2 layer gradients
plt.imshow(solver.net.params['conv2_2'][0].diff[:, 0].reshape(8, 4, 3, 3)
       .transpose(0, 2, 1, 3).reshape(8*4, 3*3), cmap='gray', interpolation='none')
plt.show()
'''

# plot softmax over iterations for 8 test images
correct = ((solver.test_nets[0].blobs['prob'].data.argmax(1)[:8] == solver.test_nets[0].blobs['label'].data[:8]))
print(sum(correct), '/', len(correct))
for i in range(8):
    plt.figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    plt.figure(figsize=(10, 2))
    plt.imshow(output[NUM_TRAINING_ITERATIONS-5:NUM_TRAINING_ITERATIONS, i].T, interpolation='nearest', cmap='gray')
    plt.xlabel('iteration')
    plt.ylabel('label')
plt.show()

# plot trainloss and test accuracy over iterations
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(NUM_TRAINING_ITERATIONS), train_loss)
ax2.plot(TEST_INTERVAL * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()


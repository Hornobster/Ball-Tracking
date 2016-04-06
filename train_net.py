#!/usr/bin/python
import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

def loadBatch(hdf5, batch_size, n):
    data_arr = np.zeros((batch_size, 1, 100, 100))
    label_arr = np.zeros((batch_size))
    f = h5py.File(hdf5, "r")
    
    images = [i for i in f.keys()[n*batch_size:n*batch_size+batch_size]]
    
    for idx, i in enumerate(images):
        data_arr[idx, 0, ...] = f[i][...]
        label_arr[idx] = np.int32(f[i].attrs['HAS_SPHERE'])
    
    data_arr /= 256.0 # transform to [0, 1)

    return data_arr, label_arr

# setup solver
caffe.set_device(0)
caffe.set_mode_gpu()

solver = None
solver = caffe.SGDSolver('lenet_auto_solver.prototxt')

niter = 200
test_interval = 25
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval) + 1))
output = np.zeros((niter, 8, 2))

testData, testLabel = loadBatch('test_dataset.hdf5', 50, 0)

# test the trained net over test_dataset
def testNet(i, test_dataset, solver):
    print ('Iteration', i, 'testing...')
    correct = 0
    for test_it in range(20):
        # load new test batch
        d, l = loadBatch(test_dataset, 50, test_it)
        solver.test_nets[0].blobs['data'].data[...] = d
        solver.test_nets[0].blobs['label'].data[...] = l

        solver.test_nets[0].forward(start='conv1_1')
        correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
        test_acc[i // test_interval] = correct / 1e3
    print(test_acc[i // test_interval])


# main training loop
for it in range(niter):
    # load batch
    trainData, trainLabel = loadBatch('train_dataset.hdf5', 50, it)

    solver.net.blobs['data'].data[...] = trainData
    solver.net.blobs['label'].data[...] = trainLabel

    solver.step(1)

    train_loss[it] = solver.net.blobs['loss'].data
    
    # small test for first 8 images
    solver.test_nets[0].blobs['data'].data[...] = testData
    solver.test_nets[0].blobs['label'].data[...] = testLabel
    solver.test_nets[0].forward(start='conv1_1')
    output[it] = solver.test_nets[0].blobs['prob'].data[:8]
    
    if it % test_interval == 0:
        testNet(it, 'test_dataset.hdf5', solver)

# last test
testNet(200, 'test_dataset.hdf5', solver)

# save model
solver.net.save('balltracker.caffemodel')

# reload first test batch for plotting
solver.test_nets[0].blobs['data'].data[...] = testData
solver.test_nets[0].blobs['label'].data[...] = testLabel
solver.test_nets[0].forward(start='conv1_1')

'''
# plot scores over iterations for 8 test images
for i in range(8):
    plt.figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    plt.figure(figsize=(10, 2))
    plt.imshow(output[:200, i].T, interpolation='nearest', cmap='gray')
    plt.xlabel('iteration')
    plt.ylabel('label')
plt.show()
'''

# plot conv2 layer gradients
plt.imshow(solver.net.params['conv2_2'][0].diff[:, 0].reshape(8, 4, 3, 3)
       .transpose(0, 2, 1, 3).reshape(8*4, 3*3), cmap='gray', interpolation='none')
plt.show()

# plot probs
_, ax3 = plt.subplots()
ax4 = ax3.twinx()
ax3.plot(np.arange(len(hassphere_probs)), hassphere_probs)
ax4.plot(np.arange(len(nosphere_probs)), nosphere_probs, 'r')
ax3.set_ylabel('hassphere_probs')
ax4.set_ylabel('nosphere_probs')
plt.show()

# plot softmax over iterations for 8 test images
correct = ((solver.test_nets[0].blobs['prob'].data.argmax(1)[:8] == solver.test_nets[0].blobs['label'].data[:8]))
print(sum(correct), '/', len(correct))
for i in range(8):
    plt.figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    plt.figure(figsize=(10, 2))
    plt.imshow(output[195:200, i].T, interpolation='nearest', cmap='gray')
    plt.xlabel('iteration')
    plt.ylabel('label')
plt.show()

# plot trainloss and test accuracy over iterations
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()


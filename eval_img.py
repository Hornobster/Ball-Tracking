#!/usr/bin/python
import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

netdescriptor, model, hdfdb, imageId = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# load model
net = caffe.Net(netdescriptor, model, caffe.TEST)

# prepare data
f = h5py.File(hdfdb, 'r')

net.blobs['data'].data[0, 0, ...] = f[imageId][...]
net.blobs['data'].data[0, 0, ...] /= 256.0
f.close()

net.forward()
softmax_probabilities = net.blobs['prob'].data[0]

print(softmax_probabilities)

if (softmax_probabilities.argmax(0) == 0):
    print('No sphere')
else:
    print('Has sphere')

# activation heatmap
actual_image = np.empty_like(net.blobs['data'].data[0,0])
np.copyto(actual_image, net.blobs['data'].data[0,0])
heatmap = np.zeros((50, 50))
for x in range(50):
    for batch in range(1):
        for y in range(50):
            occluded_image = np.empty_like(actual_image)
            np.copyto(occluded_image, actual_image)
            
            # occlude 21x21 patch with stride 2
            occluded_image[max(0, x*2 - 10):min(100, x*2 + 11), max(0, y*2 - 10):min(100, y*2 + 11)] = 0.5
            net.blobs['data'].data[y,0] = occluded_image
            
        net.forward()
        print('%d / 2500' % (x * 50 + batch * 50))
        heatmap[x,...] = 1.0 - net.blobs['prob'].data[...,1]

import matplotlib
cmap_heatmap = heatmap - np.min(heatmap[np.nonzero(heatmap)])
cmap_heatmap = cmap_heatmap.astype(np.float32) / np.max(cmap_heatmap)
cmap_heatmap = matplotlib.cm.jet(cmap_heatmap, bytes=True)
cmap_heatmap = Image.fromarray(cmap_heatmap)
colored = ImageChops.multiply(Image.fromarray(actual_image * 256).resize((50, 50)).convert('RGBA'), cmap_heatmap)

plt.figure()
plt.imshow(actual_image, cmap='gray')
plt.figure()
plt.imshow(heatmap)
plt.figure()
plt.imshow(colored) 
plt.show()


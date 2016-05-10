#!/usr/bin/python
import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageDraw

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

import caffe

def analyse_frame(net, model, image, batch_size, patch_size, interval):
    # open the image and convert to array
    image = image.convert('L');
    image_array = np.array(image) / 256.0 # normalize to [0, 1)

    # calculate variables for the for loops
    image_w = image.size[0]
    image_h = image.size[1]

    steps_w = (image_w - patch_size) / interval + 1
    rows_per_batch = batch_size / (steps_w + 1) # add 1 to account for the possible last patch in each row, not counted by steps_w
    num_batches = (image_h - patch_size) / (rows_per_batch * interval)

    # max probability of sphere
    found = False
    max_prob = 0.0
    max_prob_x = 0
    max_prob_y = 0

    # main loop
    for batch in range(num_batches):
	net.blobs['data'].data[:] = 0 # zero out the matrix

	# load patches
	index = 0;
	for row in range(rows_per_batch):
	    h = batch * rows_per_batch + row
	    for w in range(steps_w):
		net.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, w * interval:w * interval + patch_size]

		index += 1
	    # last column
	    net.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, image_w - patch_size:image_w]

	    index += 1

	# inference
	net.forward()
	max_prob_idx = net.blobs['prob'].data[:, 1].argmax(0)
	m_prob = net.blobs['prob'].data[max_prob_idx, 1]

	# update max probability
	if m_prob >= 0.5 and m_prob > max_prob:
	    found = True
	    max_prob = m_prob

	    tmp_row = max_prob_idx / (steps_w + 1)
	    tmp_w = max_prob_idx % (steps_w + 1)
	    tmp_h = batch * rows_per_batch + tmp_row
	
	    if tmp_w != steps_w:
		max_prob_x = tmp_w * interval
		max_prob_y = tmp_h * interval
	    else:
		max_prob_x = image_w - patch_size
		max_prob_y = tmp_h * interval

    # last batch has less than rows_per_batch rows
    remaining_rows = ((image_h - patch_size) - (num_batches * rows_per_batch + 1) * interval) / interval + 1

    net.blobs['data'].data[:] = 0 # zero out the matrix
    index = 0
    for row in range(remaining_rows):
	h = num_batches * rows_per_batch + row + 1
	for w in range(steps_w):
	    net.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, w * interval:w * interval + patch_size]
	    index += 1
	net.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, image_w - patch_size:image_w]
	index += 1

    # last row
    for w in range(steps_w):
	net.blobs['data'].data[index, 0, ...] = image_array[image_h - patch_size:image_h, w * interval:w * interval + patch_size]
	index += 1
    net.blobs['data'].data[index, 0, ...] = image_array[image_h - patch_size:image_h, image_w - patch_size:image_w]

    # inference
    net.forward()
    max_prob_idx = net.blobs['prob'].data[:, 1].argmax(0)
    m_prob = net.blobs['prob'].data[max_prob_idx, 1]

    # update max probability
    if m_prob > max_prob:
	found = True
	max_prob = m_prob

	tmp_row = max_prob_idx / (steps_w + 1)
	tmp_w = max_prob_idx % (steps_w + 1)
	tmp_h = num_batches * rows_per_batch + tmp_row + 1

	if tmp_row == remaining_rows:
	    if tmp_w != steps_w:
		max_prob_x = tmp_w * interval
		max_prob_y = image_h - patch_size
	    else:
		max_prob_x = image_w - patch_size
		max_prob_y = image_h - patch_size
	if tmp_w != steps_w:
	    max_prob_x = tmp_w * interval
	    max_prob_y = tmp_h * interval
	else:
	    max_prob_x = image_w - patch_size
	    max_prob_y = tmp_h * interval

    return (found, max_prob, max_prob_x, max_prob_y)
    
# if the script is called from command line and not imported
if __name__ == '__main__':
    # setup solver
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # get console arguments
    netdescriptor, model, imageFilename = sys.argv[1], sys.argv[2], sys.argv[3]

    # load model
    net = caffe.Net(netdescriptor, model, caffe.TEST)

    # open the image and convert to array
    image = Image.open(imageFilename).convert('L');

    interval = 25
    patch_size = 100
    batch_size = 100

    found, max_prob, max_prob_x, max_prob_y = analyse_frame(net, model, image, batch_size, patch_size, interval)

    if found: 
	# draw rectangle around solution
	image_color = image.convert('RGB')

	draw = ImageDraw.Draw(image_color)

	draw.rectangle([(max_prob_x, max_prob_y), (max_prob_x + patch_size, max_prob_y + patch_size)], outline='red')
	draw.rectangle([(max_prob_x + 1, max_prob_y + 1), (max_prob_x + patch_size - 1, max_prob_y + patch_size - 1)], outline='red')
	draw.rectangle([(max_prob_x + 2, max_prob_y + 2), (max_prob_x + patch_size - 2, max_prob_y + patch_size - 2)], outline='red')

	plt.imshow(image_color)
	plt.show()
    else:
	print ("No ball found!")

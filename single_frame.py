#!/usr/bin/python
import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageChops, ImageDraw

def analyse(classifier, regressor, image, batch_size, patch_size, interval):
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
	classifier.blobs['data'].data[:] = 0 # zero out the matrix

	# load patches
	index = 0;
	for row in range(rows_per_batch):
	    h = batch * rows_per_batch + row
	    for w in range(steps_w):
		classifier.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, w * interval:w * interval + patch_size]
		index += 1
	    # last column
	    classifier.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, image_w - patch_size:image_w]

	    index += 1

	# inference
	classifier.forward()
	max_prob_idx = classifier.blobs['prob'].data[:, 1].argmax(0)
	m_prob = classifier.blobs['prob'].data[max_prob_idx, 1]

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
    remaining_rows = ((image_h - patch_size) - (num_batches * rows_per_batch * interval)) / interval + 1

    classifier.blobs['data'].data[:] = 0 # zero out the matrix
    index = 0
    for row in range(remaining_rows):
	h = num_batches * rows_per_batch + row
	for w in range(steps_w):
	    classifier.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, w * interval:w * interval + patch_size]
	    index += 1
	classifier.blobs['data'].data[index, 0, ...] = image_array[h * interval:h * interval + patch_size, image_w - patch_size:image_w]

	index += 1

    # last row
    for w in range(steps_w):
	classifier.blobs['data'].data[index, 0, ...] = image_array[image_h - patch_size:image_h, w * interval:w * interval + patch_size]
	index += 1
    classifier.blobs['data'].data[index, 0, ...] = image_array[image_h - patch_size:image_h, image_w - patch_size:image_w]

    # inference
    classifier.forward()
    max_prob_idx = classifier.blobs['prob'].data[:, 1].argmax(0)
    m_prob = classifier.blobs['prob'].data[max_prob_idx, 1]

    # update max probability
    if m_prob >= 0.5 and m_prob > max_prob:
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
	elif tmp_w != steps_w:
	    max_prob_x = tmp_w * interval
	    max_prob_y = tmp_h * interval
	else:
	    max_prob_x = image_w - patch_size
	    max_prob_y = tmp_h * interval

    center_x = 0.0
    center_y = 0.0
    radius = 0.0

    if found:
        regressor.blobs['data'].data[0, 0, ...] = image_array[max_prob_y:max_prob_y + patch_size, max_prob_x:max_prob_x + patch_size]
        regressor.forward()
        center_x, center_y, radius = regressor.blobs['score'].data[0] * 100.0

    return (found, max_prob, max_prob_x, max_prob_y, center_x, center_y, radius)
    
def analyseAroundPoint(classifier, regressor, image, batch_size, patch_size, interval, old_x, old_y):
    assert (batch_size >= 9), "analyseAroundPoint: batch_size must be 9 or greater!"

    # open the image and convert to array
    image = image.convert('L');
    image_array = np.array(image) / 256.0 # normalize to [0, 1)

    # calculate variables for the for loops
    image_w = image.size[0]
    image_h = image.size[1]

    # if frame is smaller than classifier's patch
    # return "not found"
    if image_w < patch_size and image_h < patch_size:
        print ('Frame is smaller than classifier\'s patch! Frame: %dx%d Patch: %dx%d' % (image_w, image_h, patch_size, patch_size))
        return (False, 0, 0, 0, 0, 0, 0)

    # zero out the matrix
    classifier.blobs['data'].data[:] = 0

    # analyse a 3x3 grid around old_x, old_y
    # the middle cell goes from old_x, old_y to old_x + patch_size, old_y + patch_size
    for row in range(3):
        for col in range(3):
            # compute x1, x2
            x1 = max(0, old_x + interval * (col - 1))
            x2 = min(image_w, x1 + patch_size)

            if x2 - patch_size != x1:
                x1 = x2 - patch_size

            # compute y1, y2
            y1 = max(0, old_y + interval * (row - 1))
            y2 = min(image_h, y1 + patch_size)

            if y2 - patch_size != y1:
                y1 = y2 - patch_size

            # load patches
            index = row * 3 + col;
            classifier.blobs['data'].data[index, 0, ...] = image_array[y1:y2, x1:x2]

    # max probability of sphere
    found = False
    max_prob = 0.0
    max_prob_x = 0
    max_prob_y = 0

    # inference
    classifier.forward()
    max_prob_idx = classifier.blobs['prob'].data[:, 1].argmax(0)
    max_prob = classifier.blobs['prob'].data[max_prob_idx, 1]

    # update max probability, if prob > 50% and valid index
    if max_prob_idx < 9 and max_prob >= 0.5:
	found = True

	tmp_row = max_prob_idx / 3
        tmp_col = max_prob_idx % 3

        # compute x1, x2
        max_prob_x = max(0, old_x + interval * (tmp_col - 1))
        x2 = min(image_w, max_prob_x + patch_size)

        if x2 - patch_size != max_prob_x:
            max_prob_x = x2 - patch_size

        # compute y1, y2
        max_prob_y = max(0, old_y + interval * (tmp_row - 1))
        y2 = min(image_h, max_prob_y + patch_size)

        if y2 - patch_size != max_prob_y:
            max_prob_y = y2 - patch_size

    # variables for regression
    center_x = 0.0
    center_y = 0.0
    radius = 0.0

    if found:
        regressor.blobs['data'].data[0, 0, ...] = image_array[max_prob_y:max_prob_y + patch_size, max_prob_x:max_prob_x + patch_size]
        regressor.forward()
        center_x, center_y, radius = regressor.blobs['score'].data[0] * 100.0

    return (found, max_prob, max_prob_x, max_prob_y, center_x, center_y, radius)

def probabilityMap(classifier, image, batch_size, patch_size, interval):
    # open the image and convert to array
    image = image.convert('L');

    # calculate variables for the for loops
    image_w = image.size[0]
    image_h = image.size[1]

    # if frame is smaller than classifier's patch
    # return "not found"
    if image_w < patch_size and image_h < patch_size:
        print ('Frame is smaller than classifier\'s patch! Frame: %dx%d Patch: %dx%d' % (image_w, image_h, patch_size, patch_size))
        return None

    # zero out the matrix
    classifier.blobs['data'].data[:] = 0

    # compute image padding
    if interval * (image_w / interval) == image_w:
        padded_w = image_w
    else:
        padded_w = interval * (image_w / interval + 1)

    if interval * (image_h / interval) == image_h:
        padded_h = image_h
    else:
        padded_h = interval * (image_h / interval + 1)

    # only pad image if needed
    if image_w != padded_w or image_h != padded_h:
        image_padded = Image.new('L', (padded_w, padded_h))
        image_padded.paste(image, ((padded_w - image_w) / 2, (padded_h - image_h) / 2))
    else:
        image_padded = image

    image_array = np.array(image_padded) / 256.0 # normalize to [0, 1)

    # loop variables
    prob_row = (padded_h - patch_size) / interval + 1
    prob_col = (padded_w - patch_size) / interval + 1

    probs = np.zeros((prob_row * prob_col), dtype='float')
    prob_idx = 0

    index = 0
    for row in range(0, padded_h - patch_size + interval, interval):
        for col in range(0, padded_w - patch_size + interval, interval):
            classifier.blobs['data'].data[index, 0, ...] = image_array[row:row + patch_size, col:col + patch_size]

            index += 1

            # once batch is filled, calculate probabilities with the classifier
            if index == batch_size:
                # inference
                classifier.forward()

                probs[prob_idx:prob_idx + batch_size] = classifier.blobs['prob'].data[:, 1]

                # update indexes
                index = 0
                prob_idx += batch_size

    # inference for last batch
    if index > 0:
        classifier.forward()

        probs[prob_idx:] = classifier.blobs['prob'].data[:index, 1]

    probs = probs.reshape((prob_row, prob_col))

    # initialise probability map to zero
    probabilityMap = np.zeros(image_array.shape, dtype='float')
    probabilityMapCounters = np.zeros(image_array.shape, dtype='uint8')

    for row in range(0, prob_row):
        for col in range(0, prob_col):
            probabilityMap[row * interval:row * interval + patch_size, col * interval:col * interval + patch_size] += probs[row, col]
            probabilityMapCounters[row * interval:row * interval + patch_size, col * interval:col * interval + patch_size] += 1

    # compute average per pixel
    probabilityMap /= probabilityMapCounters

    # crop
    probabilityMap = probabilityMap[(padded_h - image_h) / 2:(padded_h - image_h) / 2 + image_h, (padded_w - image_w) / 2:(padded_w - image_w) / 2 + image_w]

    return probabilityMap

# if the script is called from command line and not imported
if __name__ == '__main__':
    caffe_root = os.getenv('CAFFE_ROOT', './')
    sys.path.insert(0, caffe_root + '/python')

    import caffe

    # setup solver
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # get console arguments
    classifierDescriptor, regressorDescriptor, classifierModel, regressorModel, imageFilename = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    # load models
    classifier = caffe.Net(classifierDescriptor, classifierModel, caffe.TEST)
    regressor = caffe.Net(regressorDescriptor, regressorModel, caffe.TEST)

    # open the image and convert to array
    image = Image.open(imageFilename).convert('L');

    image_color = image.convert('RGB')

    interval = 25
    patch_size = 100
    batch_size = 100

    # compose original image with computed probability map
    probMap = probabilityMap(classifier, image, batch_size, patch_size, interval)
    image_color = ImageChops.multiply(image_color, Image.fromarray(np.uint8(cm.jet(probMap, bytes=True))).convert('RGB'))

    # compute max probability patch
    found, max_prob, max_prob_x, max_prob_y, center_x, center_y, radius = analyse(classifier, regressor, image, batch_size, patch_size, interval)

    # draw rectangle around solution
    if found: 
	draw = ImageDraw.Draw(image_color)

	draw.rectangle([(max_prob_x, max_prob_y), (max_prob_x + patch_size, max_prob_y + patch_size)], outline='red')
	draw.rectangle([(max_prob_x + 1, max_prob_y + 1), (max_prob_x + patch_size - 1, max_prob_y + patch_size - 1)], outline='red')
	draw.rectangle([(max_prob_x + 2, max_prob_y + 2), (max_prob_x + patch_size - 2, max_prob_y + patch_size - 2)], outline='red')

        draw.ellipse([max_prob_x + center_x - radius, max_prob_y + center_y - radius, max_prob_x + center_x + radius, max_prob_y + center_y + radius], outline='red')
        draw.ellipse([max_prob_x + center_x - radius - 1, max_prob_y + center_y - radius - 1, max_prob_x + center_x + radius + 1, max_prob_y + center_y + radius + 1], outline='red')
        draw.ellipse([max_prob_x + center_x - radius - 2, max_prob_y + center_y - radius - 2, max_prob_x + center_x + radius + 2, max_prob_y + center_y + radius + 2], outline='red')
    else:
	print ("No ball found!")

    plt.imshow(image_color)
    plt.show()


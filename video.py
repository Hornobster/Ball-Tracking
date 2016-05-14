#!/usr/bin/python
import single_frame as frame
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from PIL import Image, ImageDraw

caffe_root = os.getenv('CAFFE_ROOT', './')
sys.path.insert(0, caffe_root + '/python')

# check command line arguments
if len(sys.argv) != 7:
    print ("Usage: video.py classifierDescriptor regressorDescriptor classifierModel regressorModel inputFilename outputFilename")
    sys.exit(1)

FFMPEG_BIN = 'ffmpeg'
FFPROBE_BIN = 'ffprobe'

def getVideoInfo(filename):
    command = [FFPROBE_BIN,
               '-v', 'error', # if there are errors, print them
               '-show_streams', # show all streams in file
               '-of', 'json', # print in json
               filename]

    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)

    infos = json.loads(pipe.stdout.read())['streams'][0] # first stream
    errors = pipe.stderr.read()
    pipe.terminate()

    width = infos['width']
    height = infos['height']
    n_frames = int(infos['nb_frames'])

    # if video is rotated, swap width and height
    if infos['tags']['rotate'] == '90' or infos['tags']['rotate'] == '-90':
        width, height = height, width

    print (errors)

    return width, height, n_frames

# get console arguments
classifierDescriptor, regressorDescriptor, classifierModel, regressorModel, inputFilename, outputFilename = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

# read video information
width, height, n_frames = getVideoInfo(inputFilename)

import caffe

# setup solver
caffe.set_device(0)
caffe.set_mode_gpu()

# load models
classifier = caffe.Net(classifierDescriptor, classifierModel, caffe.TEST)
regressor = caffe.Net(regressorDescriptor, regressorModel, caffe.TEST)

# credit: http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
# console commands to read and write videos using FFMPEG
read_command = [FFMPEG_BIN,
                '-i', inputFilename,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']

write_command = [FFMPEG_BIN,
                 '-y', # overwrite output file if exists
                 '-f', 'rawvideo',
                 '-s', '%dx%d' % (width, height),
                 '-pix_fmt', 'rgb24',
                 '-r', '30', # frames per second
                 '-i', '-', # input comes from pipe
                 '-an',
                 '-vcodec', 'libx264',
                 '-pix_fmt', 'yuv420p',
                 outputFilename]

inPipe = sp.Popen(read_command, stdout = sp.PIPE, stderr=sp.PIPE, bufsize = 10**8)
outPipe = sp.Popen(write_command, stdin = sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

# frame counter
f = 0

done = False
while not done:
    sys.stdout.write('Analysing frame %d / %d. %.1f%%\r' % (f, n_frames, float(f) / n_frames * 100))
    sys.stdout.flush()

    # read 1 frame
    raw_image = inPipe.stdout.read(width * height * 3)
    image_color = np.fromstring(raw_image, dtype='uint8')
    if image_color.size == width * height * 3:
        image_color = image_color.reshape((height, width, 3))
        image_color = Image.fromarray(image_color)

        image = image_color.convert('L')

        interval = 25
        patch_size = 100
        batch_size = 100

        found, max_prob, max_prob_x, max_prob_y, center_x, center_y, radius = frame.analyse(classifier, regressor, image, batch_size, patch_size, interval)

        if found: 
            # draw rectangle around solution
            draw = ImageDraw.Draw(image_color)

            # draw best matching patch
            draw.rectangle([(max_prob_x, max_prob_y), (max_prob_x + patch_size, max_prob_y + patch_size)], outline='red')
            draw.rectangle([(max_prob_x + 1, max_prob_y + 1), (max_prob_x + patch_size - 1, max_prob_y + patch_size - 1)], outline='red')
            draw.rectangle([(max_prob_x + 2, max_prob_y + 2), (max_prob_x + patch_size - 2, max_prob_y + patch_size - 2)], outline='red')

            # draw from regressor's prediction (center and radius)
            draw.ellipse([max_prob_x + center_x - radius, max_prob_y + center_y - radius, max_prob_x + center_x + radius, max_prob_y + center_y + radius], outline='red')
            draw.ellipse([max_prob_x + center_x - radius - 1, max_prob_y + center_y - radius - 1, max_prob_x + center_x + radius + 1, max_prob_y + center_y + radius + 1], outline='red')
            draw.ellipse([max_prob_x + center_x - radius - 2, max_prob_y + center_y - radius - 2, max_prob_x + center_x + radius + 2, max_prob_y + center_y + radius + 2], outline='red')
        else:
            print ("No ball found!\n")
        
        outPipe.stdin.write(image_color.tostring())

        inPipe.stdout.flush()
        outPipe.stdin.flush()
    else:
        inPipe.terminate()
        outPipe.terminate()
        done = True

    # increase frame counter
    f += 1

print('\nDone.')


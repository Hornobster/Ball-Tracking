#!/usr/bin/python
import sys
import os
import random
import math
from PIL import Image

# get all backgrounds filenames in 'backgrounds' directory
bkgDir = os.path.join(os.getcwd(), './backgrounds')
bkgFilenames = [f for f in os.listdir(bkgDir) if os.path.isfile(os.path.join(bkgDir, f))]

# get all spheres filenames in 'spheres' directory
spheresDir = os.path.join(os.getcwd(), './spheres')
spheresFilenames = [f for f in os.listdir(spheresDir) if os.path.isfile(os.path.join(spheresDir, f))]

# create 'dataset' directory, if it doesn't exist
datasetDir = os.path.join(os.getcwd(), './dataset')
if not os.path.exists(datasetDir):
    os.makedirs(datasetDir)

# generate N dataset images
for x in range(10):
    try:
        # choose a random background
        bkgFilename = random.choice(bkgFilenames)
        background = Image.open(os.path.join(bkgDir, bkgFilename))

        # create new greyscale image
        result = Image.new('L', (100, 100))

        # copy a 100x100px patch from background and paste it into result
        bkgPatch = background.crop((0, 0, 100, 100))
        result.paste(bkgPatch, (0, 0, 100, 100))

        # with probability 50% add a sphere to the image
        if random.random() < 0.5:
            # choose a random sphere
            sphereFilename = random.choice(spheresFilenames)
            sphere = Image.open(os.path.join(spheresDir, sphereFilename))

            # copy and resize patch from sphere and paste it into result
            spherePatch = sphere.crop((5, 5, 205, 205))
            sphereDiameter = int(random.random() * 90 + 10) # random integer value in [10, 100]
            spherePatch = spherePatch.resize((sphereDiameter, sphereDiameter))
            result.paste(spherePatch, \
                         (50 - int(math.floor(sphereDiameter / 2.0)), 50 - int(math.floor(sphereDiameter / 2.0)), \
                         50 + int(math.ceil(sphereDiameter / 2.0)), 50 + int(math.ceil(sphereDiameter / 2.0))), \
                         spherePatch) # the second spherePatch is used as alpha mask

        # save result
        result.save(os.path.join(datasetDir, 'data%d.png' % x))
    except IOError as e:
        print('I/O Error(%d): %s' % (e.errno, e.strerror))


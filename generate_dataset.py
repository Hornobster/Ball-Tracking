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
        result = Image.new('L', (200, 200))

        # check that the background image is large enough
        if (background.size[0] < 100 or background.size[1] < 100):
            print('Warning: background must be at least 100x100px. %s skipped.' % bkgFilename)
            continue

        # copy a 100x100px patch from background and paste it into result
        left, top = int(random.random() * (background.size[0] - 100)), int(random.random() * (background.size[1] - 100))
        bkgPatch = background.crop((left, top, left + 100, top + 100))
        result.paste(bkgPatch, (50, 50, 150, 150))

        # with probability 50% add a sphere to the image
        if random.random() < 0.5:
            # choose a random sphere
            sphereFilename = random.choice(spheresFilenames)
            sphere = Image.open(os.path.join(spheresDir, sphereFilename))

            # copy and resize patch from sphere and paste it into result
            spherePatch = sphere.crop((5, 5, 205, 205))
            sphereDiameter = int(random.random() * 90 + 10) # random integer value in [10, 100]
            sphereCenter = (random.random() * 100 + 50, random.random() * 100 + 50)
            spherePatch = spherePatch.resize((sphereDiameter, sphereDiameter))
            pasteBox = (int(sphereCenter[0] - math.floor(sphereDiameter / 2.0)), int(sphereCenter[1] - math.floor(sphereDiameter / 2.0)), \
                        int(sphereCenter[0] + math.ceil(sphereDiameter / 2.0)), int(sphereCenter[1] + math.ceil(sphereDiameter / 2.0))) 
            result.paste(spherePatch, \
                         pasteBox,
                         spherePatch) # the second spherePatch is used as alpha mask

        # crop to 100x100
        result = result.crop((50, 50, 150, 150))

        # save result
        result.save(os.path.join(datasetDir, 'data%d.png' % x))
    except IOError as e:
        print('I/O Error(%d): %s' % (e.errno, e.strerror))


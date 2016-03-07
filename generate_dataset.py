#!/usr/bin/python
import sys
import os
import random
import math
from PIL import Image
import numpy as np
import h5py

# get all backgrounds filenames in 'backgrounds' directory
bkgDir = os.path.join(os.getcwd(), './backgrounds')
bkgFilenames = [f for f in os.listdir(bkgDir) if os.path.isfile(os.path.join(bkgDir, f))]

# get all spheres filenames in 'spheres' directory
spheresDir = os.path.join(os.getcwd(), './spheres')
spheresFilenames = [f for f in os.listdir(spheresDir) if os.path.isfile(os.path.join(spheresDir, f))]

# create HDF5 file
f = h5py.File('dataset.hdf5', 'w')

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

        # defaults, used when saving the HDF5 file
        hasSphere = False
        sphereDiameter = 1
        sphereCenter = (0, 0)

        # with probability 50% add a sphere to the image
        if random.random() < 0.5:
            hasSphere = True

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

        # save result to HDF5 DB
        dset = f.create_dataset('%03d' % x, (100, 100), dtype='uint8')
        dset[...] = np.array(result)

        # set attributes for grayscale images
        dset.attrs['CLASS'] = np.str_('IMAGE')
        dset.attrs['VERSION'] = np.str_('1.2')
        dset.attrs['IMAGE_SUBCLASS'] = np.str_('IMAGE_GRAYSCALE')
        dset.attrs['IMAGE_WHITE_IS_ZERO'] = np.uint8(0)
        
        # save attributes for training
        dset.attrs['HAS_SPHERE'] = hasSphere
        if (hasSphere):
            dset.attrs['RADIUS'] = sphereDiameter / 2
            dset.attrs['CENTER_X'] = sphereCenter[0] - 50
            dset.attrs['CENTER_Y'] = sphereCenter[1] - 50
    except IOError as e:
        print('I/O Error(%d): %s' % (e.errno, e.strerror))

f.flush()
f.close()

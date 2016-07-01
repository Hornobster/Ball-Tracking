#!/usr/bin/env python
import sys
import os
import random
import math
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import h5py

if len(sys.argv) != 5:
    print('Usage: %s datasetDir numSamples batchSize sphereProbability' % sys.argv[0])
    sys.exit(1)

datasetDir, N, BATCH_SIZE, probability = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])

# get all backgrounds filenames in 'backgrounds' directory
bkgDir = os.path.join(os.getcwd(), './backgrounds')
bkgFilenames = [f for f in os.listdir(bkgDir) if os.path.isfile(os.path.join(bkgDir, f))]

# get all spheres filenames in 'spheres' directory
spheresDir = os.path.join(os.getcwd(), './spheres')
spheresFilenames = [f for f in os.listdir(spheresDir) if os.path.isfile(os.path.join(spheresDir, f))]

# create directory for dataset, if it doesn't exist
datasetDir = os.path.join(os.getcwd(), datasetDir)
if not os.path.exists(datasetDir):
     os.makedirs(datasetDir)

# create HDF5 file for mean
meanHdf = h5py.File(os.path.join(datasetDir, './dataset_mean.hdf5'), 'w')

# prepare mean matrix
datasetMean = np.zeros((100, 100), dtype='double')

numBatches = N / BATCH_SIZE

batch = 0
while batch < numBatches:
    if batch % 10 == 0:
        print('%d / %d %f%%' % (batch, numBatches, float(batch) / numBatches * 100))

    # create HDF5 file for dataset
    f = h5py.File(os.path.join(datasetDir, './dataset_batch%d.hdf5' % batch), 'w')

    x = 0
    while x < BATCH_SIZE:
        try:
            # choose a random background
            bkgFilename = random.choice(bkgFilenames)
            background = Image.open(os.path.join(bkgDir, bkgFilename))

            # create new greyscale image
            result = Image.new('L', (200, 200))

            # check that the background image is large enough
            if (background.size[0] < 100 or background.size[1] < 100):
                print('Warning: background must be at least 100x100px. %s deleted.' % bkgFilename)
                os.remove(os.path.join(bkgDir, bkgFilename))
                bkgFilenames.remove(bkgFilename)
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
            if random.random() < probability:
                hasSphere = True

                # choose a random sphere
                sphereFilename = random.choice(spheresFilenames)
                sphere = Image.open(os.path.join(spheresDir, sphereFilename))

                # copy and resize patch from sphere and paste it into result
                sphereDiameter = int(random.random() * 90 + 10) # random integer value in [10, 100]
                sphereCenter = (random.random() * 100 + 50, random.random() * 100 + 50)
                sphere= sphere.resize((sphereDiameter, sphereDiameter))
                pasteBox = (int(sphereCenter[0] - math.floor(sphereDiameter / 2.0)), int(sphereCenter[1] - math.floor(sphereDiameter / 2.0)), \
                            int(sphereCenter[0] + math.ceil(sphereDiameter / 2.0)), int(sphereCenter[1] + math.ceil(sphereDiameter / 2.0))) 

                # random brightness
                sphereEnhanced = sphere.convert('RGB')
                enhancer = ImageEnhance.Brightness(sphereEnhanced)
                sphereEnhanced = enhancer.enhance(random.random() + 0.5)
                
                # random blur
                blur = ImageFilter.GaussianBlur(int(random.random() * sphereDiameter / 20))
                sphereEnhanced = sphereEnhanced.filter(blur)
                blurredAlpha = sphere.split()[-1].filter(blur)
                sphere.putalpha(blurredAlpha)

                result.paste(sphereEnhanced, \
                             pasteBox,
                             sphere) # the second sphere is used as alpha mask

            # crop to 100x100
            result = result.crop((50, 50, 150, 150))

            # random brightness
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(random.random() + 0.5)

            # save result to HDF5 DB
            dset = f.create_dataset('%07d' % x, (100, 100), dtype='uint8')
            dset[...] = np.array(result)
            
            # update mean
            datasetMean += dset[...].astype('double') / N

            # set attributes for grayscale images
            dset.attrs['CLASS'] = np.str_('IMAGE')
            dset.attrs['VERSION'] = np.str_('1.2')
            dset.attrs['IMAGE_SUBCLASS'] = np.str_('IMAGE_GRAYSCALE')
            dset.attrs['IMAGE_WHITE_IS_ZERO'] = np.uint8(0)
            
            # save attributes for training
            dset.attrs['HAS_SPHERE'] = np.uint8(hasSphere)
            if (hasSphere):
                dset.attrs['RADIUS'] = np.float(sphereDiameter / 2)
                dset.attrs['CENTER_X'] = np.float(sphereCenter[0] - 50)
                dset.attrs['CENTER_Y'] = np.float(sphereCenter[1] - 50)
        except IOError as e:
            print('I/O Error(%d): %s' % (e.errno, e.strerror))
            
        x += 1

    # release resources
    f.flush()
    f.close()

    batch += 1

# save mean to HDF5 DB
dset = meanHdf.create_dataset('mean', (100, 100), dtype='double')
dset[...] = datasetMean / 256.0

# set attributes for grayscale images
dset.attrs['CLASS'] = np.str_('IMAGE')
dset.attrs['VERSION'] = np.str_('1.2')
dset.attrs['IMAGE_SUBCLASS'] = np.str_('IMAGE_GRAYSCALE')
dset.attrs['IMAGE_WHITE_IS_ZERO'] = np.uint8(0)

# release resources
meanHdf.flush()
meanHdf.close()


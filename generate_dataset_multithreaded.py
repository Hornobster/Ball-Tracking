#!/usr/bin/python
import sys
import os
import random
import math
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import h5py
import threading
import signal

# handle process killing
quitSignal = False
def quit(signum, frame):
    print('Quitting...')
    # join all remaining threads
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()

    sys.exit(signum)

signal.signal(signal.SIGTERM, quit)
signal.signal(signal.SIGINT, quit)

# get all backgrounds filenames in 'backgrounds' directory
bkgDir = os.path.join(os.getcwd(), './backgrounds')
bkgFilenames = [f for f in os.listdir(bkgDir) if os.path.isfile(os.path.join(bkgDir, f))]

# get all spheres filenames in 'spheres' directory
spheresDir = os.path.join(os.getcwd(), './spheres')
spheresFilenames = [f for f in os.listdir(spheresDir) if os.path.isfile(os.path.join(spheresDir, f))]

# create directory for dataset, if it doesn't exist
datasetDir = os.path.join(os.getcwd(), './dataset')
if not os.path.exists(datasetDir):
    os.makedirs(datasetDir)

# create HDF5 file for mean
meanHdf = h5py.File(os.path.join(datasetDir, './dataset_mean.hdf5'), 'w')

# prepare mean matrix
datasetMean = np.zeros((100, 100), dtype='double')

# generate N dataset images
N = 10000
BATCH_SIZE = 200
N_THREADS = 8

numBatches = N / BATCH_SIZE

class Worker (threading.Thread):
    def __init__(self, batch_id):
        threading.Thread.__init__(self)
        self.batch_id = batch_id

    def run(self):
        # create HDF5 file for dataset
        f = h5py.File(os.path.join(datasetDir, './dataset_batch%d.hdf5' % self.batch_id), 'w')

        batchMean = np.zeros((100, 100), dtype='double')
        x = 0
        while x < BATCH_SIZE:
            try:
                # choose a random background
                bkgListLock.acquire()
                bkgFilename = random.choice(bkgFilenames)
                bkgListLock.release()

                background = Image.open(os.path.join(bkgDir, bkgFilename))

                # create new greyscale image
                result = Image.new('L', (200, 200))

                # check that the background image is large enough
                if (background.size[0] < 100 or background.size[1] < 100):
                    bkgListLock.acquire()
                    print('Warning: background must be at least 100x100px. %s deleted.' % bkgFilename)
                    os.remove(os.path.join(bkgDir, bkgFilename))
                    bkgFilenames.remove(bkgFilename)
                    bkgListLock.release()
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
                if True:
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
                batchMean += dset[...].astype('double') / N

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
                
            x += 1

        # update global mean
        global datasetMean
        meanLock.acquire()
        datasetMean += batchMean
        meanLock.release()

        # release resources
        f.flush()
        f.close()

        # notify main thread that this worker has finished
        batchFinishCondition.acquire()

        global workersBusy
        global batchFinished
        workersBusy -= 1
        batchFinished += 1

        batchFinishCondition.notify()
        batchFinishCondition.release()

        # print progress
        if batchFinished % 10 == 0:
            print('%d / %d %f%%' % (batchFinished, numBatches, float(batchFinished) / numBatches * 100))

# synchronisation variables
bkgListLock = threading.Lock()
meanLock = threading.Lock()

batchFinishCondition = threading.Condition()

workersBusy = 0
batchFinished = 0

# start the threads
batch = 0
while batch < numBatches and not quitSignal:
    batchFinishCondition.acquire()
    while workersBusy >= N_THREADS:
        batchFinishCondition.wait()
    
    while workersBusy < N_THREADS:
        workersBusy += 1

        worker = Worker(batch)
        worker.start()
        batch += 1

    batchFinishCondition.release()

# wait for all workers to finish
batchFinishCondition.acquire()
while batchFinished < numBatches:
    batchFinishCondition.wait()
batchFinishCondition.release()

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

